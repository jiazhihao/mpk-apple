"""Agent wire contracts, including tool-result continuation, without GPU weights."""
import json
from types import SimpleNamespace

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from monolith.serve import Backend, create_app, parse_args
from monolith.serving.protocol import ChatRequest, APIError, parse_completion, responses_request
from monolith.serving.clients import client_config, endpoint, launch
from monolith.models.catalog import SERVING_MODELS, default_draft

TOOL = {'type': 'function', 'function': {'name': 'read_file', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'}}, 'required': ['path']}}}
XML = '<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>'


def make_client(text=XML):
    seen = []
    def complete(request):
        seen.append(request)
        return text, 'stop', 12, 30
    return TestClient(create_app(SimpleNamespace(complete=complete), 'local', 'test')), seen


def events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']


def test_chat_xml_tools_and_result_history():
    client, seen = make_client()
    body = {'model': 'local', 'messages': [{'role': 'user', 'content': 'Read the file'}], 'tools': [TOOL]}
    response = client.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer test'}).json()
    choice = response['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    call = choice['message']['tool_calls'][0]
    assert json.loads(call['function']['arguments']) == {'path': 'a.py', 'limit': 10}
    body['messages'] += [choice['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': 'contents'}]
    body['stream'] = True
    chunks = events(client.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer test'}))
    assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    assert chunks[1]['choices'][0]['delta']['tool_calls'][0]['index'] == 0
    messages, tools = seen[-1].template_inputs()
    assert messages[-2]['tool_calls'][0]['function']['arguments']['path'] == 'a.py'
    assert messages[-1]['role'] == 'tool' and tools == [TOOL]


@pytest.mark.parametrize('stream', [False, True])
def test_anthropic_tools(stream):
    client, seen = make_client()
    body = {'model': 'local', 'max_tokens': 100, 'stream': stream, 'system': [{'type': 'text', 'text': 'Be useful', 'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': 'Read it'}],
            'tools': [{'name': 'read_file', 'input_schema': TOOL['function']['parameters']}]}
    response = client.post('/v1/messages', json=body, headers={'x-api-key': 'test'})
    if stream:
        data = events(response)
        assert data[0]['type'] == 'message_start' and data[-1]['type'] == 'message_stop'
        assert data[0]['message']['usage']['input_tokens'] == 12
        assert data[-2]['delta']['stop_reason'] == 'tool_use'
        block = next(e['content_block'] for e in data if e['type'] == 'content_block_start')
        args = json.loads(next(e['delta']['partial_json'] for e in data if e['type'] == 'content_block_delta'))
        block['input'] = args
    else:
        data = response.json()
        assert data['stop_reason'] == 'tool_use' and data['usage']['input_tokens'] == 12
        block = data['content'][0]
    body['messages'] += [{'role': 'assistant', 'content': [block]}, {'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': block['id'], 'content': [{'type': 'text', 'text': 'file contents'}]}]}]
    assert client.post('/v1/messages', json=body, headers={'x-api-key': 'test'}).status_code == 200
    assert seen[-1].messages[-1].role == 'tool'


def test_responses_tool_events_and_continuation():
    client, seen = make_client()
    body = {'model': 'local', 'input': 'Read it', 'stream': True, 'store': False,
            'tools': [{'type': 'function', **TOOL['function']}]}
    data = events(client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}))
    assert [e['sequence_number'] for e in data] == list(range(len(data)))
    assert data[0]['type'] == 'response.created' and data[-1]['type'] == 'response.completed'
    item = data[-1]['response']['output'][0]
    assert item['type'] == 'function_call'
    body['input'] = [{'role': 'user', 'content': 'Read it'}, item, {'type': 'function_call_output', 'call_id': item['call_id'], 'output': 'contents'}]
    assert client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}).status_code == 200
    assert seen[-1].messages[-1].content == 'contents'


def test_custom_tool_preserves_multiline_input():
    patch = '*** Begin Patch\n  indented\n*** End Patch'
    client, _ = make_client('<tool_call><function=apply_patch><parameter=input>\n' + patch + '\n</parameter></function></tool_call>')
    body = {'model': 'local', 'input': 'Make the patch', 'tools': [{'type': 'custom', 'name': 'apply_patch', 'format': {'type': 'grammar', 'syntax': 'lark', 'definition': '...'}}]}
    data = client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}).json()
    assert data['output'][0]['type'] == 'custom_tool_call' and data['output'][0]['input'] == patch


@pytest.mark.parametrize('path,body', [
    ('/v1/responses', {'input': 'hi', 'previous_response_id': 'old'}),
    ('/v1/responses', {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'x'}]}]}),
    ('/v1/responses', {'input': 'hi', 'tools': [{'type': 'web_search'}]}),
    ('/v1/messages', {'max_tokens': 10, 'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}),
    ('/v1/messages', {'messages': []}),
])
def test_protocol_errors_never_reach_backend(path, body):
    client, seen = make_client()
    assert client.post(path, json={'model': 'local', **body}).status_code == 401
    assert client.post(path, json={'model': 'local', **body}, headers={'Authorization': 'Bearer test'}).status_code == 400
    assert seen == []


@pytest.mark.parametrize('protocol,path,fields', [
    ('chat', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]}),
    ('messages', '/v1/messages', {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 10}),
    ('responses', '/v1/responses', {'input': 'hi'}),
])
def test_text_stream_lifecycle(protocol, path, fields):
    client, _ = make_client('hello')
    data = events(client.post(path, json={'model': 'local', 'stream': True, **fields}, headers={'Authorization': 'Bearer test'}))
    if protocol == 'chat':
        assert data[1]['choices'][0]['delta']['content'] == 'hello'
    elif protocol == 'messages':
        assert data[2]['delta']['text'] == 'hello'
    else:
        assert next(e for e in data if e['type'] == 'response.output_text.delta')['delta'] == 'hello'
        assert data[-1]['response']['output'][0]['content'][0]['text'] == 'hello'


def test_tools_fail_closed_and_systems_merge():
    request = ChatRequest(model='local', messages=[{'role': 'system', 'content': 'a'}, {'role': 'developer', 'content': 'b'}, {'role': 'user', 'content': 'c'}], tools=[TOOL], tool_choice='required')
    assert len([m for m in request.template_inputs()[0] if m['role'] == 'system']) == 1
    for bad in ('plain text', XML.replace('read_file', 'delete_file'), XML[:-5], XML.replace('10', 'oops')):
        with pytest.raises(APIError):
            parse_completion(bad, request, 'stop')


def test_known_targets_default_to_published_nvfp4_heads(tmp_path):
    for entry in SERVING_MODELS:
        args = parse_args(['--model', entry.target])
        assert args.draft == entry.draft and args.draft_block_size is None
        assert parse_args(['--model', entry.target, '--no-draft']).draft is None
        assert parse_args(['--model', entry.target, '--draft', 'custom/head']).draft == 'custom/head'
        directory = tmp_path / entry.target.split('/')[-1]
        directory.mkdir()
        config = {'architectures': [entry.architecture], 'text_config': {'hidden_size': entry.hidden_size, 'num_hidden_layers': entry.layers}}
        (directory/'config.json').write_text(json.dumps(config))
        assert default_draft(str(directory)) == entry.draft
        config['text_config']['hidden_size'] += 1
        (directory/'config.json').write_text(json.dumps(config))
        assert default_draft(str(directory)) is None
    assert parse_args(['--model', 'other/target']).draft is None


def test_launchers_scope_configuration_and_redact_keys(monkeypatch, capsys):
    from monolith.serving import clients
    monkeypatch.setenv('LITHOS_METAL_API_KEY', 'private-test-token')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'cloud-token')
    monkeypatch.setattr(clients, 'discover', lambda url, key: {'id': 'local', 'context_window': 8192})
    for name in ('opencode', 'claude', 'codex', 'hermes', 'run'):
        args = SimpleNamespace(command=name, url='http://localhost:8000/v1', model=None, print_config=True, args=['--', 'hello'])
        assert launch(args) == 0
        output = capsys.readouterr().out
        assert 'private-test-token' not in output and 'cloud-token' not in output
        assert 'http://localhost:8000' in output
    command, env = client_config('codex', 'http://localhost:8000', 'local', 'key')
    assert 'model_providers.lithos-metal.wire_api="responses"' in command
    command, env = client_config('claude', 'http://localhost:8000', 'local', 'key')
    assert env['ANTHROPIC_API_KEY'] == 'key' and env['ANTHROPIC_AUTH_TOKEN'] == 'key'
    with pytest.raises(ValueError):
        endpoint('http://user:password@localhost:8000')


def test_launch_preserves_client_permissions_and_arguments(monkeypatch):
    from monolith.serving import clients
    launched = []
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps({'permission': {'bash': 'ask'}, 'provider': {'existing': {'name': 'Existing'}}}))
    monkeypatch.setattr(clients.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(clients.subprocess, 'call', lambda command, env: launched.append((command, env)) or 0)
    args = SimpleNamespace(command='opencode', url='http://localhost:8000', model='local', print_config=False, args=['--', 'run', 'hello'])
    assert launch(args) == 0
    command, env = launched[0]
    assert command == ['opencode', 'run', 'hello']
    config = json.loads(env['OPENCODE_CONFIG_CONTENT'])
    assert config['permission'] == {'bash': 'ask'} and 'existing' in config['provider']
    assert config['provider']['lithos-metal']['options']['baseURL'] == 'http://localhost:8000/v1'
    assert json.loads(clients.os.environ['OPENCODE_CONFIG_CONTENT'])['permission'] == {'bash': 'ask'}
    args.command = 'hermes'
    args.args = ['--', '-q', 'hello']
    launch(args)
    command, env = launched[-1]
    assert command[:4] == ['hermes', 'chat', '--provider', 'custom']
    assert env['CUSTOM_BASE_URL'] == env['OPENAI_BASE_URL'] == 'http://localhost:8000/v1'


def test_env_print_config_is_redacted(monkeypatch, capsys):
    monkeypatch.setenv('LITHOS_METAL_API_KEY', 'secret-local-key')
    launch(SimpleNamespace(command='env', url='http://localhost:8000', model='local', print_config=True, args=[]))
    assert 'secret-local-key' not in capsys.readouterr().out


def test_stream_failure_releases_generation_lock():
    backend = SimpleNamespace(complete=lambda request: (_ for _ in ()).throw(RuntimeError('sensitive path')))
    client = TestClient(create_app(backend, 'local'))
    body = {'model': 'local', 'input': 'Hi', 'stream': True}
    response = client.post('/v1/responses', json=body)
    assert events(response)[-1]['type'] == 'response.failed'
    assert 'sensitive path' not in response.text
    backend.complete = lambda request: ('Recovered', 'stop', 2, 1)
    assert client.post('/v1/responses', json={**body, 'stream': False}).status_code == 200


def test_token_count_uses_same_tool_template():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or [1, 2, 3])
    client = TestClient(create_app(SimpleNamespace(tokenizer=tokenizer), 'local'))
    response = client.post('/v1/messages/count_tokens', json={'model': 'local', 'system': 'Hello',
        'messages': [{'role': 'user', 'content': 'Read'}],
        'tools': [{'name': 'read_file', 'input_schema': TOOL['function']['parameters']}]})
    assert response.json() == {'input_tokens': 3}
    assert calls[0][1]['tools'][0]['function']['name'] == 'read_file'
    assert calls[0][1]['enable_thinking'] is False

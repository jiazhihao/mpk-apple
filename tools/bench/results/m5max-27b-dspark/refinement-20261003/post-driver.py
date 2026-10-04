"""Final architecture/precision comparisons, serialized after all screens."""
import copy,json,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'cooperative-driver.log').read_text()
    if 'DONE cooperative screen' in log:break
    if 'Traceback' in log:raise RuntimeError('alternate attention driver failed')
    time.sleep(1)
else:raise RuntimeError('prior work timeout')
def run(label,args):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
def score(x):return statistics.median(t['gpu_ms']for t in x['samples']['candidate'])
common=['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-bf16-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--reps','9','--warmup','5']
selected=json.loads((r/'selected-refined-contexts.json').read_text())
mega=copy.deepcopy(selected)
for ctx in ('4096','8192','16384'):
    if selected[ctx]['draft']['mixer'].get('mode')!='native':continue
    choices=[json.loads(l)for l in (r/('mixer-confirm-'+ctx+'.jsonl')).read_text().splitlines()]
    choices=[x for x in choices if 'samples'in x and x['config']['mixer'].get('mode')!='native']
    mega[ctx]['draft'].update(min(choices,key=score)['config'])
(r/'selected-mega-alternative.json').write_text(json.dumps(mega,indent=2))
for ctx in ('4096','8192','16384'):
    if selected[ctx]==mega[ctx]:continue
    run('native-versus-mega-'+ctx,common+['--config',str(r/'selected-refined-contexts.json'),'--config-key',ctx,'--contexts',ctx,'--out',str(r/('native-versus-mega-'+ctx+'.json')),'--compare-config',str(r/'selected-mega-alternative.json')])
# Only run another full round for an alternate attention algorithm if it wins
# the paired micro screen by at least 1.5 percent.
coop=copy.deepcopy(selected)
for ctx in ('128','32768'):
    choices=[json.loads(l)for l in (r/('mixer-cooperative-'+ctx+'.jsonl')).read_text().splitlines()]
    choices=[x for x in choices if 'samples'in x]
    if not choices:continue
    x=min(choices,key=score)
    ratio=score(x)/statistics.median(t['gpu_ms']for t in x['samples']['baseline'])
    if ratio>=.985:continue
    coop[ctx]['draft'].update(x['config'])
(r/'selected-cooperative-alternative.json').write_text(json.dumps(coop,indent=2))
for ctx in ('128','32768'):
    if coop[ctx]==selected[ctx]:continue
    args=common+['--config',str(r/'selected-cooperative-alternative.json'),'--config-key',ctx,'--contexts',ctx,'--out',str(r/('cooperative-full-'+ctx+'.json')),'--compare-config',str(r/'selected-refined-contexts.json')]
    if ctx=='128':args+=['--check-generation','--generation-tokens','128']
    run('cooperative-full-'+ctx,args)
# Compare actual target tokens and acceptance before selecting any precision
# experiment for paired timing. BF16 remains the default.
base=json.loads((r/'baseline-generation.json').read_text())['generations']
quality={}
for fmt in ('bf16','fp8_e4m3','int8','int4_affine','nvfp4'):
    p=r/('full-'+fmt+'-128.json')
    if not p.exists():continue
    d=json.loads(p.read_text());gs=d.get('generations')
    if not gs:continue
    same=all(x['prompt']==y['prompt'] and x['tokens']==y['tokens']for x,y in zip(base,gs)) and len(base)==len(gs)
    ac=[v for x in gs for v in x['accepted']]
    dec=sum(x['decode_tokens']for x in gs);ms=sum(x['decode_ms']for x in gs)
    quality[fmt]=dict(tokens_equal=same,acceptance_equal=all(x['accepted']==y['accepted']for x,y in zip(base,gs)),
                      output_tokens=sum(len(x['tokens'])for x in gs),decode_tokens=dec,decode_gpu_ms=ms,
                      gpu_ms_per_delivered_decode_token=ms/dec,mean_accepted=sum(ac)/len(ac),steps=len(ac))
(r/'generation-quality.json').write_text(json.dumps(quality,indent=2))
choices={fmt:q for fmt,q in quality.items() if fmt!='bf16' and q['tokens_equal']}
if choices:
    winner=min(choices,key=lambda f:choices[f]['gpu_ms_per_delivered_decode_token'])
    args=['.venv/bin/python','tools/bench/dspark_compare_packs.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--baseline-pack','/tmp/monolith-m5max/dspark/pack-bf16-33k','--candidate-pack','/tmp/monolith-m5max/dspark/pack-'+winner+'-33k','--baseline-config',str(r/'selected-refined-contexts.json'),'--candidate-config',str(r/('selected-'+winner+'-contexts.json')),'--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json']
    for ctx in ('128','32768'):
        run('paired-'+winner+'-'+ctx,args+['--context',ctx,'--out',str(r/('paired-'+winner+'-'+ctx+'.json'))])
print('DONE final comparisons',flush=True)

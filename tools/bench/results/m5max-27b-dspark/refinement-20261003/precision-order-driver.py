"""Preserve the previous projection arithmetic order while retaining faster crews."""
import copy,json,os,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'final-validation-driver.log').read_text()
    if 'DONE final validation'in log:break
    if 'Traceback'in log:raise RuntimeError('final validation failed')
    time.sleep(1)
else:raise RuntimeError('prior jobs timeout')
def run(label,args,env=None):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a')as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
old=json.loads((r/'selected-nvfp4-contexts.json').read_text())
new=json.loads((r/'selected-nvfp4-refined-contexts.json').read_text())
# Same tile geometry and lossless operand layout; preserve the old output
# projection's traversal (QKV retains its existing explicit override).
mix=new['128']['draft']['mixer'];mix['q_outer']=old['128']['draft']['mixer'].get('q_outer',0)
if 'ksplit'in old['128']['draft']['mixer']:mix['ksplit']=old['128']['draft']['mixer']['ksplit']
else:mix.pop('ksplit',None)
new['128']['draft']['mlp']=copy.deepcopy(old['128']['draft']['mlp'])
path=r/'selected-nvfp4-order-preserved-contexts.json';path.write_text(json.dumps(new,indent=2))
base=r/'order-baseline.json';base.write_text(json.dumps(old['128']))
configs=r/'order-check.json';configs.write_text(json.dumps([new['128']['draft']],indent=2))
for inject in ('1','8'):
    run('order-shader-validation',['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/'program-nvfp4.json'),'--baseline',str(base),'--configs',str(configs),'--out',str(r/'order-shader-validation.jsonl'),'--kind','all','--contexts','128','--inject',inject,'--reps','1','--steps','1'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
rows=[json.loads(l)for l in (r/'order-shader-validation.jsonl').read_text().splitlines()]
assert all('error'not in x for x in rows),rows
run('full-nvfp4-order-preserved-128',['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-nvfp4-keep-w1-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--config',str(path),'--compare-config',str(r/'selected-nvfp4-contexts.json'),'--config-key','128','--contexts','128','--reps','9','--warmup','5','--check-generation','--generation-tokens','128','--out',str(r/'full-nvfp4-order-preserved-128.json')])
# Bounds-check the exact final long-context recipe after its full-round gate.
base.write_text(json.dumps(new['32768']));configs.write_text('[{}]')
for inject in ('1','8'):
    run('final-nvfp4-shader-validation',['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/'program-nvfp4.json'),'--baseline',str(base),'--configs',str(configs),'--out',str(r/'final-nvfp4-shader-validation.jsonl'),'--kind','all','--contexts','32768','--inject',inject,'--reps','1','--steps','1'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
rows=[json.loads(l)for l in (r/'final-nvfp4-shader-validation.jsonl').read_text().splitlines()]
assert all('error'not in x for x in rows),rows
print('DONE arithmetic-order confirmation',flush=True)

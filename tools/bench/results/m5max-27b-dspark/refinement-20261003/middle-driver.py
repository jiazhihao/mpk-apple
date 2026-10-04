"""Dedicated middle-context fused mixer search against the selected native path."""
import copy,json,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'post-driver.log').read_text()
    if 'DONE final comparisons'in log:break
    if 'Traceback'in log:raise RuntimeError('prior comparison failed')
    time.sleep(1)
else:raise RuntimeError('prior comparison timeout')
def run(label,args):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a')as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
def score(x):return statistics.median(s['gpu_ms']for s in x['samples']['candidate'])
selected=json.loads((r/'selected-refined-contexts.json').read_text())
original=json.loads(Path('tools/bench/results/m5max-27b-dspark/selected-contexts.json').read_text())
grid=[]
for workers in (80,160,240):
 for sg in (4,8):
  for qm in (16,24,32):
   for ct in (2,4,8,16):
    for prep in (False,True):
     c=dict(workers=workers,sgs=sg,tn=32,split=True,compact=True,barrier='simd',task_barrier=False,schedule='queue',task_grain='tile',task_seed=True,task_seed_bound=True,attention_qm=qm,attention_key_tile=32,attention_chunk_tiles=ct,attention_cached_prefix=True,attention_compact_partials=True,attention_alias_scratch=True,attention_prepare=prep,attention_task_order='chunk',gemm_overrides={'0':dict(tn=16,ksplit=sg,q_outer=1,ragged_teams=True)})
     grid.append({'mixer':c})
common=['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--kind','mixer','--resume']
mega=copy.deepcopy(selected)
for ctx in ('4096','8192','16384'):
 base=r/('middle-baseline-'+ctx+'.json');base.write_text(json.dumps(selected[ctx]))
 configs=grid+[{'mixer':original[ctx]['draft']['mixer']},{'mixer':selected[ctx]['draft']['mixer']}]
 inp=r/('middle-grid-'+ctx+'.json');inp.write_text(json.dumps(configs,indent=2))
 out=r/('middle-grid-'+ctx+'.jsonl')
 run('middle-grid-'+ctx,common+['--baseline',str(base),'--configs',str(inp),'--out',str(out),'--contexts',ctx])
 rows=[json.loads(l)for l in out.read_text().splitlines()];rows=[x for x in rows if 'samples'in x]
 finalists=sorted(rows,key=score)[:8]
 inp=r/('middle-finalists-'+ctx+'.json');inp.write_text(json.dumps([x['config']for x in finalists],indent=2))
 out=r/('middle-finalists-'+ctx+'.jsonl')
 run('middle-finalists-'+ctx,common+['--baseline',str(base),'--configs',str(inp),'--out',str(out),'--contexts',ctx,'--reps','7','--steps','30','--warmup-ms','50'])
 rows=[json.loads(l)for l in out.read_text().splitlines()];rows=[x for x in rows if 'samples'in x and x['config']['mixer'].get('mode')!='native']
 if rows:mega[ctx]['draft'].update(min(rows,key=score)['config'])
(r/'selected-middle-alternative.json').write_text(json.dumps(mega,indent=2))
common=['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-bf16-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--reps','9','--warmup','5']
for ctx in ('4096','8192','16384'):
 if mega[ctx]==selected[ctx]:continue
 run('middle-paired-'+ctx,common+['--config',str(r/'selected-middle-alternative.json'),'--config-key',ctx,'--contexts',ctx,'--out',str(r/('middle-paired-'+ctx+'.json')),'--compare-config',str(r/'selected-refined-contexts.json')])
print('DONE middle-context search',flush=True)

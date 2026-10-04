import json,os,subprocess,time,copy
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for attempt in range(7200):
 log=(r/'next-driver.log').read_text()
 if 'END pack-int4_affine 0' in log:break
 if 'Traceback' in log:raise RuntimeError('prior sequential driver failed')
 time.sleep(1)
else:raise RuntimeError('prior sequential work did not finish within two hours')
def run(label,args,env=None):
 print('START',label,flush=True)
 with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 print('END',label,rc,flush=True)
 if rc:raise RuntimeError(label)
def sweep(kind,grid,contexts='128',baseline='128',reps=3,steps=5,warm=0):
 run(kind+'-'+grid,['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/('baseline-'+baseline+'.json')),'--configs',str(r/(kind+'-'+grid+'.json')),'--out',str(r/(kind+'-'+grid+'.jsonl')),'--kind',kind,'--contexts',contexts,'--resume','--reps',str(reps),'--steps',str(steps),'--warmup-ms',str(warm)])
run('wide-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_draft_program.py','tests/kernels/test_gdn_block_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
for kind,grid,contexts,baseline in [('lm_head','narrow-repaired','128','128'),('markov_chain','fusion','128','128'),('mlp','independent','128','128'),('mixer','native','128,32768','32768'),('mixer','wide','128,32768','32768')]:
 sweep(kind,grid,contexts,baseline)
# Final scheduling sweep uses the actual winning short and long geometries.
rows=[json.loads(line) for grid in ('geometry','native','wide') for line in (r/('mixer-'+grid+'.jsonl')).read_text().splitlines()]
configs=[]
for ctx in (128,32768):
 choices=sorted([x for x in rows if x.get('context')==ctx and 'best_ms' in x],key=lambda x:x['best_ms']['candidate'])
 b=copy.deepcopy(next(x['config']['mixer'] for x in choices if x['config']['mixer'].get('mode')!='native'))
 for order in ('head','chunk'):
  for tiles in (1,2,4,8):
   for batch in (1,2,4):configs.append({'mixer':dict(b,attention_task_order=order,attention_task_tiles=tiles,task_batch=batch)})
 for sg in (1,2,4,8):
  if sg>b['sgs']:continue
  for unroll in (1,2,4,8,16,32):configs.append({'mixer':dict(b,merge_sgs=sg,merge_unroll=unroll)})
 for extra in ({'cache_external_inputs':True},{'cache_external_inputs':'const'},{'barrier':'leader'},{'barrier':'serial'},{'task_barrier':True},{'schedule':'stages','task_seed':False,'task_seed_bound':False}):configs.append({'mixer':dict(b,**extra)})
configs=list({json.dumps(x,sort_keys=True):x for x in configs}.values())
(r/'mixer-scheduling.json').write_text(json.dumps(configs,indent=2))
sweep('mixer','scheduling','128,32768','32768')
# Retest the leading projection configurations with longer warmup and batches.
for kind,grids in [('markov',('geometry','activation')),('feature',('geometry',)),('context_kv',('geometry',)),('lm_head',('geometry','layout','narrow-repaired')),('mlp',('geometry','independent'))]:
 rows=[json.loads(line) for grid in grids for line in (r/(kind+'-'+grid+'.jsonl')).read_text().splitlines()]
 good=sorted([x for x in rows if 'best_ms' in x],key=lambda x:x['best_ms']['candidate'])
 cfgs=list({json.dumps(x['config'],sort_keys=True):x['config'] for x in good}.values())[:10]
 (r/(kind+'-finalists.json')).write_text(json.dumps(cfgs,indent=2))
 sweep(kind,'finalists',reps=7,steps=50,warm=50)
for kind in ('feature_scalar','context_kv_scalar'):
 args=['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/'baseline-128.json'),'--configs',str(r/(kind+'-geometry.json')),'--out',str(r/(kind+'-geometry.jsonl')),'--kind',kind,'--inject','1','--resume']
 run(kind+'-geometry',args)
 rows=[json.loads(line) for line in (r/(kind+'-geometry.jsonl')).read_text().splitlines()]
 good=sorted([x for x in rows if 'best_ms' in x],key=lambda x:x['best_ms']['candidate'])
 cfgs=list({json.dumps(x['config'],sort_keys=True):x['config'] for x in good}.values())[:10]
 (r/(kind+'-finalists.json')).write_text(json.dumps(cfgs,indent=2))
 args[args.index('--configs')+1]=str(r/(kind+'-finalists.json'))
 args[args.index('--out')+1]=str(r/(kind+'-finalists.jsonl'))
 run(kind+'-finalists',args+['--warmup-ms','50','--reps','7','--steps','50'])
print('DONE refinement',flush=True)

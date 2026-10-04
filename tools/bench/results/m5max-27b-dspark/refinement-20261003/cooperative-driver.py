"""Last alternate attention algorithm, serialized after queued measurements."""
import copy,json,os,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'quant-driver.log').read_text()
    if 'DONE precision experiments' in log:break
    if 'Traceback' in log:raise RuntimeError('precision driver failed')
    time.sleep(1)
else:raise RuntimeError('prior work timeout')
def run(label,args,env=None):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
run('cooperative-shader-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_draft_program.py','tests/kernels/test_attention_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
selected=json.loads((r/'selected-refined-contexts.json').read_text())
configs=[]
for workers in (80,160,240):
    for sgs in (4,8):
        for qm in (8,16,32):
            for kn in (32,64,128):
                if sgs*qm*kn*2>32000:continue
                c=dict(workers=workers,sgs=sgs,tn=32,compact=True,split=True,barrier='simd',task_barrier=False,schedule='queue',task_grain='tile',task_seed=True,task_seed_bound=True,attention_prepare=True,attention_style='cooperative',attention_qm=qm,attention_key_tile=kn,attention_chunk_tiles=1,attention_compact_partials=True)
                for cached in (False,True):configs.append({'mixer':dict(c,attention_cached_prefix=cached)})
(r/'mixer-cooperative.json').write_text(json.dumps(configs,indent=2))
for ctx in (128,32768):
    (r/('cooperative-baseline-'+str(ctx)+'.json')).write_text(json.dumps(selected[str(ctx)],indent=2))
    run('mixer-cooperative-'+str(ctx),['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/('cooperative-baseline-'+str(ctx)+'.json')),'--configs',str(r/'mixer-cooperative.json'),'--out',str(r/('mixer-cooperative-'+str(ctx)+'.jsonl')),'--kind','mixer','--contexts',str(ctx),'--resume','--warmup-ms','30','--reps','5','--steps','15'])
print('DONE cooperative screen',flush=True)

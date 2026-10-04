"""Final shared-compiler regressions and cache-option confirmation."""
import copy,json,os,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'precision-refine-driver.log').read_text()
    if 'DONE precision refinement'in log:break
    if 'Traceback'in log:raise RuntimeError('precision refinement failed')
    time.sleep(1)
else:raise RuntimeError('prior jobs timeout')
def run(label,args,env=None):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a')as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
run('final-target-shader-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_gdn_block_static.py','tests/kernels/test_mlp_block_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
selected=json.loads((r/'selected-refined-contexts.json').read_text())
for ctx in ('128','32768'):
    stem='cache-final-'+ctx;configs=[]
    for cache in (False,True,'const'):
        c=copy.deepcopy(selected[ctx]['draft']['mixer']);c['cache_external_inputs']=cache;configs.append({'mixer':c})
    base=r/(stem+'-baseline.json');base.write_text(json.dumps(selected[ctx]))
    inp=r/(stem+'.json');inp.write_text(json.dumps(configs,indent=2))
    run(stem,['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(base),'--configs',str(inp),'--out',str(r/(stem+'.jsonl')),'--kind','mixer','--contexts',ctx,'--reps','9','--steps','50','--warmup-ms','100'])
print('DONE final validation',flush=True)

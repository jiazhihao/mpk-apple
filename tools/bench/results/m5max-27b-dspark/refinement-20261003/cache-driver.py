"""Retest the repaired immutable-input variants after precision jobs finish."""
import json,os,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'repack-driver.log').read_text()
    if 'DONE supported precision experiments' in log:break
    if 'Traceback' in log:raise RuntimeError('precision jobs failed')
    time.sleep(1)
else:raise RuntimeError('precision jobs timeout')
def run(label,args,env=None):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a')as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
run('cached-inputs-shader-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_draft_program.py','tests/kernels/test_attention_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
run('cached-inputs-repaired',['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/'baseline-32768.json'),'--configs',str(r/'cached-inputs-repaired.json'),'--out',str(r/'cached-inputs-repaired.jsonl'),'--kind','mixer','--contexts','128,32768','--reps','7','--steps','30','--warmup-ms','50'])
print('DONE cached-input confirmation',flush=True)

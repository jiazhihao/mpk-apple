import json,os,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for attempt in range(3600):
 if 'END context_kv 0' in (r/'sweep-driver.log').read_text():break
 time.sleep(1)
else:raise RuntimeError('previous sequential sweep did not complete within one hour')
def run(label,args,env=None):
 print('START',label,flush=True)
 with (r/(label+'.log')).open('a') as log:
  result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env)
 print('END',label,result.returncode,flush=True)
 if result.returncode:raise RuntimeError(label)
run('prepared-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_draft_program.py','tests/kernels/test_attention_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
for kind,grid,contexts,baseline in [('markov','activation','128','128'),('markov_chain','fusion','128','128'),('mixer','geometry','128,32768','32768'),('lm_head','layout','128','128')]:
 run(kind+'-'+grid,['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/('baseline-'+baseline+'.json')),'--configs',str(r/(kind+'-'+grid+'.json')),'--out',str(r/(kind+'-'+grid+'.jsonl')),'--kind',kind,'--contexts',contexts,'--resume'])
for fmt in ('fp8_e4m3','nvfp4','int8','int4_affine'):
 out=Path('/tmp/monolith-m5max/dspark')/('pack-'+fmt+'-33k')
 if (out/'manifest.json').exists():continue
 run('pack-'+fmt,['.venv/bin/python','tools/pack_weights.py','--model','/tmp/monolith-models/Qwen3.8-27B-DSpark','--out',str(out),'--drafter-kind','dspark','--max-context','33024','--scale-placement','block','--quantize',fmt])

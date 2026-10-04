import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from monolith.core.profile import COST_FORMAT, load_profile
from monolith.generate import load_session
fmt=sys.argv[1]
quant_pack=sys.argv[2] if len(sys.argv)>2 else '/tmp/monolith-m5max/dspark/pack-'+fmt+'-33k'
p=load_profile('profiles/apple-m5-max-40c.json')
p.accelerator_min_t['bf16']=2
p.accelerator_min_t[COST_FORMAT.get(fmt,fmt)]=2
s=load_session('/tmp/monolith-models/Qwen3.8-27B-NVFP4','/tmp/monolith-m5max/attention-tasks/pack-33k',profile=p,max_context=33024,autotune=False,eos=-1,drafter_dir='/tmp/monolith-models/Qwen3.8-27B-DSpark',drafter_pack=quant_pack,drafter_options={'attention':'mma'},verify='fixed',verify_length=7,prefill_chunk_size=128,prefill_attention='v3',accelerator='on',decoder_kernel_config=None)
s._compile(8,dynamic=True,prefill=False).save('/tmp/monolith-m5max/dspark/exhaustive/program-'+fmt+'.json')
print('COMPILED',fmt,flush=True)
# Exercise the whole draft on this pack under Metal shader bounds validation
# before its performance sweep. Instrumented timings are never reported.
import os,subprocess
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
base=json.loads((r/'selected-refined-contexts.json').read_text())['128']
base['draft'].pop('markov_fusion',None)
(r/('validation-'+fmt+'-baseline.json')).write_text(json.dumps(base))
(r/('validation-'+fmt+'-configs.json')).write_text('[{}]')
out=r/('validation-'+fmt+'.jsonl')
rc=subprocess.run(['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/('program-'+fmt+'.json')),'--baseline',str(r/('validation-'+fmt+'-baseline.json')),'--configs',str(r/('validation-'+fmt+'-configs.json')),'--out',str(out),'--kind','all','--contexts','128','--reps','1','--steps','1'],env=dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1')).returncode
assert rc==0
last=json.loads(out.read_text().splitlines()[-1])
assert 'error' not in last,last.get('error')
print('VALIDATED',fmt,flush=True)

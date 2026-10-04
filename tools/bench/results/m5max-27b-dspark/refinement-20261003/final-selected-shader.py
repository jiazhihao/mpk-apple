import json,os,subprocess
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
selected=json.loads((r/'selected-nvfp4-refined-contexts.json').read_text())
configs=r/'final-selected-identity.json';configs.write_text('[{}]')
out=r/'final-selected-nvfp4-shader.jsonl'
for ctx in ('128','32768'):
    base=r/('final-selected-'+ctx+'.json');base.write_text(json.dumps(selected[ctx]))
    for inject in ('1','8'):
        args=['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/'program-nvfp4.json'),'--baseline',str(base),'--configs',str(configs),'--out',str(out),'--kind','all','--contexts',ctx,'--inject',inject,'--reps','1','--steps','1']
        subprocess.run(args,check=True,env=dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
rows=[json.loads(l)for l in out.read_text().splitlines()]
assert len(rows)==4 and all('error'not in x for x in rows),rows
print('PASS four exact selected NVFP4 shader checks',flush=True)

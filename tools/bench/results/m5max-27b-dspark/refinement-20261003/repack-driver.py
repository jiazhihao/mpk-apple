"""Supported precision experiments after the BF16 architecture comparisons."""
import json,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'middle-driver.log').read_text()
    if 'DONE middle-context search'in log:break
    if 'Traceback'in log:raise RuntimeError('architecture comparison failed')
    time.sleep(1)
else:raise RuntimeError('prior work timeout')
def run(label,args):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
for fmt in ('fp8_e4m3','int8','int4_affine','nvfp4'):
    dest=Path('/tmp/monolith-m5max/dspark')/('pack-'+fmt+'-keep-w1-33k')
    if (dest/'manifest.json').exists():continue
    run('pack-'+fmt+'-keep-w1',['.venv/bin/python','tools/pack_weights.py','--model','/tmp/monolith-models/Qwen3.8-27B-DSpark','--out',str(dest),'--drafter-kind','dspark','--max-context','33024','--scale-placement','block','--quantize',fmt,'--quantize-keep','markov_w1'])
run('quant-keep-w1-driver',['.venv/bin/python','-u',str(r/'quant-keep-w1-driver.py')])
base=json.loads((r/'baseline-generation.json').read_text())['generations']
quality={}
for fmt in ('bf16','fp8_e4m3','int8','int4_affine','nvfp4'):
    p=r/('full-'+fmt+'-128.json')
    if not p.exists():continue
    gs=json.loads(p.read_text()).get('generations')
    if not gs:continue
    same=len(gs)==len(base) and all(x['prompt']==y['prompt'] and x['tokens']==y['tokens']for x,y in zip(base,gs))
    ac=[v for x in gs for v in x['accepted']];dec=sum(x['decode_tokens']for x in gs);ms=sum(x['decode_ms']for x in gs)
    quality[fmt]=dict(tokens_equal=same,acceptance_equal=all(x['accepted']==y['accepted']for x,y in zip(base,gs)),
                      output_tokens=sum(len(x['tokens'])for x in gs),decode_tokens=dec,decode_gpu_ms=ms,
                      gpu_ms_per_delivered_decode_token=ms/dec,mean_accepted=sum(ac)/len(ac),steps=len(ac))
(r/'generation-quality.json').write_text(json.dumps(quality,indent=2))
choices={fmt:q for fmt,q in quality.items()if fmt!='bf16'and q['tokens_equal']}
if choices:
    winner=min(choices,key=lambda f:choices[f]['gpu_ms_per_delivered_decode_token'])
    args=['.venv/bin/python','tools/bench/dspark_compare_packs.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--baseline-pack','/tmp/monolith-m5max/dspark/pack-bf16-33k','--candidate-pack','/tmp/monolith-m5max/dspark/pack-'+winner+'-keep-w1-33k','--baseline-config',str(r/'selected-refined-contexts.json'),'--candidate-config',str(r/('selected-'+winner+'-contexts.json')),'--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json']
    for ctx in ('128','32768'):
        run('paired-'+winner+'-'+ctx,args+['--context',ctx,'--out',str(r/('paired-'+winner+'-'+ctx+'.json'))])
print('DONE supported precision experiments',flush=True)

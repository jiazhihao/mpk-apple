"""Exploratory precision experiments; never overwrite the original BF16 pack."""
import copy,json,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'confirm-driver.log').read_text()
    if 'DONE BF16 confirmation' in log:break
    if 'Traceback' in log:raise RuntimeError('confirmation driver failed')
    time.sleep(1)
else:raise RuntimeError('confirmation timeout')
def run(label,args):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
    print('END',label,rc,flush=True)
    return rc
selected=json.loads((r/'selected-refined-contexts.json').read_text())
for fmt in ('fp8_e4m3','int8','int4_affine','nvfp4'):
    if run('compile-'+fmt,['.venv/bin/python',str(r/'compile-quant.py'),fmt]):continue
    baseline=copy.deepcopy(selected['128'])
    baseline['accelerator_min_t']={fmt:2}
    # Original scalar W2 geometry is a neutral reference for each new format.
    baseline['draft'].pop('markov_fusion',None)
    (r/('baseline-'+fmt+'.json')).write_text(json.dumps(baseline,indent=2))
    configurations={}
    for kind in ('mlp','feature','context_kv','markov'):
        cfgs=[]
        if kind=='markov':
            for w in (80,160,320):
                for sg in (4,8,16):
                    for rg in (1,2,4,8):
                        cfgs.append({kind:dict(workers=w,sgs=sg,rg=rg,rsplit=1,x_preconvert=True,x_hoist=True)})
        else:
            for w in (80,160):
                for sg in (4,8):
                    for tn in (16,32):
                        for tk in (64,128):
                            for ks in (1,4):
                                c=dict(workers=w,sgs=sg,tn=tn,compact=True,mode='native',staged_tk=tk,ksplit=ks,ragged_teams=True)
                                if fmt=='nvfp4':c.update(nvfp4_layout='tile',nvfp4_tile_block=32,q_outer=1)
                                if fmt=='fp8_e4m3':c.update(fp8_layout='tile',fp8_tile_block=8,fp8_decode='half',q_outer=0)
                                cfgs.append({kind:c})
        grid=r/(fmt+'-'+kind+'.json');out=grid.with_suffix('.jsonl')
        grid.write_text(json.dumps(cfgs,indent=2))
        args=['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/('program-'+fmt+'.json')),'--baseline',str(r/('baseline-'+fmt+'.json')),'--configs',str(grid),'--out',str(out),'--kind',kind,'--resume']
        if run(fmt+'-'+kind,args):continue
        a=[json.loads(l) for l in out.read_text().splitlines()];a=[x for x in a if 'best_ms'in x]
        if a:configurations.update(min(a,key=lambda x:statistics.median(t['gpu_ms']for t in x['samples']['candidate']))['config'])
    configs=copy.deepcopy(selected)
    for cfg in configs.values():
        cfg['accelerator_min_t']={fmt:2}
        cfg['draft'].pop('markov_fusion',None)
        cfg['draft'].update(copy.deepcopy(configurations))
    path=r/('selected-'+fmt+'-contexts.json');path.write_text(json.dumps(configs,indent=2))
    common=['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-'+fmt+'-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--config',str(path),'--reps','7','--warmup','4']
    run('full-'+fmt+'-128',common+['--config-key','128','--contexts','128','--out',str(r/('full-'+fmt+'-128.json')),'--check-generation','--generation-tokens','128'])
    run('full-'+fmt+'-32768',common+['--config-key','32768','--contexts','32768','--out',str(r/('full-'+fmt+'-32768.json'))])
print('DONE precision experiments',flush=True)

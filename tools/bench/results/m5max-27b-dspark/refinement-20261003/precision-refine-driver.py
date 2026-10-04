"""Final geometry pass for the fastest token-matching experimental precision."""
import copy,json,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(21600):
    log=(r/'cache-driver.log').read_text()
    if 'DONE cached-input confirmation'in log:break
    if 'Traceback'in log:raise RuntimeError('preceding validation failed')
    time.sleep(1)
else:raise RuntimeError('preceding jobs timeout')
def run(label,args):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a')as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
quality=json.loads((r/'generation-quality.json').read_text())
choices={k:v for k,v in quality.items()if k!='bf16'and v['tokens_equal']}
if not choices:
    print('DONE precision refinement: no eligible format',flush=True);raise SystemExit(0)
fmt=min(choices,key=lambda k:choices[k]['gpu_ms_per_delivered_decode_token'])
print('FORMAT',fmt,flush=True)
original=r/('selected-'+fmt+'-contexts.json');selected=json.loads(original.read_text())
refined=copy.deepcopy(selected)
def score(x):return statistics.median(y['gpu_ms']for y in x['samples']['candidate'])
def sweep(kind,ctx,configs,inject=8):
    stem='precision-refine-'+fmt+'-'+kind+'-'+str(ctx)
    base=r/(stem+'-baseline.json');base.write_text(json.dumps(selected[str(ctx)]))
    inp=r/(stem+'.json');inp.write_text(json.dumps(configs,indent=2))
    out=r/(stem+'.jsonl')
    args=['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program',str(r/('program-'+fmt+'.json')),'--baseline',str(base),'--configs',str(inp),'--out',str(out),'--kind',kind,'--contexts',str(ctx),'--inject',str(inject),'--resume']
    run(stem,args)
    rows=[json.loads(l)for l in out.read_text().splitlines()];rows=[x for x in rows if 'samples'in x]
    if not rows:return {}
    finalists=sorted(rows,key=score)[:8]
    inp=r/(stem+'-finalists.json');inp.write_text(json.dumps([x['config']for x in finalists],indent=2))
    out=r/(stem+'-finalists.jsonl')
    args[args.index('--configs')+1]=str(inp);args[args.index('--out')+1]=str(out)
    run(stem+'-finalists',args+['--reps','7','--steps','30','--warmup-ms','50'])
    rows=[json.loads(l)for l in out.read_text().splitlines()];rows=[x for x in rows if 'samples'in x]
    return min(rows,key=score)['config'] if rows else {}
for ctx in (128,32768):
    b=selected[str(ctx)]['draft']['mixer'];cfgs=[]
    for w in (80,160,240):
     for sg in (4,8):
      for tn in (16,32):
       for qo in (0,1):
        for block in ((8,32)if fmt=='fp8_e4m3'else (16,32)if fmt=='nvfp4'else (None,)):
         c=copy.deepcopy(b);c.update(workers=w,sgs=sg,tn=tn,ksplit=sg,q_outer=qo)
         c['gemm_overrides']={'0':dict(tn=16,ksplit=sg,q_outer=qo,ragged_teams=True)}
         if fmt=='fp8_e4m3':c.update(fp8_layout='tile',fp8_tile_block=block,fp8_decode='half')
         if fmt=='nvfp4':c.update(nvfp4_layout='tile',nvfp4_tile_block=block)
         cfgs.append({'mixer':c})
    cfgs.append({'mixer':copy.deepcopy(b)})
    refined[str(ctx)]['draft'].update(sweep('mixer',ctx,cfgs))
for kind in ('markov','feature_scalar','context_kv_scalar'):
    if kind=='markov':cfgs=json.loads((r/'markov-activation.json').read_text())
    else:
        cfgs=[]
        for w in (80,160,320):
         for sg in (8,16):
          for rg in (1,2,4):
           for rs in (1,4,8):
            if (16//rs)%rg:continue
            cfgs.append({kind:dict(workers=w,sgs=sg,rg=rg,rsplit=rs,x_preconvert=True,x_hoist=False)})
    changed=sweep(kind,128,cfgs,1 if kind.endswith('_scalar')else 8)
    for c in refined.values():c['draft'].update(copy.deepcopy(changed))
# Vary gate/up and down independently around this format's own winner.
base_mlp=selected['128']['draft']['mlp'];cfgs=[]
for item in json.loads((r/'mlp-independent.json').read_text()):
    c=copy.deepcopy(base_mlp);c['gemm_overrides']=item['mlp']['gemm_overrides'];cfgs.append({'mlp':c})
for groups in (80,160,320,640,1280):
    c=copy.deepcopy(base_mlp);c['gemm_overrides']={'1':dict(groups=groups)};cfgs.append({'mlp':c})
for block in ((8,16,32,64)if fmt=='fp8_e4m3'else (16,32,64,128)if fmt=='nvfp4'else (None,)):
 for qo in (0,1):
    c=copy.deepcopy(base_mlp);c['q_outer']=qo
    if fmt=='fp8_e4m3':c['fp8_tile_block']=block
    if fmt=='nvfp4':c['nvfp4_tile_block']=block
    cfgs.append({'mlp':c})
cfgs.append({'mlp':copy.deepcopy(base_mlp)})
changed=sweep('mlp',128,cfgs)
for c in refined.values():c['draft'].update(copy.deepcopy(changed))
path=r/('selected-'+fmt+'-refined-contexts.json');path.write_text(json.dumps(refined,indent=2))
common=['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-'+fmt+'-keep-w1-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--config',str(path),'--compare-config',str(original),'--reps','9','--warmup','5']
for ctx in ('128','32768'):
    extra=['--check-generation','--generation-tokens','128']if ctx=='128'else[]
    run('full-'+fmt+'-refined-'+ctx,common+['--config-key',ctx,'--contexts',ctx,'--out',str(r/('full-'+fmt+'-refined-'+ctx+'.json'))]+extra)
# Keep the complete results; promotion is a model-visible decision based on
# paired ranges, generation acceptance and the format's correctness checks.
print('DONE precision refinement',fmt,flush=True)

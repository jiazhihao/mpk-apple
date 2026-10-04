"""Serial confirmation after the broad screen. No concurrent GPU jobs."""
import copy,json,os,statistics,subprocess,time
from pathlib import Path
r=Path('/tmp/monolith-m5max/dspark/exhaustive')
for _ in range(14400):
    log=(r/'final-driver.log').read_text()
    if 'DONE refinement' in log:break
    if 'Traceback' in log:raise RuntimeError('screen driver failed')
    time.sleep(1)
else:raise RuntimeError('screen timeout')
def run(label,args,env=None):
    print('START',label,flush=True)
    with (r/(label+'.log')).open('a') as f:rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('END',label,rc,flush=True)
    if rc:raise RuntimeError(label)
def rows(patterns):
    return [json.loads(l) for p in patterns for l in p.read_text().splitlines() if l.strip()]
def good(a):return [x for x in a if 'best_ms' in x]
def score(x):return statistics.median(z['gpu_ms'] for z in x['samples']['candidate'])
def unique(a):return list({json.dumps(x,sort_keys=True):x for x in a}.values())
def sweep(kind,name,configs,contexts='128',baseline='128',reps=7,steps=30,inject=8):
    (r/(name+'.json')).write_text(json.dumps(configs,indent=2))
    run(name,['.venv/bin/python','tools/bench/dspark_kernel_tune.py','--program','/tmp/monolith-m5max/dspark/mma-tensor-program.json','--baseline',str(r/('baseline-'+baseline+'.json')),'--configs',str(r/(name+'.json')),'--out',str(r/(name+'.jsonl')),'--kind',kind,'--contexts',contexts,'--resume','--reps',str(reps),'--steps',str(steps),'--warmup-ms','50','--inject',str(inject)])
if not (r/'narrow-shader-validation.passed').exists():
    run('narrow-shader-validation',['.venv/bin/python','-m','pytest','-q','tests/kernels/test_mlp_block_static.py'],dict(os.environ,MTL_SHADER_VALIDATION='1',MTL_SHADER_VALIDATION_REPORT_TO_STDERR='1'))
    (r/'narrow-shader-validation.passed').write_text('96 passed with shader validation\n')
# Combine the independent gate/up and down winners, retaining the old joint recipe.
old=json.loads((r/'baseline-128.json').read_text())
a=good(rows([r/'mlp-independent.jsonl']))
configs=[{}]
for stage in ('0','1'):
    winners=sorted([x for x in a if set(x['config']['mlp'].get('gemm_overrides',{}))=={stage}],key=score)[:3]
    configs=[dict(c,**{stage:copy.deepcopy(x['config']['mlp']['gemm_overrides'][stage])}) for c in configs for x in winners]
sweep('mlp','mlp-coupled',[{'mlp':dict(old['draft']['mlp'],gemm_overrides=c)} for c in configs])
# Whole Markov chain: compare fused candidates with tuned ordinary W2.
a=good(rows([r/'markov-finalists.jsonl']))
configs=[x['config'] for x in sorted(a,key=score)[:3]]
a=good(rows([r/'markov_chain-fusion.jsonl']))
configs += [x['config'] for x in sorted(a,key=score)[:4]]
sweep('markov_chain','markov-chain-confirm',unique(configs))
# Cover projection-specific choices that the shared attention sweep held fixed.
a=good(rows([r/('mixer-'+n+'.jsonl') for n in ('geometry','wide','scheduling')]))
configs=[]
for ctx in (128,32768):
    recipe=copy.deepcopy(min([x for x in a if x['context']==ctx],key=score)['config']['mixer'])
    for stage in ('0','1'):
        for tn in (16,32):
            for ks in (1,2,4,8):
                if ks>recipe['sgs']:continue
                for outer in (0,1):
                    configs.append({'mixer':dict(recipe,gemm_overrides={stage:dict(tn=tn,ksplit=ks,q_outer=outer,ragged_teams=True)})})
    for extra in ({'short_decode':True},{'narrow_weights':True},{'unroll':True},
                  {'k_unroll':2},{'k_unroll':4},{'k_unroll':8},{'scalar_sgs':1},{'scalar_sgs':2},
                  {'restrict_weights':True},{'arrival':'store'},{'arrival':'register'},
                  {'flag_stride':4},{'flag_stride':16},{'poll_sgs':2}):
        configs.append({'mixer':dict(recipe,**extra)})
sweep('mixer','mixer-projection-refine',unique(configs),'128,32768','32768',reps=3,steps=5)
# Retest at all requested contexts with each context's previous production recipe.
a=good(rows([r/('mixer-'+n+'.jsonl') for n in ('geometry','wide','native','scheduling','projection-refine')]))
configs=[]
for ctx in (128,32768):
    for native in (False,True):
        choices=[x for x in a if x['context']==ctx and (x['config']['mixer'].get('mode')=='native')==native]
        configs += [x['config'] for x in sorted(choices,key=score)[:(5 if not native else 1)]]
configs=unique(configs)
for ctx in (128,4096,8192,16384,32768):
    source=json.loads(Path('tools/bench/results/m5max-27b-dspark/selected-contexts.json').read_text())[str(ctx)]
    (r/('baseline-'+str(ctx)+'.json')).write_text(json.dumps(source,indent=2))
    sweep('mixer','mixer-confirm-'+str(ctx),configs,str(ctx),str(ctx),steps=20)
# Assemble the best ordinary-precision candidate, with no precision conversion.
selected=json.loads(Path('tools/bench/results/m5max-27b-dspark/selected-contexts.json').read_text())
shared={}
for kind in ('feature','context_kv','lm_head','feature_scalar','context_kv_scalar'):
    best=min(good(rows([r/(kind+'-finalists.jsonl')])),key=score)
    shared.update(best['config'])
mlp=min(good(rows([r/'mlp-finalists.jsonl',r/'mlp-coupled.jsonl'])),key=score)
shared.update(mlp['config'])
markov=min(good(rows([r/'markov-chain-confirm.jsonl'])),key=score)
shared.update(markov['config'])
for ctx,cfg in selected.items():
    cfg['draft'].update(copy.deepcopy(shared))
    mixer=min(good(rows([r/('mixer-confirm-'+ctx+'.jsonl')])),key=score)
    cfg['draft'].update(mixer['config'])
(r/'selected-refined-contexts.json').write_text(json.dumps(selected,indent=2))
common=['.venv/bin/python','tools/bench/dspark_round_latency.py','--model','/tmp/monolith-models/Qwen3.8-27B-NVFP4','--pack','/tmp/monolith-m5max/attention-tasks/pack-33k','--drafter','/tmp/monolith-models/Qwen3.8-27B-DSpark','--drafter-pack','/tmp/monolith-m5max/dspark/pack-bf16-33k','--profile','profiles/apple-m5-max-40c.json','--inputs','tools/bench/results/m5max-27b-dspark/inputs.json','--reps','9','--warmup','5']
for ctx in selected:
    path=r/('full-bf16-'+ctx+'.json')
    if path.exists() and (ctx!='128' or 'generations' in json.loads(path.read_text())):continue
    args=common+['--config',str(r/'selected-refined-contexts.json'),'--config-key',ctx,'--contexts',ctx,'--out',str(path),'--compare-config','tools/bench/results/m5max-27b-dspark/selected-contexts.json']
    if ctx=='128':args+=['--check-generation','--generation-tokens','128','--profile-draft']
    if ctx=='32768':args+=['--profile-draft']
    run('full-bf16-'+ctx,args)
run('baseline-generation',common+['--config','tools/bench/results/m5max-27b-dspark/selected-contexts.json','--config-key','128','--contexts','128','--out',str(r/'baseline-generation.json'),'--check-generation','--generation-tokens','128'])
print('DONE BF16 confirmation',flush=True)

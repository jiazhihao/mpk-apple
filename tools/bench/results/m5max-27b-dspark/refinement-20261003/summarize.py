"""Rebuild trial coverage and compact statistics from retained result files."""
import collections,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
coverage={}
for p in sorted(r.glob('*.jsonl')):
    rows=[json.loads(line)for line in p.read_text().splitlines()if line.strip()]
    failures=[x for x in rows if 'error'in x]
    coverage[p.name]={'trials':len(rows),'passed':sum('samples'in x for x in rows),'rejected':len(failures),
                     'contexts':sorted(set(x.get('context')for x in rows if 'context'in x)),
                     'kinds':sorted(set(x.get('kind')for x in rows if 'kind'in x)),
                     'failure_first_lines':dict(collections.Counter(x['error'].splitlines()[0]for x in failures))}
primary={}
for p in sorted(r.glob('full-bf16-*.json')):
    d=json.loads(p.read_text())
    for c in d['contexts']:
        q=c.get('recipe_comparison')
        if q is None:continue
        out={}
        for field,metric,label in [('pairs','gpu_ms','full'),('stages','draft','draft')]:
            out[label]={side:{'min':min(vals),'median':statistics.median(vals),'max':max(vals),'samples':len(vals)}
                        for side in ['baseline','candidate']for vals in [[x[side][metric]for x in q[field]]]}
        out.update(next_drafts_equal=q['next_drafts_equal'],split_tokens_equal=c['split_bit_exact'],dispatches=c['dispatches'])
        primary[str(c['context'])]=out
base=json.loads((r/'baseline-generation.json').read_text())['generations']
generation={}
for p in sorted(r.glob('full-*-128.json')):
    d=json.loads(p.read_text());a=d.get('generations')
    if not a:continue
    same=len(a)==len(base)and all(b['prompt']==x['prompt']and b['tokens']==x['tokens']for b,x in zip(base,a))
    ac=[v for x in a for v in x['accepted']];ms=sum(x['decode_ms']for x in a);n=sum(x['decode_tokens']for x in a)
    generation[p.name]={'tokens_equal':same,'decode_tokens':n,'gpu_decode_ms':ms,'gpu_ms_per_decode_token':ms/n,
                        'rounds':len(ac),'accepted_proposals':sum(ac),'mean_accepted':sum(ac)/len(ac)}
out={'trial_count':sum(x['trials']for x in coverage.values()),'passed_trials':sum(x['passed']for x in coverage.values()),
     'rejected_trials':sum(x['rejected']for x in coverage.values()),'coverage':coverage,'bf16_paired':primary,'generation':generation,
     'interpretation':'Trials include repeated confirmations and rejected configurations. Hot microbenchmarks shortlist; paired real-prefill rounds and generation determine final selection.'}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['trial_count'],'recorded trials;',out['rejected_trials'],'rejected')

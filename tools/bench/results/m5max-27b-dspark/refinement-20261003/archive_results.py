"""Archive completed evidence and reproducibility hashes, excluding weight blobs."""
import datetime,hashlib,json,platform,shutil,subprocess
from pathlib import Path
root=Path.cwd();r=Path('/tmp/monolith-m5max/dspark/exhaustive')
d=root/'tools/bench/results/m5max-27b-dspark/refinement-20261003'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
files=[]
for folder,pattern in [('monolith','*.py'),('kernels','*.metal'),('tests/contract','*.py'),('tests/kernels','*.py'),('tools/bench','*.py')]:
    files.extend((root/folder).rglob(pattern))
files.extend([root/'profiles/apple-m5-max-40c.json',root/'tools/pack_weights.py'])
source={str(p.relative_to(root)):digest(p)for p in sorted(set(files)) if 'tools/bench/results/' not in str(p.relative_to(root))}
packs={}
for fmt in ('bf16','fp8_e4m3','int8','int4_affine','nvfp4'):
    p=Path('/tmp/monolith-m5max/dspark')/('pack-bf16-33k'if fmt=='bf16'else 'pack-'+fmt+'-keep-w1-33k')
    m=p/'manifest.json';data=json.loads(m.read_text());weight=p/data['pack']
    packs[fmt]={'path':str(p),'manifest_sha256':digest(m),'weight_sha256':digest(weight),'nbytes':weight.stat().st_size,'requantize':data.get('quantize')}
    print('HASHED',fmt,flush=True)
m=Path('/tmp/monolith-m5max/attention-tasks/pack-33k/manifest.json');data=json.loads(m.read_text());weight=m.parent/data['pack']
packs['target']={'path':str(m.parent),'manifest_sha256':digest(m),'weight_sha256':digest(weight),'nbytes':weight.stat().st_size}
(r/'pack-manifest-target.json').write_text(m.read_text())
meta={'measured_date':'2026-10-03','archived_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
      'uncommitted_worktree':True,'python':platform.python_version(),'macos':platform.mac_ver()[0],
      'chip':'Apple M5 Max','gpu_cores':40,'memory_gb':48,
      'target_revision':'482ca0f3832238542f8f5295dde86b5f22711d80',
      'draft_revision':'b9a5dbdf03bc999c6c73c426b19c2d9041cea393',
      'draft_checkpoint_sha256':'2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7',
      'input_sha256':digest(root/'tools/bench/results/m5max-27b-dspark/inputs.json'),
      'sources':source,'packs':packs,
      'validation':{'final_cpu_contracts':100,'final_draft_attention_shader_oracles':179,
                    'final_gdn_mlp_shader_oracles':185,'exact_selected_nvfp4_shader_cases':4},
      'excluded_blobs':'Compiled program JSON and weight blobs remain in the recorded local paths; original experiment grids, commands, failures and samples are retained.'}
(r/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
subprocess.run(['.venv/bin/python',str(r/'summarize.py')],check=True)
for p in r.iterdir():
    if p.is_file()and p.suffix in ('.json','.jsonl','.log','.py','.passed','.metal')and not p.name.startswith('program-'):
        shutil.copy2(p,d/p.name)
shutil.copy2(r/'selected-refined-contexts.json',d.parent/'selected-contexts.json')
q=json.loads((r/'selected-nvfp4-refined-contexts.json').read_text())
(d.parent/'selected-nvfp4-endpoints.json').write_text(json.dumps({k:q[k]for k in ('128','32768')},indent=2)+'\n')
print('ARCHIVED',d,flush=True)

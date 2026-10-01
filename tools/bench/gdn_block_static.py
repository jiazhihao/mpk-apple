"""Experimental static full-GDN compiler; never selected by production emission.

Inline the production task bodies into one fixed-worker kernel. The normalized
multi-dispatch program is a required control: cooperative activation loading and
common threadgroup geometry change performance independently of fusion.
"""
import re, subprocess, copy
from pathlib import Path
from monolith.runtime.program import KernelSpec, OpSpec, BufferSpec

def normalize(p,sgs,mode="coop",groups=None):
    if sgs not in (4,8,16,32) or (mode=='staged' and sgs>8):
        raise ValueError('unsupported SIMD-group geometry')
    p=copy.deepcopy(p)
    changed=set()
    for o in p.ops:
        k=p.kernels[o.kernel]
        if k.function not in ('gemm_tile','x_permute','rmsnorm_stat','gdn_mixer','gdn_norm'):
            raise ValueError(f'unsupported experimental task: {k.function}')
        if mode=='coop' and k.function=='gemm_tile' and '#define WEIGHTS_PER_WORD 32u' in k.source:
            raise ValueError('cooperative 32-column tiles do not support packed 4-bit weights')
        if k.function=='gemm_tile':
            if o.kernel not in changed:
                if mode=='staged':
                    k.source=k.source.replace('tensor<device bfloat,', 'tensor<threadgroup bfloat,')
                    k.source=k.source.replace('tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));',f'threadgroup bfloat activations[{sgs}][TK * TM];')
                    k.source=k.source.replace('auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);', f"""for (uint ai=lane; ai<TK*TM; ai+=32u)
        activations[sg % {sgs}u][ai]=xp[(ulong)(ai/TK)*K+kp*TK+ai%TK];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tA_t tA(activations[sg % {sgs}u], dextents<int,2>(int(TK),int(TM)));
      auto sA=tA.slice<int(TK),int(TM)>(0,0);""")
                else:
                    k.macros.update(TM='16',TK='32u',SCALE_CACHE='0')
                    k.source=k.source.replace('tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));','auto aT=op.get_left_input_cooperative_tensor<bfloat,bfloat,float>();')
                    k.source=k.source.replace('get_destination_cooperative_tensor<tA_t, decltype(bT), float>()','get_destination_cooperative_tensor<decltype(aT), decltype(bT), float>()')
                    k.source=k.source.replace('auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);', """for(uint16_t ai=0;ai<aT.get_capacity();ai++) {
      auto c=aT.get_multidimensional_index(ai);
      aT[ai]=(c[1]<T_act && c[0]<TK) ? xp[(ulong)c[1]*K+kp*TK+c[0]] : bfloat(0);
     }""").replace('op.run(sA,bT,cT)','op.run(aT,bT,cT)').replace('op.run(sA, bT, cT)','op.run(aT, bT, cT)')
                k.source=k.source.split('kernel void coop_layout')[0]
                changed.add(o.kernel)
            if int(k.macros.get('KSPLIT','1').rstrip('u'))>1:
                k.macros['KSPLIT']=f'{sgs}u'
                if k.macros.get('SCALE_CACHE')=='1': k.macros['SCALE_CACHE']='0'
                k.macros['STATIC_GEMM_P_N_SG']=f'{o.grid[0]*sgs}u'
            else:
                o.grid=(o.grid[0]*o.threadgroup[0]//(32*sgs),1,1)
        elif k.function=='gdn_mixer':
            k.macros['LOCAL_GROUPS']=f'{sgs}u'
            o.grid=(o.grid[0]*o.threadgroup[0]//(32*sgs),1,1)
        else:
            if mode=='coop':
                if k.function=='x_permute': k.macros['TK']='32u'
                elif k.function=='gdn_norm': k.macros['PERM_TK']='32u'
            assert o.threadgroup[0]==32,k.function
            o.grid=((o.grid[0]+sgs-1)//sgs,1,1)
        if groups is not None and k.function=='gemm_tile':
            import struct
            o.grid=(groups,1,1)
            k.macros['STATIC_GEMM_P_N_SG']=f'{groups*sgs}u'
            pname=next(n for slot,n,off in o.bindings if slot==4)
            param=bytearray(p.buffers[pname].init);struct.pack_into('<I',param,8,groups*sgs);p.buffers[pname].init=bytes(param)
        o.threadgroup=(32*sgs,1,1)
    return p

def stage(k,i,immutable=()):
    src=re.sub(r'^\s*#include[^\n]*','',k.source,flags=re.M)
    defs='\n'.join(f'#define {a} {b}' for a,b in k.macros.items())
    src=subprocess.run(['xcrun','clang','-E','-P','-x','c++','-'],input=defs+'\n'+src,text=True,capture_output=True,check=True).stdout
    # Only the desired kernel becomes a callable task. Remove all other entries.
    signature=None;body=None
    for m in reversed(list(re.finditer(r'kernel void (\w+)\((.*?)\)\s*\{',src,re.S))):
        end=m.end(); depth=1
        while depth:
            if src[end]=='{':depth+=1
            elif src[end]=='}':depth-=1
            end+=1
        if m[1]==k.function:signature=m[2];body=src[m.end():end-1]
        src=src[:m.start()]+src[end:]
    assert signature is not None,k.function
    scratch=[]
    def shared(m):
        scratch.append(f'{m[1]} {m[2]}{m[3]};');return ''
    body=re.sub(r'threadgroup\s+(\w+)\s+(\w+)\s*((?:\[[^\]]+\])*)\s*;',shared,body)
    for decl in scratch:
        name=re.match(r'\w+ (\w+)',decl)[1]
        body=re.sub(r'\b'+name+r'\b','sm.'+name,body)
    pars=[];clean=[]
    for param in signature.split(','):
        param=param.strip();m=re.search(r'\[\[(.*?)\]\]',param);attr=m[1] if m else None
        plain=re.sub(r'\s*\[\[.*?\]\]','',param)
        mm=re.match(r'(.*?)\b(\w+)\s*$',plain);typ,name=mm.groups()
        pars.append((typ.strip(),name,attr));clean.append(plain)
    src+='\nstruct Scratch { '+ ' '.join(scratch or ['uint unused;'])+' };\n'
    src+='static inline void task('+','.join(clean)+', threadgroup Scratch& sm) {\n'+body+'\n}\n'
    # The tensor view type is part of MPP and cannot carry coherent qualification;
    # other device pointers (including its backing activation pointer) can.
    src=re.sub(r'\bdevice\b','coherent(device) device',src)
    for name in immutable:
        src=re.sub(r'coherent\(device\) (device[^,;(){}=\n]*\b'+re.escape(name)+r'\b)',r'\1',src)
    if k.function=='gemm_tile':
        # uint4 pointers in this kernel address immutable packed weights only.
        src=src.replace('coherent(device) device const uint4*','device const uint4*')

    return f'namespace s{i} {{\n'+src+'\n}\n',pars

def merge(p,workers,sgs):
    src='#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n'
    names=list(dict.fromkeys(n for o in p.ops for _,n,_ in o.bindings));bi={n:i for i,n in enumerate(names)}
    if len(names)>=30: raise ValueError('too many Metal buffer bindings')
    arguments=[]
    for n,i in bi.items():
        typ='constant' if p.buffers[n].role=='params' else ('device' if p.buffers[n].role=='weights' else 'coherent(device) device')
        arguments.append(f'{typ} uchar* b{i} [[buffer({i})]]')
    arguments+= [f'coherent(device) device atomic_uint* flags [[buffer({len(names)})]]','uint tid [[thread_index_in_threadgroup]]','uint worker [[threadgroup_position_in_grid]]']
    calls=[];sizes=[]
    for i,o in enumerate(p.ops):
        k=p.kernels[o.kernel]
        _,probe=stage(k,i)
        immutable=[name for typ,name,attr in probe if attr.startswith('buffer(') and
                                                    p.buffers[next(n for slot,n,off in o.bindings if slot==int(attr[7:-1]))].role=='weights']
        text,pars=stage(k,i,immutable);src+=text;sizes.append(f'sizeof(s{i}::Scratch)')
        byslot={slot:(bi[n],off) for slot,n,off in o.bindings}
        args=[]
        for typ,name,attr in pars:
            if attr.startswith('buffer('):
                slot=int(attr[7:-1]);b,off=byslot[slot]
                ptr=f'(b{b}+{off}ul)'
                if name not in immutable: typ=re.sub(r'\bdevice\b','coherent(device) device',typ)
                args.append(f'*({typ.replace("&","*")}){ptr}' if '&' in typ else f'({typ}){ptr}')
            else:
                value={'thread_position_in_grid':f'task_id*{32*sgs}u+tid','thread_index_in_simdgroup':'tid%32u','threads_per_simdgroup':'32u','threadgroup_position_in_grid':'task_id','thread_index_in_threadgroup':'tid','simdgroup_index_in_threadgroup':'tid/32u'}.get(attr)
                assert value,(typ,name,attr)
                args.append(f'uint3({value},0,0)' if typ=='uint3' else value)
        args.append(f'*(threadgroup s{i}::Scratch*)scratch')
        sync='if (!stage_barrier(flags, worker, tid, ok)) return;' if i and o.barrier_before else 'threadgroup_barrier(mem_flags::mem_threadgroup);'
        calls.append(f'{{ using namespace s{i};\n{sync}\nfor (uint task_id=worker;task_id<{o.grid[0]}u;task_id+={workers}u) {{ task('+','.join(args)+'); threadgroup_barrier(mem_flags::mem_threadgroup); }\n}\n')
    barrier=Path(__file__).with_name('gdn_static.metal').read_text().split('kernel void gdn_static(')[0]
    src+=f'\n#define WORKERS {workers}u\n'+barrier
    src+='\nkernel void full_gdn('+','.join(arguments)+') {\nthreadgroup uint ok;\nthreadgroup uchar scratch['+('max('+','.join(sizes)+')' if len(sizes)==2 else 'max({'+','.join(sizes)+'})')+'];\n'+''.join(calls)+'}\n'
    # MSL max is binary, not an initializer-list overload.
    size=sizes[0]
    for x in sizes[1:]:size=f'(({size})>({x})?({size}):({x}))'
    src=re.sub(r'threadgroup uchar scratch\[.*?\];',f'threadgroup uchar scratch[{size}];',src)
    out=copy.deepcopy(p);out.kernels={'mega':KernelSpec(src,'full_gdn',{},4<<16)}
    out.buffers['mega.flags']=BufferSpec((workers+1)*4)
    out.ops=[OpSpec('mega',[(i,n,0) for n,i in bi.items()]+[(len(names),'mega.flags',0)],(workers,1,1),(32*sgs,1,1))]
    src=src.replace('threadgroup uint ok;','').replace(';\n{ using namespace s0;', ';\nthreadgroup uint& ok=*((threadgroup uint*)scratch);\n{ using namespace s0;',1)
    out.kernels['mega'].source=src
    return out

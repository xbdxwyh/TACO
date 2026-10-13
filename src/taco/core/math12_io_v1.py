"""Immutable JSON artifacts and two-slot resumable checkpoints for this suite."""
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(x) for x in stream if x.strip()]
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value,immutable=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if read(path)!=value:raise ValueError('Immutable artifact changed: '+str(path))
        return
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def bound(path,value,binding):save(path,dict(binding=binding,sha256=digest(value),payload=value))
def unbound(path,binding):
    r=read(path)
    if r['binding']!=binding or r['sha256']!=digest(r['payload']):raise ValueError('Artifact binding changed: '+str(path))
    return r['payload']
def progress(out,phase,saved,total):save(out/'PROGRESS.json',dict(phase=phase,saved=saved,total=total),False)
def checkpoint(out,job,obj,binding):
    import torch
    pointer=out/'RESUME'/f'{job}.json';previous=read(pointer) if pointer.exists() else None
    slot=1-previous['slot'] if previous else 0;path=pointer.parent/f'{job}_{slot}.pt'
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
    save(pointer,dict(binding=binding,slot=slot,file=path.name,sha256=sha(path)),False)
def resume(out,job,binding,device):
    import torch
    pointer=out/'RESUME'/f'{job}.json'
    if not pointer.exists():return None
    r=read(pointer);path=pointer.parent/r['file']
    if r['binding']!=binding or r['sha256']!=sha(path):raise ValueError('Checkpoint changed')
    obj=torch.load(path,map_location=device,weights_only=True)
    if obj['binding']!=binding:raise ValueError('Checkpoint protocol changed')
    return obj

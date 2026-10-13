"""Author's pinned Math-Shepherd checkpoint and ki/plus-minus recipe."""
import ast
import math
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parent
import torch
from taco.core.math12_io_v1 import read,sha,digest

MODEL=None  # Configured by NativeAdapter from a local path.
MAX_TOKENS=4096
TAG='ки'


def tokenizer(path=None):
    from transformers import AutoTokenizer
    # Official source is LlamaTokenizer with BOS; keep its legacy SentencePiece
    # behavior explicitly. The repo has no chat template: do not add one.
    return AutoTokenizer.from_pretrained(path or MODEL,local_files_only=True,trust_remote_code=False,use_fast=False,legacy=True)

def label_tokens(tok):
    labels=tok.encode('+ -')[1:];tag=tok.encode(TAG)[-1]
    if labels!=[648,387] or tag!=12902:raise ValueError('Official token identities differ')
    return labels,tag

def encode(tok,case):
    if any(TAG in s for s in [case['question']]+case['steps']):raise ValueError('Public text contains reserved step tag')
    if len(case['steps'])!=len(case['targets']) or len(set(case['targets']))!=len(case['targets']):raise ValueError('Misaligned targets')
    text=case['question']+' '+'\n'.join(f'Step {i}: {s} {TAG}' for i,s in enumerate(case['steps'],1))
    ids=tok.encode(text);_,tag=label_tokens(tok);pos=[i for i,v in enumerate(ids) if v==tag]
    if len(pos)!=len(case['targets']):raise ValueError('Step marker count differs')
    return dict(input_ids=ids,positions=pos)

def strict_load_info(info,keys,expected):
    for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        if info.get(k):raise ValueError('Non-strict load '+k+': '+str(info[k]))
    if set(keys)!=set(expected):raise ValueError('Checkpoint tensor identity mismatch')

def load():
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    print('LOAD_MATH_SHEPHERD: official checkpoint/config; BF16 eager batch1; no quantization',flush=True)
    model,info=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',output_loading_info=True,weights_only=True)
    index=read(MODEL/'pytorch_model.bin.index.json')['weight_map']
    strict_load_info(info,model.state_dict(),index)
    assert len(index)==291 and 'lm_head.weight' in index
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    model.eval();model.config.use_cache=False
    model.math12_load_audit=dict(parameters=sum(p.numel() for p in model.parameters()),tensors=len(index),
        strict_loading=info,inference_dtype='bfloat16',attention='eager',quantized=False,
        architecture=model.__class__.__name__,published_config_preserved=True,weights_only=True)
    print('MODEL_LOADED '+str(model.math12_load_audit),flush=True)
    return model

def probabilities(logits,positions,labels):
    # Exactly the author's two-vocabulary-logit normalization, not sigmoid or
    # the full-vocabulary probability; labels ordered positive then negative.
    return logits[:,:,labels].softmax(dim=-1)[0,positions,0].float().cpu().tolist()

def predict(model,tok,case):
    enc=encode(tok,case);n=len(enc['input_ids']);start=time.monotonic();peak=0.
    if n>MAX_TOKENS:ps=[None]*len(case['targets']);status='context_overflow'
    else:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.inference_mode():
            ids=torch.tensor([enc['input_ids']],device='cuda')
            logits=model(input_ids=ids,use_cache=False).logits
            ps=probabilities(logits,enc['positions'],label_tokens(tok)[0])
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1024**3;status='ok'
        del logits,ids
    r=dict(public_sha256=digest(case),targets=case['targets'],probabilities=ps,input_tokens=n,
        positions=enc['positions'],encoded_sha256=digest(enc),seconds=time.monotonic()-start,
        peak_allocated_GiB=peak,status=status)
    validate(r,case,tok);return r

def validate(r,case,tok):
    enc=encode(tok,case)
    assert r['public_sha256']==digest(case) and r['targets']==case['targets']
    assert r['encoded_sha256']==digest(enc) and r['positions']==enc['positions']
    assert r['input_tokens']==len(enc['input_ids']) and len(r['probabilities'])==len(case['targets'])
    if r['status']=='ok':
        assert r['input_tokens']<=MAX_TOKENS and all(p is not None and math.isfinite(p) and 0<=p<=1 for p in r['probabilities'])
    else:
        assert r['status']=='context_overflow' and r['input_tokens']>MAX_TOKENS
        assert all(p is None for p in r['probabilities'])

def evidence(case,pred,prepared):
    values=dict(zip(case['targets'],pred['probabilities']))
    targets=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')]+['answer']
    return {t:values[t] for t in targets}



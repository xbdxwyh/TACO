"""Pinned, unmodified Qwen PRM implementation and official extra_0 interface."""
from pathlib import Path
import gc
import math
import time
import torch
from taco.core.math12_io_v1 import read, sha, digest

ROOT=Path(__file__).resolve().parent
MODEL=None  # Configured by NativeAdapter from a local path.
SYSTEM='Please reason step by step, and put your final answer within \\boxed{}.'
MAX_TOKENS=4096


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)

def encode(tok,case):
    if any('<extra_0>' in s for s in [case['question']]+case['steps']):
        raise ValueError('Public text contains reserved reward separator')
    messages=[dict(role='system',content=SYSTEM),dict(role='user',content=case['question']),
              dict(role='assistant',content='<extra_0>'.join(case['steps'])+'<extra_0>')]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    ids=tok.encode(text)
    separator=tok.encode('<extra_0>')
    if len(separator)!=1:raise ValueError('Reward separator must be exactly one token')
    positions=[i for i,t in enumerate(ids) if t==separator[0]]
    if len(positions)!=len(case['steps']) or len(positions)!=len(case['targets']):
        raise ValueError('Reward positions are misaligned')
    return dict(input_ids=ids,positions=positions)

def load():
    from transformers import AutoModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 GPU required')
    print('LOAD_QWEN: pinned original custom code; BF16, eager, batch=1, no quantization',flush=True)
    model,info=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',
        output_loading_info=True)
    for key in ('missing_keys','mismatched_keys','error_msgs'):
        if info.get(key):raise ValueError('Non-strict model load '+key+': '+str(info[key]))
    # The released checkpoint also stores a language-generation head; the
    # unmodified official PRM class has no such layer and uses score.0/score.2.
    # Permit precisely this documented extra, never missing reward parameters.
    if set(info.get('unexpected_keys',[]))!={'lm_head.weight'}:
        raise ValueError('Unexpected official extra tensor set: '+str(info.get('unexpected_keys')))
    model.eval();model.config.use_cache=False
    keys=set(model.state_dict())
    expected=set(read(MODEL/'model.safetensors.index.json')['weight_map'])
    if keys!=expected-{'lm_head.weight'}:raise ValueError('Model tensor names differ from official PRM index')
    if not {'score.0.weight','score.0.bias','score.2.weight','score.2.bias'}<=keys:
        raise ValueError('Trained reward head missing')
    model.math12_load_audit=dict(parameters=sum(p.numel() for p in model.parameters()),
        unused_checkpoint_keys=info['unexpected_keys'],missing_keys=info['missing_keys'],
        mismatched_keys=info['mismatched_keys'],reward_head_strict=True,attention='eager',
        explanation='Official PRM forward uses backbone plus score MLP; released lm_head.weight is unused by the official class.')
    print(f'MODEL_LOADED parameters={model.math12_load_audit["parameters"]}; backbone and reward head strict',flush=True)
    return model

def predict(model,tok,case):
    encoded=encode(tok,case);count=len(encoded['input_ids']);start=time.monotonic();peak=0.
    if count>MAX_TOKENS:
        probabilities=[None]*len(case['targets']);status='context_overflow'
    else:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.inference_mode():
            ids=torch.tensor([encoded['input_ids']],device='cuda')
            logits=model(input_ids=ids,use_cache=False)[0]
            # Official recipe: softmax in model output dtype, class 1 at separators.
            probabilities=logits.softmax(dim=-1)[0,encoded['positions'],1].float().cpu().tolist()
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1024**3;status='ok'
        del logits,ids
    result=dict(public_sha256=digest(case),targets=case['targets'],probabilities=probabilities,
        input_tokens=count,positions=encoded['positions'],encoded_sha256=digest(encoded),
        seconds=time.monotonic()-start,peak_allocated_GiB=peak,status=status)
    validate(result,case,tok)
    return result

def validate(r,case,tok):
    enc=encode(tok,case)
    assert r['public_sha256']==digest(case) and r['targets']==case['targets']
    assert r['encoded_sha256']==digest(enc) and r['positions']==enc['positions']
    assert r['input_tokens']==len(enc['input_ids']) and len(r['probabilities'])==len(case['targets'])
    if r['status']=='ok':
        assert r['input_tokens']<=MAX_TOKENS
        assert all(p is not None and math.isfinite(p) and 0<=p<=1 for p in r['probabilities'])
    else:
        assert r['status']=='context_overflow' and r['input_tokens']>MAX_TOKENS
        assert all(p is None for p in r['probabilities'])

def evidence(case,pred,prepared):
    values=dict(zip(case['targets'],pred['probabilities']))
    targets=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')]+['answer']
    return {t:values[t] for t in targets}


"""Official SPARE Qwen2.5-3B PRM, pinned offline loading and exact tag scoring.

The system instruction and (-,+) logits protocol come from the pinned official
example. No generated reasoning, remote code, reference answer or new head.
"""
from pathlib import Path
import gc
import math
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from taco.core.math12_io_v1 import read,digest,progress
from .skywork import check_file

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent/'assets/spare'
MANIFEST=read(ASSETS/'MODEL_MANIFEST.json')
INSTRUCTION=read(ASSETS/'instruction.json')
DEFAULT_MODEL=None  # Supply a local directory explicitly.
STEP_TAG=' ки'
MAX_TOKENS=1024  # Fixed, tested 8 GiB batch-one bound. Never silently truncate.




def token_spec(tokenizer):
    classes=tokenizer.convert_tokens_to_ids(['-','+'])
    tag_ids=tokenizer.encode(STEP_TAG)
    if len(set(classes))!=2 or any(i is None or i==tokenizer.unk_token_id for i in classes):
        raise ValueError('Missing official +/- vocabulary tokens')
    if not tag_ids:raise ValueError('Missing official step tag')
    return classes,tag_ids[-1]


def encode_case(tokenizer,case):
    if not case['steps'] or len(case['steps'])!=len(case['targets']) or len(set(case['targets']))!=len(case['targets']):
        raise ValueError('Step/target alignment invalid')
    classes,tag=token_spec(tokenizer)
    # Reserve the marker; never accidentally score a tag present in user data.
    if any(tag in tokenizer.encode(s) for s in [case['question'],*case['steps']]):
        raise ValueError('Public content collides with the reserved SPARE marker')
    generation="Let's think step by step.\n"+''.join(s+STEP_TAG+'\n' for s in case['steps'])
    messages=[dict(role='system',content=INSTRUCTION),dict(role='user',content=case['question']),
              dict(role='assistant',content=generation)]
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    positions=[i for i,t in enumerate(ids) if t==tag]
    if len(positions)!=len(case['targets']):raise ValueError('SPARE marker count does not match original nodes')
    return dict(input_ids=ids,positions=positions,targets=case['targets'],class_token_ids=classes,tag_token_id=tag)


def validate_loading_info(info):
    bad={k:info.get(k,[]) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(k)}
    if bad:raise ValueError('Incomplete or incompatible pretrained checkpoint: '+repr(bad))


def load_model(folder):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 required')
    config=read(folder/'config.json')
    if config.get('model_type')!='qwen2' or config.get('architectures')!=['Qwen2ForCausalLM'] or config.get('auto_map'):
        raise ValueError('Expected built-in Qwen2ForCausalLM without remote code')
    # CPU loading avoids holding a second checkpoint-sized copy on the GPU.
    model,info=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.bfloat16,
        local_files_only=True,trust_remote_code=False,use_safetensors=True,low_cpu_mem_usage=False,
        attn_implementation='sdpa',output_loading_info=True)
    validate_loading_info(info)
    if any(p.is_meta for p in model.parameters()):raise ValueError('Unloaded model parameter')
    model.config.use_cache=False;model.to('cuda').eval()
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    token_spec(tokenizer)
    print(f'MODEL_LOADED parameters={sum(p.numel() for p in model.parameters())}; official LM +/- logits; BF16',flush=True)
    return model,tokenizer


def step_probabilities(model,input_ids,positions,class_ids):
    # Preserve official full-forward and conditional (-,+) softmax semantics.
    logits=model(input_ids,use_cache=False).logits[:,:,class_ids]
    return logits.softmax(dim=-1)[:,:,1][:,positions]


def validate_prediction(r,case):
    if r['public_sha256']!=digest(case) or r['revision']!=MANIFEST['revision'] or r['targets']!=case['targets']:
        raise ValueError('Native prediction identity mismatch')
    pos=r['positions']
    if len(r['probabilities'])!=len(case['targets']) or len(pos)!=len(case['targets']):raise ValueError('Native alignment mismatch')
    if pos!=sorted(set(pos)) or any(i<0 or i>=r['input_tokens'] for i in pos):raise ValueError('Invalid marker positions')
    if r['status']=='ok':
        if r['input_tokens']>MAX_TOKENS or any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in r['probabilities']):
            raise ValueError('Invalid native probability')
    elif r['status']=='context_overflow':
        if r['input_tokens']<=MAX_TOKENS or any(v is not None for v in r['probabilities']):raise ValueError('Invalid overflow record')
    else:raise ValueError('Unrecognized native result')


def predict(model,tokenizer,case):
    encoded=encode_case(tokenizer,case);count=len(encoded['input_ids'])
    start=time.monotonic();status='ok';peak=0.
    if count>MAX_TOKENS:
        values=[None]*len(case['targets']);status='context_overflow'
    else:
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ids=torch.tensor([encoded['input_ids']],device='cuda')
            values=step_probabilities(model,ids,encoded['positions'],encoded['class_token_ids'])[0].float().cpu().tolist()
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1024**3
    r=dict(public_sha256=digest(case),revision=MANIFEST['revision'],targets=case['targets'],probabilities=values,
           positions=encoded['positions'],input_tokens=count,encoded_sha256=digest(encoded),status=status,
           seconds=time.monotonic()-start,peak_allocated_GiB=peak)
    validate_prediction(r,case)
    return r


def evidence(case,pred,prepared):
    validate_prediction(pred,case)
    values=dict(zip(pred['targets'],pred['probabilities']))
    targets=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')]+['answer']
    return {t:values[t] for t in targets}

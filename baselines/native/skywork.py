"""Pinned Skywork PRM: official newline rewards, Qwen2 backbone + trained value head.

The official PRM wrapper uses AutoModelForCausalLM hidden states + ValueHead,
then sigmoid. We omit its unused vocabulary projection and retain exactly the
checkpoint's value head. No remote Python or randomly initialized head is used.
"""
from pathlib import Path
import gc
import hashlib
import math
import time
import torch
from torch import nn
from transformers import Qwen2Config,Qwen2Model,AutoTokenizer
from taco.core.math12_io_v1 import read,digest,progress

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent/'assets/skywork'
MANIFEST=read(ASSETS/'MODEL_MANIFEST.json')
DEFAULT_MODEL=None  # Supply a local directory explicitly.
MAX_TOKENS=4096


def check_file(path,meta):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=meta['size']:return False
    h=hashlib.sha256() if meta['sha256'] else hashlib.sha1()
    if not meta['sha256']:h.update(b'blob '+str(meta['size']).encode()+b'\0')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()==(meta['sha256'] or meta['git_blob_sha1'])




def encode_case(tokenizer,case):
    """Byte-for-byte official prepare_input(..., step_token='\n') tokenization.

    Display and LaTeX may span multiple lines. Score at the LAST newline of
    each original node, retaining all intermediate lines and their positions.
    Never split labels by count or guess where a missing label belongs.
    """
    if len(case['steps'])!=len(case['targets']) or len(set(case['targets']))!=len(case['targets']):
        raise ValueError('Step/target alignment invalid')
    if not tokenizer.bos_token:raise ValueError('Pinned tokenizer must have BOS')
    nl=tokenizer.encode('\n',add_special_tokens=False)
    if len(nl)!=1:raise ValueError('Official newline delimiter is not a single token')
    ids=tokenizer.encode(tokenizer.bos_token+case['question']+'\n')
    boundaries=[];line_positions=[]
    for step in case['steps']:
        for line in step.split('\n'):
            ids.extend(tokenizer.encode(line) if line else [])
            ids.append(nl[0]);line_positions.append(len(ids)-1)
        boundaries.append(len(ids)-1)
    return dict(input_ids=ids,positions=boundaries,line_positions=line_positions,targets=case['targets'])


class SkyworkPRM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model=Qwen2Model(config)
        self.v_head=nn.Module()
        self.v_head.summary=nn.Linear(config.hidden_size,1)

    def forward(self,input_ids,positions):
        h=self.model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False,
                     return_dict=True).last_hidden_state[:,positions,:]
        return torch.sigmoid(self.v_head.summary(h.to(self.v_head.summary.weight.dtype)).squeeze(-1))


def strict_assign(model,state,config):
    # Some upstream saves include the tied LM projection. It is unused by PRM,
    # but discard it only after proving it equals the input embeddings.
    state=dict(state)
    if 'lm_head.weight' in state:
        if not config.tie_word_embeddings or not torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight']):
            raise ValueError('Unexpected untied language projection')
        del state['lm_head.weight']
    expected=model.state_dict()
    if set(expected)!=set(state):
        raise ValueError('Checkpoint keys differ: missing='+str(sorted(set(expected)-set(state)))+
                         '; unexpected='+str(sorted(set(state)-set(expected))))
    for key,value in state.items():
        if value.shape!=expected[key].shape:raise ValueError('Checkpoint shape mismatch: '+key)
    for key in ('v_head.summary.weight','v_head.summary.bias'):
        if not torch.isfinite(state[key]).all():raise ValueError('Nonfinite value head')
    if not torch.count_nonzero(state['v_head.summary.weight']):raise ValueError('Empty value head')
    model.load_state_dict(state,strict=True,assign=True)
    # inv_freq is nonpersistent, so assign=True cannot materialize a meta
    # buffer. Recreate it from the pinned RoPE config (no learned weights).
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb=Qwen2RotaryEmbedding(config,device='cpu')
    if any(p.is_meta for p in model.parameters()):raise ValueError('Unloaded parameter')
    if any(b.is_meta for b in model.buffers()):raise ValueError('Unmaterialized buffer')


def load_model(folder):
    if not torch.cuda.is_available():raise ValueError('This full configuration requires the existing CUDA GPU')
    if not torch.cuda.is_bf16_supported():raise ValueError('BF16 not supported')
    cfg=read(folder/'config.json')
    cfg.pop('auto_map',None)
    config=Qwen2Config.from_dict(cfg);config._attn_implementation='sdpa';config.use_cache=False
    state=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    with torch.device('meta'):model=SkyworkPRM(config)
    strict_assign(model,state,config)
    del state
    # ValueHead is FP32 in the official wrapper, even with a BF16 backbone.
    model.model.to(device='cuda',dtype=torch.bfloat16);model.v_head.to(device='cuda',dtype=torch.float32)
    # Keep analytic RoPE frequencies in FP32 rather than rounding them with
    # the learned BF16 weights. This matches Qwen2's FP32 rotary arithmetic.
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb=Qwen2RotaryEmbedding(config,device='cuda')
    model.eval();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    print(f'MODEL_LOADED parameters={sum(p.numel() for p in model.parameters())}; trained head verified; BF16/FP32 head',flush=True)
    return model,tokenizer


def validate_prediction(r,case):
    if r['public_sha256']!=digest(case) or r['revision']!=MANIFEST['revision'] or r['targets']!=case['targets']:
        raise ValueError('Native prediction identity mismatch')
    if len(r['probabilities'])!=len(case['targets']) or len(r['positions'])!=len(case['targets']):raise ValueError('Native alignment mismatch')
    if r['status'] not in ('ok','context_overflow'):raise ValueError('Unrecognized native result')
    if r['status']=='ok':
        if r['input_tokens']>MAX_TOKENS or any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in r['probabilities']):
            raise ValueError('Invalid native probability')
    elif r['input_tokens']<=MAX_TOKENS or any(v is not None for v in r['probabilities']):raise ValueError('Invalid overflow record')


def predict(model,tokenizer,case):
    encoded=encode_case(tokenizer,case);count=len(encoded['input_ids'])
    start=time.monotonic();status='ok';peak=0.
    if count>MAX_TOKENS:
        values=[None]*len(case['targets']);status='context_overflow'
    else:
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ids=torch.tensor([encoded['input_ids']],device='cuda')
            values=model(ids,encoded['positions'])[0].float().cpu().tolist()
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/1024**3
    r=dict(public_sha256=digest(case),revision=MANIFEST['revision'],targets=case['targets'],probabilities=values,
           positions=encoded['positions'],line_positions=encoded['line_positions'],input_tokens=count,
           encoded_sha256=digest(encoded),status=status,seconds=time.monotonic()-start,peak_allocated_GiB=peak)
    validate_prediction(r,case)
    return r


def evidence(case,pred,prepared):
    validate_prediction(pred,case)
    values=dict(zip(pred['targets'],pred['probabilities']))
    targets=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')]+['answer']
    return {t:values[t] for t in targets}

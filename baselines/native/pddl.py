"""Full official PDDL weights, original Qwen PRM class, official marker recipe."""
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent
import torch
from taco.core.math12_io_v1 import read,sha,digest
from . import qwen as qwen_native

MODEL=None  # Configured by NativeAdapter from a local path.
MAX_TOKENS=4096
encode=qwen_native.encode
validate=qwen_native.validate
evidence=qwen_native.evidence
predict=qwen_native.predict




def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)


def load():
    from transformers import AutoModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    print('LOAD_PDDL: full merged FP32 checkpoint cast to BF16; eager; batch1; no quantization/adapter',flush=True)
    model,info=AutoModel.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',output_loading_info=True)
    for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        if info.get(k):raise ValueError('Strict PDDL load failed '+k+': '+str(info[k]))
    index=read(MODEL/'model.safetensors.index.json')['weight_map']
    assert set(model.state_dict())==set(index) and len(index)==342
    assert {'score.0.weight','score.0.bias','score.2.weight','score.2.bias'}<=set(index)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    model.eval();model.config.use_cache=False
    model.math12_load_audit=dict(parameters=sum(p.numel() for p in model.parameters()),tensors=len(index),
        original_dtype='float32',inference_dtype='bfloat16',attention='eager',strict_loading=info,
        full_merged_checkpoint=True,no_quantization=True,adapter_loaded=False,
        implementation_sha256=sha(MODEL/'modeling_qwen2_rm.py'))
    print('MODEL_LOADED strict 342/342 tensors; parameters='+str(model.math12_load_audit['parameters']),flush=True)
    return model



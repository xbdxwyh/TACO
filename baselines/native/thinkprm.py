"""Frozen ThinkPRM generation with corrected public-text parser v3.
Extracted from the completed protocol; no legacy pilot driver or downloader.
"""
from pathlib import Path
import hashlib
import time
from taco.core.math12_io_v1 import read, digest, sha
from .thinkprm_parser import parse as parse_output
MANIFEST = Path(__file__).resolve().parents[1]/'assets/thinkprm/MODEL_MANIFEST.json'
def prompt_content(case):
    solution = '\n'.join(f'Step {i}: {s}' for i, s in enumerate(case['steps'], 1))
    # Exact upstream inference helper template (not the slightly different HF demo).
    return ('You are given a math problem and a proposed step-by-step solution:\n\n'
            '[Math Problem]\n\n' + case['question'] + '\n\n[Solution]\n\n' + solution + '\n\n'
            'Review and critique each step in the proposed solution to determine whether each step is correct. '
            'If the solution is incomplete, only verify the provided steps.')

def format_prompt(tokenizer, case):
    return tokenizer.apply_chat_template([dict(role='user', content=prompt_content(case))],
                                         tokenize=False, add_generation_prompt=True)

def load_model(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This frozen native adapter requires a CUDA GPU supporting BF16')
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
        attn_implementation='sdpa').to('cuda').eval()
    return tokenizer, model

def generate_one(tokenizer, model, case, max_new_tokens, case_seconds, input_limit=2048):
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    prompt = format_prompt(tokenizer, case)
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    n = encoded['input_ids'].shape[1]
    if n > input_limit:
        return dict(input_tokens=n, output_tokens=0, seconds=0, raw_output='', finish_reason='input_overflow',
                    parsed=parse_output('', len(case['steps']), 'input_overflow'))
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    start = time.monotonic()
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() - start >= case_seconds
    eos = tokenizer.eos_token_id
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                              eos_token_id=eos, pad_token_id=eos, bos_token_id=tokenizer.bos_token_id)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(**encoded, generation_config=config,
                                   stopping_criteria=StoppingCriteriaList([Deadline()]))
    torch.cuda.synchronize()
    ids = generated[0, n:].tolist()
    elapsed = time.monotonic() - start
    reason = 'eos' if ids and ids[-1] == eos else 'length' if len(ids) >= max_new_tokens else 'case_timeout'
    output = tokenizer.decode(ids, skip_special_tokens=True)
    return dict(input_tokens=n, output_tokens=len(ids), seconds=elapsed, raw_output=output,
                finish_reason=reason, parsed=parse_output(output, len(case['steps']), reason),
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)

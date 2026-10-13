"""Pinned GenPRM-1.5B, official staged prompts with a bounded Windows backend.

Official analyze/verify/execute budget, sampling settings, sequential history and
top-20 reward formula are retained. Disclosed differences: HF BF16/SDPA instead
of vLLM, restricted mathematical Python, deterministic per-step sampling seeds,
32K context with explicit unknown, and one bounded final fallback (no infinite
retry). Missing output is unknown, never a fabricated negative/neutral label.
"""
from pathlib import Path
from collections import Counter
import gc
import math
import re
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from taco.core.math12_io_v1 import read,save,sha,digest,bound,unbound,progress
from .skywork import check_file
from .spare import validate_loading_info
from .genprm_math12_executor_v2 import CodeExecutor

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent/'assets/genprm'
MANIFEST=read(ASSETS/'MODEL_MANIFEST.json')
DEFAULT_MODEL=None  # Supply a local directory explicitly.
MAX_TOKENS=32768
SETTINGS=dict(analyze=True,verify=True,execute=True,majority_num=1,max_new_tokens=2048,
    verify_rounds=3,final_fallback_tokens=20,temperature=.6,top_p=.95,top_k=20,
    repetition_penalty=1.,sampling_seed=20261001,context_limit=MAX_TOKENS,
    executor='restricted_math_ast_v2',code_wall_seconds=5,code_memory_MiB=1024,
    missing='unknown; never zero or neutral',reward='official raw top20 logprobs + sampled token; min fallback')
SYSTEM='You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'
ANALYZE="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: "
VERIFY="<verify>\nLet's use python code to find any potential error:\n```python\n"
OUTPUT='<output>\n**Judgement**: $\\boxed'


def load_model(folder):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise ValueError('CUDA BF16 required')
    cfg=read(folder/'config.json')
    if cfg.get('model_type')!='qwen2' or cfg.get('architectures')!=['Qwen2ForCausalLM'] or cfg.get('auto_map'):
        raise ValueError('Built-in Qwen2 model required; no remote code')
    model,info=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.bfloat16,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,low_cpu_mem_usage=False,attn_implementation='sdpa',output_loading_info=True)
    validate_loading_info(info)
    if any(p.is_meta for p in model.parameters()): raise ValueError('Unloaded model parameter')
    model.config.use_cache=True;model.to('cuda').eval()
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    print(f'MODEL_LOADED parameters={sum(p.numel() for p in model.parameters())}; BF16; sequential generation',flush=True)
    return model,tok

def build_prompt(messages,tokenizer):
    # Verbatim semantics of the pinned official util.build_prompt.
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    if prompt.endswith(f'{tokenizer.eos_token}\n'): prompt=prompt[:-len(f'{tokenizer.eos_token}\n')]
    elif prompt.endswith(tokenizer.eos_token): prompt=prompt[:-len(tokenizer.eos_token)]
    return prompt

def audit_public(tokenizer,cases):
    values=[]
    for case in cases:
        validate_case(case)
        msgs=[dict(role='system',content=SYSTEM)]
        for j,step in enumerate(case['steps']):
            msgs.append(dict(role='user',content=(case['question']+'\n' if j==0 else '')+step))
            values.append(len(tokenizer.encode(build_prompt(msgs,tokenizer))))
            msgs.append(dict(role='assistant',content=''))
    return dict(cases=len(cases),nodes=sum(len(c['steps']) for c in cases),
                public_only_max_tokens=max(values),public_only_min_tokens=min(values),
                generated_history='grows during generation; every actual request checked without truncation',context_limit=MAX_TOKENS)

def validate_case(case):
    if not case['steps'] or len(case['steps'])!=len(case['targets']) or len(set(case['targets']))!=len(case['targets']):
        raise ValueError('Public step alignment invalid')

def sample_token(raw,generator,top_k=20,temperature=.6,top_p=.95):
    """Raw logprobs are recorded BEFORE temperature/top-k/top-p transforms."""
    raw=raw.float();values,indices=raw.topk(min(top_k,raw.numel()))
    probs=(values/temperature).softmax(-1)
    # Same nucleus boundary: retain the first token crossing top_p.
    keep=(probs.cumsum(-1)-probs)<top_p
    probs=probs*keep;probs=probs/probs.sum()
    chosen=indices[torch.multinomial(probs,1,generator=generator).item()].item()
    return chosen

def decision_logprobs(raw,chosen,yes,no):
    probs=raw.float().log_softmax(-1);top=probs.topk(min(20,probs.numel()))
    table={str(k):float(v) for k,v in zip(top.indices.tolist(),top.values.tolist())}
    table[str(chosen)]=float(probs[chosen])
    return dict(sampled=chosen,top=table,exact_yes=float(probs[yes]),exact_no=float(probs[no]))

class Generator:
    def __init__(self,model,tokenizer): self.model,self.tokenizer=model,tokenizer

    def generate(self,prompt,limit,stops,seed,heartbeat=None):
        tokenizer=self.tokenizer
        ids=tokenizer.encode(prompt)
        start=time.monotonic()
        result=dict(prompt_sha256=digest(prompt),input_tokens=len(ids),max_new_tokens=limit,
                    stops=stops,seed=seed,token_ids=[],decision_tokens=[],text='',finish=None,seconds=0.,peak_allocated_GiB=0.)
        if len(ids)+limit>MAX_TOKENS:
            result['finish']='context_overflow';return result
        device=next(self.model.parameters()).device
        generator=torch.Generator(device=device).manual_seed(seed)
        yes,no=[tokenizer.encode(s)[-1] for s in ('Yes','No')]
        if yes==no: raise ValueError('Invalid decision tokens')
        past=None;out=None;logits=None
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        last_beat=start
        with torch.inference_mode():
            # Only last-token logits, even during prefill. No [32K,vocab] allocation.
            for begin in range(0,len(ids),512):
                chunk=torch.tensor([ids[begin:begin+512]],device=device)
                out=self.model(chunk,past_key_values=past,use_cache=True,logits_to_keep=1)
                past=out.past_key_values;logits=out.logits[0,-1]
                if heartbeat and time.monotonic()-last_beat>=5:
                    heartbeat('prefill',begin+len(chunk[0]),len(ids));last_beat=time.monotonic()
            for index in range(limit):
                chosen=sample_token(logits,generator)
                result['token_ids'].append(chosen)
                # Only Yes/No positions are needed by the official reward formula.
                if chosen in (yes,no): result['decision_tokens'].append(dict(index=index,**decision_logprobs(logits,chosen,yes,no)))
                result['text']=tokenizer.decode(result['token_ids'],skip_special_tokens=True)
                matches=[(result['text'].index(s),s) for s in stops if s in result['text']]
                if matches:
                    position,stop=min(matches);result['text']=result['text'][:position+len(stop)];result['finish']='stop';break
                if chosen==tokenizer.eos_token_id: result['finish']='eos';break
                if heartbeat and time.monotonic()-last_beat>=5:
                    heartbeat('decode',index+1,limit);last_beat=time.monotonic()
                if index+1<limit:
                    out=self.model(torch.tensor([[chosen]],device=device),past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=out.past_key_values;logits=out.logits[0,-1]
            else: result['finish']='length'
        if device.type=='cuda':
            torch.cuda.synchronize();result['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
        result['seconds']=time.monotonic()-start
        del out,past,logits
        return result

def reward(full_text,last,tokenizer):
    """Official first Yes/No} text match and last matching output-token formula.

    Report explicit unknown for a missing/misaligned decision (the upstream
    implementation uses 0.5 or raises). Also store exact two-token probability
    for audit only: it is never substituted or chosen by dev performance.
    """
    match=re.search(r'(Yes|No)\}',full_text,re.IGNORECASE)
    if not match: return dict(probability=None,status='missing_judgement')
    label=match.group(1).capitalize();yes,no=[tokenizer.encode(s)[-1] for s in ('Yes','No')]
    selected=yes if label=='Yes' else no
    hits=[r for r in last['decision_tokens'] if r['sampled']==selected]
    if not hits: return dict(probability=None,status='judgement_token_mismatch',label=label)
    record=hits[-1];table=record['top'];floor=min(table.values())
    yl=table.get(str(yes),floor);nl=table.get(str(no),floor)
    sigmoid=lambda z: math.exp(-max(0.,-z))/(math.exp(-max(0.,-z))+math.exp(-max(0.,z)))
    return dict(probability=sigmoid(yl-nl),status='ok',label=label,decision_index=record['index'],
                used_top20_floor=str(yes) not in table or str(no) not in table,
                exact_two_token_probability=sigmoid(record['exact_yes']-record['exact_no']))

def infer_step(engine,messages,step,executor,seed,heartbeat=None):
    started=time.monotonic()
    tok=engine.tokenizer;prompt=build_prompt(messages,tok);events=[];codes=[]
    analyze_start=ANALYZE.format(cur_step=step)
    def gen(text,limit,stops):
        result=engine.generate(prompt+text,limit,stops,seed+len(events),heartbeat)
        events.append(result);return result
    first=gen(analyze_start,SETTINGS['max_new_tokens'],['</analyze>\n'])
    cur=analyze_start+first['text']+VERIFY
    verdict=dict(probability=None,status='missing_judgement');last=first
    if first['finish']=='context_overflow': verdict=dict(probability=None,status='context_overflow')
    else:
        for round_index in range(SETTINGS['verify_rounds']+1):
            left=SETTINGS['max_new_tokens']-len(tok.tokenize(cur))
            fallback=left<=0 or round_index>=SETTINGS['verify_rounds']
            if fallback:
                cur=analyze_start+first['text'].split('</analyze>')[0]+'</analyze>\n'+OUTPUT
                limit=SETTINGS['final_fallback_tokens'];stops=['</output>\n']
            else: limit=left;stops=['\n```\n','</output>\n']
            last=gen(cur,limit,stops)
            text=cur+last['text']
            if last['finish']=='context_overflow': verdict=dict(probability=None,status='context_overflow');cur=text;break
            if last['text'].endswith('</output>\n'):
                verdict=reward(text,last,tok);cur=text;break
            if fallback:
                # Terminate after one official fallback; no retry-until-lucky loop.
                verdict=dict(probability=None,status='unterminated_output');cur=text;break
            executed=executor.execute(text);codes.append(executed)
            cur=text+'[Code Output]\n\n```\n'+executed['output']+'\n```\n'
    return dict(messages_sha256=digest(messages),step=step,seed=seed,text=cur,verdict=verdict,
                generations=events,code_events=codes,input_tokens=sum(e['input_tokens'] for e in events),
                generated_tokens=sum(len(e['token_ids']) for e in events),
                seconds=time.monotonic()-started,
                generation_seconds=sum(e['seconds'] for e in events),
                peak_allocated_GiB=max(e['peak_allocated_GiB'] for e in events))

def step_seed(case,index): return (int(digest(case)[:12],16)+SETTINGS['sampling_seed']+index*1009)%(2**31)

def predict(model,tokenizer,case,out,binding):
    validate_case(case);engine=Generator(model,tokenizer)
    messages=[dict(role='system',content=SYSTEM)];steps=[];seals={};iid=case['item_id']
    with CodeExecutor() as executor:
        for index,text in enumerate(case['steps']):
            messages.append(dict(role='user',content=(case['question']+'\n' if index==0 else '')+text))
            path=out/'STEPS'/iid/f'step{index+1:02d}.json';seed=step_seed(case,index)
            def beat(phase,done,total):
                save(out/'PROGRESS.json',dict(phase='genprm_'+phase,saved=done,total=total,
                    detail=f'item={iid} step={index+1}/{len(case["steps"])}'),False)
            if path.exists():
                row=unbound(path,binding)
                if row['messages_sha256']!=digest(messages) or row['seed']!=seed or row['step']!=index+1: raise ValueError('Step resume binding drift')
                executor.replay(row['code_events'])
            else:
                print(f'GENPRM item={iid} STEP {index+1}/{len(case["steps"])}; analyze -> code -> judgement',flush=True)
                row=infer_step(engine,messages,index+1,executor,seed,beat);bound(path,row,binding)
                print(f'STEP_SAVED status={row["verdict"]["status"]}; tokens={row["generated_tokens"]}; code_calls={len(row["code_events"])}; seconds={row["seconds"]:.1f}',flush=True)
            steps.append(row);seals[path.relative_to(out).as_posix()]=sha(path)
            messages.append(dict(role='assistant',content=row['text']))
    pred=dict(public_sha256=digest(case),revision=MANIFEST['revision'],targets=case['targets'],
        probabilities=[s['verdict']['probability'] for s in steps],step_statuses=[s['verdict']['status'] for s in steps],
        step_files=seals,input_tokens=sum(s['input_tokens'] for s in steps),generated_tokens=sum(s['generated_tokens'] for s in steps),
        code_calls=sum(len(s['code_events']) for s in steps),code_status=dict(Counter(e['status'] for s in steps for e in s['code_events'])),
        seconds=sum(s['seconds'] for s in steps),peak_allocated_GiB=max(s['peak_allocated_GiB'] for s in steps),
        status='ok' if all(s['verdict']['status']=='ok' for s in steps) else 'partial_unknown',
        code_seconds=sum(e['seconds'] for s in steps for e in s['code_events']))
    validate_prediction(pred,case,out,binding);return pred

def validate_prediction(r,case,out=None,binding=None):
    if r['public_sha256']!=digest(case) or r['revision']!=MANIFEST['revision'] or r['targets']!=case['targets']: raise ValueError('Native identity mismatch')
    if len(r['probabilities'])!=len(case['targets']) or len(r['step_statuses'])!=len(case['targets']): raise ValueError('Native alignment mismatch')
    for p,status in zip(r['probabilities'],r['step_statuses']):
        if status not in ('ok','missing_judgement','judgement_token_mismatch','context_overflow','unterminated_output'):
            raise ValueError('Unrecognized step status')
        if status=='ok':
            if p is None or not math.isfinite(p) or not 0<=p<=1: raise ValueError('Invalid probability')
        elif p is not None: raise ValueError('Unavailable judgement must remain unknown')
    if out is not None:
        rows=[]
        for file,h in r['step_files'].items():
            path=(out/file).resolve()
            if out.resolve() not in path.parents or sha(path)!=h: raise ValueError('Sealed step changed')
            rows.append(unbound(path,binding))
        if len(rows)!=len(case['steps']) or [s['verdict']['probability'] for s in rows]!=r['probabilities']: raise ValueError('Step record mismatch')

def evidence(case,pred,prepared):
    validate_prediction(pred,case)
    values=dict(zip(pred['targets'],pred['probabilities']))
    targets=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')]+['answer']
    return {t:values[t] for t in targets}

def aggregate_cost(rows):
    codes=Counter();statuses=Counter()
    for r in rows: codes.update(r['code_status']);statuses.update(r['step_statuses'])
    seconds=sum(r['seconds'] for r in rows);n=len(rows)
    return dict(input_tokens=sum(r['input_tokens'] for r in rows),generated_tokens=sum(r['generated_tokens'] for r in rows),
        seconds=seconds,peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in rows),
        code_calls=sum(r['code_calls'] for r in rows),code_status=dict(codes),code_seconds=sum(r['code_seconds'] for r in rows),
        step_status=dict(statuses),step_coverage=statuses['ok']/sum(statuses.values()),
        projected_516_generation_hours=seconds/n*516/3600,
        note='One generated conversation per item shared across seeds/conditions; diagnostic code calls EXTRA to recovery budget8; loading/replay/download excluded; projection from pilot uncertain.')

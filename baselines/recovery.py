"""Shared final-task controller with explicit, budgeted missing-evidence policy.

All-known inputs reproduce the frozen final policy. No truth enters this file.
"""
import copy
import math
from taco.core.math12_syntax_v2 import parse_public,claim,Unsupported
from taco.core.math12_actions_v1 import learned_state,syntax_role,apply
from taco.core.math12_obligation_policy_v1 import closes_obligation
from taco.core.math12_step_goal_policy_v1 import choose_audit,StepMeter
from taco.core.math12_model_v1 import record


def specifications():
    result=[]
    for model,missing in [('ours','verify'),('thinkprm','verify'),('thinkprm','edit')]:
        for stateful in (False,True):
            for rank in ('rule','learned'):
                for audit in ('none','uncertain'):
                    name=f'{model}_{missing}_S{int(stateful)}_{rank}_{audit}'
                    result.append(dict(name=name,model=model,missing=missing,stateful=stateful,rank=rank,audit=audit,budget=8))
    return result


def state_view(prepared, evidence, stateful):
    # Conservative obligation bookkeeping only; do not turn None into a native
    # model error label in diagnostic metrics. Unknown cannot retire old errors.
    scores={t:0. if v is None else v for t,v in evidence.items()}
    state=learned_state(prepared,scores,stateful)
    unknown={t for t,v in evidence.items() if v is None}
    unresolved=unknown-set(state['retired'])
    return state,scores,unresolved


def original_check(meter,prepared,target):
    if target=='answer':return meter.check(target,prepared['answer'],'missing_evidence_checks')
    value=meter.check_original_step(target,prepared['nodes'][int(target[1:])-1]['display'])
    meter.cost['step_audit_checks']-=1
    if not meter.cost['step_audit_checks']:del meter.cost['step_audit_checks']
    meter.cost['missing_evidence_checks']+=1;meter.events[-1]['category']='missing_evidence_checks'
    return value


def operate(payload,menus,evidence,score_fn,spec):
    if spec['missing'] not in ('verify','edit') or spec['audit'] not in ('none','uncertain') or spec['rank'] not in ('rule','learned') or spec['budget']!=8:
        raise ValueError('Unrecognized fixed policy')
    p=parse_public(payload);ev=dict(evidence)
    targets={n['node_id'] for n in p['nodes'] if n['claim']['kind'] not in ('given','cue')}|{'answer'}
    if set(ev)!=targets or any(v is not None and (not math.isfinite(v) or not 0<=v<=1) for v in ev.values()):raise ValueError('Bad evidence')
    raw_state,_,raw_unknown=state_view(p,ev,spec['stateful'])
    meter=StepMeter(p['question']);reserved=int(spec['audit']!='none');missing_events=[]
    # Main adapter first verifies unknown original observations, in fixed input
    # order. Every check spends the SAME eight-call allowance as repairs.
    if spec['missing']=='verify':
        order=sorted(raw_unknown,key=lambda t:100000 if t=='answer' else int(t[1:]))
        for t in order:
            if len(missing_events)>=8-reserved:break
            _,_,active_unknown=state_view(p,ev,spec['stateful'])
            if t not in active_unknown:continue
            v=original_check(meter,p,t);missing_events.append(dict(target=t,valid=v))
            if v is not None:ev[t]=float(v)
    state,probs,unknown=state_view(p,ev,spec['stateful'])
    required=set(state['pending'])
    if probs['answer']<.5:required.add('answer')
    before_audit=sorted(required)
    target,eligible=choose_audit(p,probs,state,spec['audit'])
    audit_valid=None;reopened=[]
    if target is not None:
        audit_valid=meter.check_original_step(target,p['nodes'][int(target[1:])-1]['display'])
        if audit_valid is False:required.add(target);reopened=[target]
    candidate_budget=8-reserved-len(missing_events)
    initial_required=sorted(required);current=copy.deepcopy(payload);accepted=[];attempts=[]
    rr=[];available=[];inputs=set();rank_unknown=0
    for t in sorted(required):
        for i,a in enumerate(menus.get(t,[])):
            try:r=record(p,t,a['text'],True)
            except Unsupported:rank_unknown+=1;continue
            rr.append(r);available.append((a,i));inputs.add(str(r))
    zz=score_fn(rr,'rank') if spec['rank']=='learned' and rr else [0.]*len(rr)
    if len(zz)!=len(rr) or any(not math.isfinite(z) for z in zz):raise ValueError('Invalid rank logits')
    queue=sorted(((-(z-math.log1p(i)),a['target'],a['candidate_id'],a) for (a,i),z in zip(available,zz)),key=lambda x:x[:3])
    for _,_,_,a in queue:
        if len(attempts)>=candidate_budget or not required:break
        if a['target'] not in required:continue
        valid=meter.check(a['target'],a['text']);attempts.append(dict(a,valid=valid,prospective_closure=None))
        if valid is True:
            closed=closes_obligation(p,a['target'],a['text'])
            current=apply(current,a);accepted.append(dict(a,closes_obligation=closed));attempts[-1]['closes_obligation']=closed
            if closed:required.remove(a['target']);unknown.discard(a['target'])
    open_goals=list(state.get('open_obligations',[]))
    for a in accepted:
        if a['target']=='answer':continue
        shape=syntax_role(p['question'],claim(a['text']))
        if shape['completes']:open_goals=[v for v in open_goals if v!=shape['scope']]
    audit_unknown=target is not None and audit_valid is None
    judgments=len(missing_events)+len(attempts)+int(target is not None)
    assert judgments<=8 and len(attempts)<=candidate_budget
    return dict(arm=spec['name'],configuration=copy.deepcopy(spec),status='SUPPORTED',
        initial_state=state,raw_initial_state=raw_state,initial_targets=initial_required,
        neural_initial_targets=before_audit,final_payload=current,accepted=accepted,attempts=attempts,
        remaining_targets=sorted(required),declared_complete=not required and not open_goals and not audit_unknown and not unknown,
        tool_cost=dict(meter.cost),tool_events=meter.events,rank_model_unique_inputs=len(inputs) if spec['rank']=='learned' else 0,
        rule_encoded_candidate_inputs=len(inputs) if spec['rank']=='rule' else 0,rank_encoding_unknown=rank_unknown,
        candidate_budget=candidate_budget,online_budget=8,online_judgments=judgments,
        evidence=dict(evidence),effective_evidence=ev,initial_probabilities=probs,
        missing_policy=spec['missing'],raw_unknown=sorted(raw_unknown),remaining_unknown=sorted(unknown),missing_checks=missing_events,
        audit_selector=spec['audit'],audit_eligible=eligible,audited_target=target,audit_valid=audit_valid,
        reopened_targets=reopened,unknown_audit=audit_unknown,audit_reserved_slots=reserved,
        progress_edits=sum(not a['closes_obligation'] for a in accepted),
        disclosure='None is missing evidence, not predicted incorrect. Missing checks plus audit plus candidate checks <=8; question initialization is additionally counted. S1/learned components are shared with our method.')

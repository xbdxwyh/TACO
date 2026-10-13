"""Frozen confirmation policy: original step-goal algorithm, S0/S1 ablation.

No gold, dataset metadata, test selection or optimization is available here.
S0 uses the established local controller (no retirement/open-goal history).
"""
import copy
import math
from .math12_syntax_v2 import parse_public,claim,Unsupported
from .math12_actions_v1 import learned_state,syntax_role,apply
from .math12_obligation_policy_v1 import closes_obligation
from .math12_step_goal_policy_v1 import choose_audit,prospective_closure,StepMeter
from .math12_model_v1 import record

PRIMARY='S1_learned_uncertain'


def configurations():
    result=[]
    for stateful in (False,True):
        for rank in ('learned','rule'):
            for audit in ('none','uncertain'):
                result.append(dict(name=f'S{int(stateful)}_{rank}_{audit}',stateful=stateful,
                    rank=rank,audit=audit,priority='plain',budget=8,head='new'))
    for name,audit,priority,budget,head in (
            ('S1_learned_random','random','plain',8,'new'),
            ('S1_learned_uncertain_closure','uncertain','closure',8,'new'),
            ('S1_learned_none7','none','plain',7,'new'),
            ('old_S1_learned_none','none','plain',8,'old')):
        result.append(dict(name=name,stateful=True,rank='learned',audit=audit,priority=priority,budget=budget,head=head))
    return result


def operate(payload,menus,probabilities,score_fn,spec):
    if type(spec['stateful']) is not bool or spec['priority'] not in ('plain','closure') or spec['audit'] not in ('none','random','uncertain'):
        raise ValueError('Unknown state/audit/priority')
    if spec['rank'] not in ('learned','rule') or spec['budget'] not in (7,8):raise ValueError('Unknown rank/budget')
    p=parse_public(payload);state=learned_state(p,probabilities,spec['stateful']);required=set(state['pending'])
    if probabilities['answer']<.5:required.add('answer')
    neural_targets=sorted(required);target,eligible=choose_audit(p,probabilities,state,spec['audit'])
    meter=StepMeter(p['question']);audit_valid=None;reopened=[]
    if target is not None:
        audit_valid=meter.check_original_step(target,p['nodes'][int(target[1:])-1]['display'])
        if audit_valid is False:required.add(target);reopened=[target]
    reserved=int(spec['audit']!='none');candidate_budget=spec['budget']-reserved
    initial_required=sorted(required);current=copy.deepcopy(payload);accepted=[];attempts=[]
    records=[];available=[];model_inputs=set();rank_unknown=0;shape_calls=shape_unknown=0
    for t in sorted(required):
        for index,a in enumerate(menus.get(t,[])):
            try:r=record(p,t,a['text'],True)
            except Unsupported:rank_unknown+=1;continue
            records.append(r);model_inputs.add(str(r));hint=None
            if spec['priority']=='closure':
                hint=prospective_closure(p,t,a['text']);shape_calls+=1;shape_unknown+=hint is None
            available.append((a,index,hint))
    logits=score_fn(records,'rank') if spec['rank']=='learned' and records else [0.]*len(records)
    if len(logits)!=len(records) or any(not math.isfinite(x) for x in logits):raise ValueError('Bad rank scores')
    order=[]
    for (a,index,hint),value in zip(available,logits):
        first=int(hint is not True) if spec['priority']=='closure' else 0
        order.append(((first,-(value-math.log1p(index)),a['target'],a['candidate_id']),a,hint))
    for _,a,hint in sorted(order,key=lambda x:x[0]):
        if len(attempts)>=candidate_budget or not required:break
        if a['target'] not in required:continue
        valid=meter.check(a['target'],a['text']);attempts.append(dict(a,valid=valid,prospective_closure=hint))
        if valid is True:
            closed=closes_obligation(p,a['target'],a['text'])
            current=apply(current,a);accepted.append(dict(a,closes_obligation=closed));attempts[-1]['closes_obligation']=closed
            if closed:required.remove(a['target'])
    open_obligations=list(state.get('open_obligations',[]))
    for a in accepted:
        if a['target']=='answer':continue
        shape=syntax_role(p['question'],claim(a['text']))
        if shape['completes']:open_obligations=[v for v in open_obligations if v!=shape['scope']]
    unknown_audit=target is not None and audit_valid is None
    candidate_calls=meter.cost.get('candidate_checks',0);audit_calls=meter.cost.get('step_audit_checks',0)
    assert candidate_calls==len(attempts)<=candidate_budget and audit_calls==int(target is not None)
    assert candidate_calls+audit_calls<=spec['budget']
    return dict(arm=spec['name'],configuration=copy.deepcopy(spec),status='SUPPORTED',initial_state=state,
        neural_initial_targets=neural_targets,initial_targets=initial_required,final_payload=current,
        accepted=accepted,attempts=attempts,remaining_targets=sorted(required),
        declared_complete=not required and not open_obligations and not unknown_audit,
        tool_cost=dict(meter.cost),tool_events=meter.events,rank_model_unique_inputs=len(model_inputs) if spec['rank']=='learned' else 0,
        rule_encoded_candidate_inputs=len(model_inputs) if spec['rank']=='rule' else 0,rank_encoding_unknown=rank_unknown,
        candidate_budget=candidate_budget,online_budget=spec['budget'],online_judgments=candidate_calls+audit_calls,
        state_source='neural_plus_budgeted_step_tool' if reserved else 'learned',
        initial_probabilities=dict(probabilities),audit_selector=spec['audit'],audit_eligible=eligible,
        audited_target=target,audit_valid=audit_valid,reopened_targets=reopened,unknown_audit=unknown_audit,
        audit_reserved_slots=reserved,audit_unused_reservation=bool(reserved and target is None),
        audit_state_update='frozen original retirement; add only the false audited target; no history recomputation',
        priority=spec['priority'],prospective_shape_calls=shape_calls,prospective_shape_unknown=shape_unknown,
        progress_edits=sum(not a['closes_obligation'] for a in accepted),
        feedback='one optional original-step judgment, then candidate validity and original closure contract',
        controller='completion_contract_v1',original_history_retirement='edits cannot erase other obligations')

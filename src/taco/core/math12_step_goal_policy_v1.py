"""One budgeted public step audit and prospective syntactic closure priority.

No labels/scorer. Initial retirement is frozen; a false audit reopens ONLY the
checked step, without recomputing history or erasing other obligations.
"""
import copy
import math
from .math12_io_v1 import digest
from .math12_syntax_v2 import parse_public,claim,Unsupported
from .math12_actions_v1 import learned_state,syntax_role,apply
from .math12_obligation_policy_v1 import Meter,closes_obligation
from .math12_exact_v2 import judge
from .math12_model_v1 import record

RANDOM_SALT=83109


def configurations():
    result=[]
    for priority in ('plain','closure'):
        for audit in ('none','random','uncertain'):
            result.append(dict(name=f'{priority}_{audit}',priority=priority,audit=audit,rank='learned',budget=8))
        result.append(dict(name=f'{priority}_none_budget7',priority=priority,audit='none',rank='learned',budget=7))
        for audit in ('none','uncertain'):
            result.append(dict(name=f'rule_{priority}_{audit}',priority=priority,audit=audit,rank='rule',budget=8))
    return result


def choose_audit(prepared,probabilities,state,selector):
    excluded=set(state['pending'])|set(state['retired'])
    eligible=[n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')
              and n['node_id'] not in excluded and probabilities[n['node_id']]>=.5]
    if selector=='none' or not eligible:return None,eligible
    if selector=='uncertain':return min(eligible,key=lambda t:(probabilities[t],int(t[1:]))),eligible
    if selector!='random':raise ValueError('Unknown step selector')
    # Stable public-content random priority, independent of hidden family/variant
    # ids, labels, dict order, and all later edits/rank decisions. One fixed draw.
    public=dict(question=prepared['question'],answer=prepared['answer'],
                steps=[(n['node_id'],n['display']) for n in prepared['nodes']])
    return min(eligible,key=lambda t:(digest([RANDOM_SALT,public,t]),int(t[1:]))),eligible


class StepMeter(Meter):
    def check_original_step(self,target,text):
        self.cost['step_audit_checks']+=1;reason=None
        try:valid=bool(judge(self.get(),claim(text))['admissible'])
        except Unsupported as exc:valid=None;reason=str(exc)
        # Original observations can legitimately restate a given. The candidate
        # repair check's RESTATED_GIVEN_NOT_REPAIR rule must not apply here.
        self.events.append(dict(target=target,text=text,valid=valid,reason=reason,category='step_audit_checks'))
        return valid


def prospective_closure(prepared,target,text):
    """Public shape only, NOT mathematical validity or verified completion."""
    if target=='answer':return True
    try:
        old=syntax_role(prepared['question'],prepared['nodes'][int(target[1:])-1]['claim'])
        new=syntax_role(prepared['question'],claim(text))
        if new.get('restated_given') or new['scope'] in ('cue','given'):return False
        return bool(new['completes'] and new['scope']==old['scope']) if old['completes'] else bool(new['scope']==old['scope'] or new['completes'])
    except Unsupported:return None


def operate(payload,menus,probabilities,score_fn,spec):
    if spec['priority'] not in ('plain','closure') or spec['audit'] not in ('none','random','uncertain'):
        raise ValueError('Unknown policy')
    if spec['rank'] not in ('learned','rule') or spec['budget'] not in (7,8):raise ValueError('Unknown rank/budget')
    p=parse_public(payload);state=learned_state(p,probabilities,True);required=set(state['pending'])
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
    queue=sorted(order,key=lambda x:x[0])
    for _,a,hint in queue:
        if len(attempts)>=candidate_budget or not required:break
        if a['target'] not in required:continue
        valid=meter.check(a['target'],a['text'])
        attempts.append(dict(a,valid=valid,prospective_closure=hint))
        if valid is True:
            closed=closes_obligation(p,a['target'],a['text'])
            current=apply(current,a);accepted.append(dict(a,closes_obligation=closed))
            attempts[-1]['closes_obligation']=closed
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

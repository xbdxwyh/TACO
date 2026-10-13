"""Completion-aware controller. Frozen v1 menus/models; no hidden labels.

A verified intermediate may be committed as progress, but a slot originally
expressing a goal stays open until that goal is expressed. Scope is syntactic.
This conservative controller is not a general proof-equivalence checker.
"""
import copy
import math
from collections import Counter
from .math12_syntax_v2 import Unsupported, parse_public, claim, signature
from .math12_exact_v2 import context, judge, answer_ok
from .math12_state_v2 import state_trace
from .math12_actions_v1 import apply, syntax_role, learned_state
from .math12_model_v1 import record

ARMS = tuple(f'{source}_S{s}_R{r}' for source in ('learned','tool') for s in (0,1) for r in (0,1)) + ('tool_S1_rule_R1','direct_tool_reference')
BUDGET = 8


class Meter:
    def __init__(self,q):self.q=q;self.ctx=None;self.cost=Counter();self.events=[]
    def get(self):
        if self.ctx is None:self.ctx=context(self.q);self.cost['question_solves']+=1
        return self.ctx
    def check(self,target,text,category='candidate_checks'):
        self.cost[category]+=1;reason=None
        try:
            if target=='answer':valid=answer_ok(self.get(),text)
            else:
                c=claim(text);shape=syntax_role(self.q,c)
                valid=False if shape.get('restated_given') else judge(self.get(),c)['admissible']
                if shape.get('restated_given'):reason='RESTATED_GIVEN_NOT_REPAIR'
        except Unsupported as exc:valid=None;reason=str(exc)
        self.events.append(dict(target=target,text=text,valid=valid,reason=reason,category=category))
        return valid


def initial(prepared,meter,probabilities,source,stateful):
    if source=='learned':
        state=learned_state(prepared,probabilities,stateful)
        return state,probabilities['answer']>=.5
    js=[]
    for n in prepared['nodes']:
        if n['claim']['kind'] in ('cue','given'):j=dict(scope=n['claim']['kind'],admissible=True,completes=False)
        else:
            meter.cost['original_step_checks']+=1
            j=judge(meter.get(),n['claim'])
            shape=syntax_role(prepared['question'],n['claim']);j.update(shape)
        js.append(j)
    state=state_trace(prepared['nodes'],js,prepared['question']['kind'])
    if not stateful:state=dict(state,pending=[n['node_id'] for n,j in zip(prepared['nodes'],js) if not j['admissible']],retired=[])
    answer=meter.check('answer',prepared['answer'],'original_answer_checks')
    return state,answer is True


def closes_obligation(prepared, target, text):
    """Called only AFTER a true mathematical check; never changes truth labels."""
    if target == 'answer':
        return True
    old = syntax_role(prepared['question'], prepared['nodes'][int(target[1:])-1]['claim'])
    new = syntax_role(prepared['question'], claim(text))
    if new.get('restated_given') or new['scope'] in ('cue', 'given'):
        return False
    if old['completes']:
        return new['completes'] and new['scope'] == old['scope']
    # Preserve legitimate progress-to-goal promotions; do not ban all scope changes.
    return new['scope'] == old['scope'] or new['completes']


def operate(payload,menus,probabilities,score_fn,arm,budget=BUDGET):
    p=parse_public(payload);meter=Meter(p['question']);source=arm.split('_')[0]
    stateful='_S1_' in arm;adaptive=arm.endswith('R1');rule=arm=='tool_S1_rule_R1'
    state,aok=initial(p,meter,probabilities,source,stateful)
    required=set(state['pending'])
    if not aok:required.add('answer')
    initial_required=sorted(required);current=copy.deepcopy(payload);accepted=[];attempts=[];visited=set()
    model_inputs=set();rank_unknown=0
    def order(prepared,targets):
        nonlocal rank_unknown
        available=[];records=[]
        for target in sorted(targets):
            for index,a in enumerate(menus.get(target,[])):
                key=(target,a['candidate_id'])
                if key in visited:continue
                if rule:available.append((a,-math.log1p(index)));continue
                try:r=record(prepared,target,a['text'],True)
                except Unsupported:rank_unknown+=1;continue
                records.append(r);available.append((a,index));model_inputs.add(str(r))
        if not rule:
            logits=score_fn(records,'rank')
            available=[(a,score-math.log1p(index)) for (a,index),score in zip(available,logits)]
        return [a for a,score in sorted(available,key=lambda x:(-x[1],x[0]['target'],x[0]['candidate_id']))]
    queue=order(p,required) if required else []
    while required and len(attempts)<budget and queue:
        a=queue.pop(0);key=(a['target'],a['candidate_id'])
        if a['target'] not in required or key in visited:continue
        visited.add(key);valid=meter.check(a['target'],a['text'])
        attempts.append(dict(**a,valid=valid))
        if valid is True:
            closed=closes_obligation(p,a['target'],a['text'])
            current=apply(current,a);accepted.append(dict(a,closes_obligation=closed))
            attempts[-1]['closes_obligation']=closed
            if closed:required.remove(a['target'])
            # Accepted edits fix that obligation only. They cannot silently
            # retire a DIFFERENT original error as an observed self-correction.
            if adaptive and required:queue=order(parse_public(current),required)
    open_obligations=state.get('open_obligations',[])
    # If a required goal was repaired, its successful check establishes that
    # obligation. Original missed errors remain visible to independent scoring.
    for a in accepted:
        if a['target']=='answer':continue
        shape=syntax_role(p['question'],claim(a['text']))
        if shape['completes']:open_obligations=[v for v in open_obligations if v!=shape['scope']]
    return dict(arm=arm,status='SUPPORTED',initial_state=state,initial_targets=initial_required,
        final_payload=current,accepted=accepted,attempts=attempts,remaining_targets=sorted(required),
        declared_complete=not required and not open_obligations,tool_cost=dict(meter.cost),
        tool_events=meter.events,rank_model_unique_inputs=len(model_inputs),rank_encoding_unknown=rank_unknown,
        candidate_budget=budget,state_source=source,feedback='candidate validity, then public syntactic closure; truth unchanged',
        controller='completion_contract_v1',
        progress_edits=sum(not a['closes_obligation'] for a in accepted),
        original_history_retirement='observed corrections only; edits cannot erase other obligations')


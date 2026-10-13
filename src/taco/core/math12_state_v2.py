"""Typed public state contract and syntax-only repair menus for interface audit.

No training, hidden labels, solver-generated candidates, or online neural model.
The Fraction tool is called explicitly; all judgments have a tool cost.
"""
import re
from .math12_syntax_v2 import Unsupported, parse_public, claim, signature, plain
from .math12_exact_v2 import context, judge, answer_ok

MENU_CAP=32
PROBE_CAP=4


def menu(text):
    """Only one public surface edit per candidate. No reference answer insertion."""
    result=[]; seen={text}
    def push(t,operation):
        if t not in seen and len(result)<MENU_CAP:
            seen.add(t);result.append({'candidate_id':f'c{len(result)}','text':t,'operation':operation})
    s=text
    for match in re.finditer(r'[+\-*/<>]',s):
        old=match.group()
        for new in {'+':('-',),'−':('+',),'-':('+',),'*':('+','/'),'/':('*',),'<':('>',),'>':('<',)}[old]:
            push(s[:match.start()]+new+s[match.end():],'operator_replace')
    for match in re.finditer(r'(?<![A-Za-z])\d+',s):
        v=int(match.group())
        for n in (v-1,v+1,-v):push(s[:match.start()]+str(n)+s[match.end():],'integer_replace')
    for a,b in [('f(g(', 'g(f('),('g(f(', 'f(g(')]:
        if a in s:push(s.replace(a,b,1),'composition_order')
    return result


def state_trace(nodes,judgments,kind):
    """Only original observed corrections may retire original history.

Scope-aware restart: total -> mean and inner -> composed target reopen the
downstream obligation. A true y step cannot erase an erroneous x step.
"""
    pending={}; retired=[]; snapshots=[]; achieved=set(); rejoins=[]
    required={'x','y'} if kind=='system' else {'goal'}
    for node,j in zip(nodes,judgments):
        iid=node['node_id']; role=j['scope']
        if role not in ('cue','given'):
            if not j['admissible']:
                pending[iid]=role; achieved.discard(role)
                if role in ('inner','total'):achieved.discard('goal')
            elif j.get('restated_given',False):pass
            else:
                clear={role}
                if (kind=='weighted' and role=='total') or (kind=='function' and role=='inner'):
                    clear.add('goal')
                    if any(v=='goal' for v in pending.values()):achieved.discard('goal')
                gone=[n for n,s in pending.items() if s in clear]
                if gone:
                    retired.extend(gone);rejoins.append({'at':iid,'retired':gone})
                    for n in gone:del pending[n]
                if j['completes']:achieved.add(role)
        snapshots.append({'through':iid,'pending':sorted(pending),'retired':list(retired),
                          'open_obligations':sorted(required-achieved)})
    return {'snapshots':snapshots,'pending':sorted(pending),'retired':retired,'rejoins':rejoins,
            'open_obligations':sorted(required-achieved)}


def predict_public(payload):
    try:
        prepared=parse_public(payload); ctx=context(prepared['question'])
        judgments=[]; menus={}; probes=[]
        for node in prepared['nodes']:
            c=node['claim']; j=judge(ctx,c)
            given=prepared['question'].get('claim')
            if given and c['kind'] not in ('given','cue') and signature(c)==signature(given):
                j=dict(j,restated_given=True,completes=False)
            judgments.append(j)
            if c['kind'] not in ('given','cue'):menus[node['node_id']]=menu(node['display'])
        aok=answer_ok(ctx,prepared['answer']); menus['answer']=menu(prepared['answer'])
        # A tiny deterministic public menu probe is an engineering cross-check,
        # not a candidate coverage estimate or recovery performance experiment.
        targets=list(menus)
        for offset in range(PROBE_CAP):
            target=targets[offset%len(targets)] if targets else None
            if target is None or not menus[target]:continue
            candidate=menus[target][(offset//len(targets))%len(menus[target])]
            try: valid=answer_ok(ctx,candidate['text']) if target=='answer' else judge(ctx,claim(candidate['text']))['admissible']
            except Unsupported:valid=None
            probes.append({'target':target,**candidate,'valid':valid})
        state=state_trace(prepared['nodes'],judgments,ctx['kind'])
        accepted=not state['pending'] and not state['open_obligations'] and aok
        return {'status':'SUPPORTED','prepared':prepared,'judgments':judgments,'state':state,
                'answer_correct':aok,'declared_complete':accepted,'menus':menus,'probes':probes,
                'cost':{'question_tool_solves':1,'original_step_checks':len(judgments),
                        'answer_checks':1,'candidate_probe_checks':len(probes)},
                'source':'EXPLICIT_EXACT_TOOL_NOT_NEURAL','recovery_executed':False}
    except Unsupported as exc:
        return {'status':'UNKNOWN','reason':str(exc),'source':'EXPLICIT_EXACT_TOOL_NOT_NEURAL',
                'recovery_executed':False}

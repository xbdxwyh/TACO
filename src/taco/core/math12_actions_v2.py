"""Shared public algebra/continuation menus; no reference solution injection.

Preserves v1's completed pilot. New candidates normalize written equations or
continue from earlier *possible* edits, before any validity labels are available.
This is explicitly symbolic candidate construction, not neural generation.
"""
from .math12_actions_v1 import (apply,syntax_role,learned_state,constant_fold,render,
    answer_from_step,edit_distance,MENU_CAP,build as base_build)
from .math12_syntax_v2 import Unsupported,claim,strip_question
from .math12_exact_v2 import value,mul,add,ZERO
from .math12_policy_v1 import polynomial_text


def equation_candidates(text,isolate):
    c=claim(strip_question(text))
    if c['kind']!='chain' or len(c['asts'])<2:return []
    left,right=(value(a) for a in c['asts'][:2])
    a=mul(left[0],right[1]);b=mul(right[0],left[1]);p=add(a,b,-1)
    out=[('clear_written_denominators',polynomial_text(a)+'='+polynomial_text(b))]
    names=set(p)-{ZERO}
    if isolate and len(names)==1:
        (i,j)=next(iter(names))
        if (i,j) in ((1,0),(0,1)):
            v='x' if i else 'y';out.append(('isolate_written_variable',v+'='+str(-p.get(ZERO,0)/p[(i,j)])))
    return out


def build(payload):
    data=base_build(payload);p=data['prepared'];q=p['question'];menus={};previous=[]
    cost=data['generation_cost'];cost.update(algebra_normalization_attempts=0,continuation_copy_attempts=0)
    nodes={n['node_id']:n for n in p['nodes']}
    for target,base in data['menus'].items():
        original=p['answer'] if target=='answer' else nodes[target]['display'];extra=[];seen={original}
        def push(text,operation):
            if text in seen or len(text)>1024:return
            try:
                if target=='answer' and q['kind']=='system':
                    for part in text.split(','):claim(part)
                else:claim('x='+text if target=='answer' and q['kind']=='equation' and '=' not in text else text)
            except Unsupported:return
            seen.add(text);extra.append(dict(target=target,text=text,operation=operation))
        if target!='answer' and q['kind'] in ('equation','system'):
            # Given equations may be cleared of denominators, but are not solved
            # here. Isolation starts from actual/possible written intermediate work.
            sources=[(original,True)]
            if q['kind']=='equation' and target=='s2':sources.append((nodes['s1']['display'],False))
            sources+=previous
            for text,isolate in sources:
                cost['algebra_normalization_attempts']+=1
                try:
                    for operation,newtext in equation_candidates(text,isolate):push(newtext,operation)
                except Unsupported:continue
        if target=='answer':
            for n in p['nodes']:
                if n['node_id'] not in menus:continue
                for a in menus[n['node_id']]:
                    cost['continuation_copy_attempts']+=1
                    try:push(answer_from_step(q,a['text'],original),'copy_proposed_visible_value')
                    except Unsupported:pass
            if q['kind']=='system':
                assignments={'x':[], 'y':[]}
                for text in [a['text'] for mm in menus.values() for a in mm]:
                    try:c=claim(text)
                    except Unsupported:continue
                    aa=c.get('asts',[])
                    if len(aa)!=2 or aa[0] not in (('var','x'),('var','y')):continue
                    value_ast=constant_fold(aa[1])
                    if value_ast[0]!='num':continue
                    key=aa[0][1];val=render(value_ast)
                    if val not in assignments[key]:assignments[key].append(val)
                for x,y in zip(assignments['x'],assignments['y']):push(f'x={x}, y={y}','combine_visible_assignments')
                for x in assignments['x'][:6]:
                    for y in assignments['y'][:6]:push(f'x={x}, y={y}','combine_visible_assignments')
        # Stable interleaving reserves slots for public continuation without
        # displacing every local edit. Rule ordering never consults validity.
        joined=[];dedup={original}
        for i in range(max(len(base),len(extra))):
            for lane in (base,extra):
                if i<len(lane) and lane[i]['text'] not in dedup:
                    dedup.add(lane[i]['text']);joined.append(lane[i])
        joined=joined[:MENU_CAP]
        joined.sort(key=lambda a:(edit_distance(original,a['text']),a['operation'],a['text']))
        menus[target]=[dict(a,candidate_id=f'c{i}') for i,a in enumerate(joined)]
        if target!='answer':previous=[(nodes[target]['display'],True)]+[(a['text'],True) for a in menus[target]]
    return dict(data,menus=menus,generation_cost=cost)

"""Post-seal independent scoring. Never imported by the public policy module."""
from collections import Counter, defaultdict
import numpy as np
from .math12_syntax_v2 import Unsupported, parse_public
from .math12_reference_v2 import context_ref, judge_ref, answer_ref
from .math12_actions_v1 import syntax_role, learned_state


def original_truth(payload,gold):
    p=parse_public(payload);boundary=gold['source_gold']['repair_step']
    ids={n['node_id'] for n in p['nodes'] if n['claim']['kind'] not in ('given','cue')}
    bad={f's{r["step_index"]}' for r in gold['steps'] if not r['globally_valid'] and f's{r["step_index"]}' in ids}
    retired={i for i in bad if boundary is not None and int(i[1:])<boundary}
    return p,bad,retired,boundary


def state_metrics(payload,gold,probabilities):
    p,bad,retired,boundary=original_truth(payload,gold);out={}
    math_ids=[n['node_id'] for n in p['nodes'] if n['claim']['kind'] not in ('given','cue')]
    for s in (0,1):
        state=learned_state(p,probabilities,bool(s));counts=Counter()
        for iid in math_ids:
            actual=iid in bad;pred=probabilities[iid]<.5
            counts['step_'+('tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn')]+=1
        snapshots={x['through']:x for x in state['snapshots']}
        for n in p['nodes']:
            t=int(n['node_id'][1:]);available={i for i in math_ids if int(i[1:])<=t}
            expected=(bad & available)-(retired if boundary is not None and t>=boundary else set())
            predicted=set(snapshots[n['node_id']]['pending']) if s else {i for i in available if probabilities[i]<.5}
            counts['prefix_sets']+=1;counts['prefix_exact']+=predicted==expected
            for i in available:
                actual=i in expected;pred=i in predicted
                counts['active_'+('tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn')]+=1
        counts['final_set_exact']=set(state['pending'])==bad-retired
        counts['retired_set_exact']=set(state['retired'])==retired
        counts['false_retired_active']=len(set(state['retired']) & (bad-retired))
        out['S'+str(s)]=dict(counts)
    return out


def final_score(original,gold,result):
    _,bad,retired,boundary=original_truth(original,gold)
    if result['status']!='SUPPORTED':
        return dict(supported=False,complete=False,declared=False,false_accept=False,edits=0,unknown=True,failures=['PUBLIC_INTERFACE_UNKNOWN'])
    failures=[];unknown=[];achieved=set()
    try:
        p=parse_public(result['final_payload']);ref=context_ref(p['question'])
        edited={a['target'] for a in result['accepted']}
        for n in p['nodes']:
            iid=n['node_id']
            if n['claim']['kind'] in ('cue','given'):continue
            # Only original observed retractions are honored. An edit at another
            # node does not grant a fresh retraction of an unrepaired bad node.
            if iid in retired and iid not in edited:continue
            try:valid=bool(judge_ref(ref,n['claim']))
            except Unsupported as e:unknown.append([iid,str(e)]);continue
            if not valid:failures.append(iid)
            shape=syntax_role(p['question'],n['claim'])
            if valid and shape['completes']:achieved.add(shape['scope'])
        try:answer=bool(answer_ref(ref,p['answer']))
        except Unsupported as e:answer=False;unknown.append(['answer',str(e)])
        if not answer:failures.append('answer')
        required={'x','y'} if p['question']['kind']=='system' else {'goal'}
        failures+=['open:'+s for s in sorted(required-achieved)]
    except Unsupported as e:unknown.append(['trace',str(e)])
    complete=not failures and not unknown
    return dict(supported=True,complete=complete,declared=result['declared_complete'],
        false_accept=bool(result['declared_complete'] and not complete),edits=len(result['accepted']),
        unknown=bool(unknown),unknown_detail=unknown,failures=failures,
        candidate_checks=result['tool_cost'].get('candidate_checks',0),
        tool_calls=sum(result['tool_cost'].values()),rank_model_unique_inputs=result['rank_model_unique_inputs'],
        original_error_count=len(bad-retired))


def ratio(n,d):return n/d if d else None


def panel(rows):
    erroneous=[r for r in rows if r['variant'] in ('logical_error','execution_slip')]
    clean=[r for r in rows if r['variant'] in ('correct','self_corrected')]
    return dict(n=len(rows),error_n=len(erroneous),clean_n=len(clean),
        repair_success=sum(r['complete'] for r in erroneous),repair_rate=ratio(sum(r['complete'] for r in erroneous),len(erroneous)),
        clean_edited=sum(r['edits']>0 for r in clean),clean_broken=sum(not r['complete'] for r in clean),
        false_accept=sum(r['false_accept'] for r in rows),unknown=sum(r['unknown'] for r in rows),
        unresolved=sum(not r['declared'] for r in rows),unsupported=sum(not r['supported'] for r in rows),
        edits=sum(r['edits'] for r in rows),candidate_checks=sum(r.get('candidate_checks',0) for r in rows),
        tool_calls=sum(r.get('tool_calls',0) for r in rows),rank_model_unique_inputs=sum(r.get('rank_model_unique_inputs',0) for r in rows))


def summarize(rows,state_rows):
    panels={};paired=[]
    for seed in sorted({r['seed'] for r in rows}):
        subset=[r for r in rows if r['seed']==seed]
        for arm in sorted({r['arm'] for r in subset}):
            rr=[r for r in subset if r['arm']==arm]
            panels[f'{arm}/seed{seed}']=dict(overall=panel(rr),by_skill={k:panel([r for r in rr if r['skill']==k]) for k in sorted({r['skill'] for r in rr})},
                by_variant={v:panel([r for r in rr if r['variant']==v]) for v in sorted({r['variant'] for r in rr})})
            rates=[x['repair_rate'] for x in panels[f'{arm}/seed{seed}']['by_skill'].values() if x['repair_rate'] is not None]
            panels[f'{arm}/seed{seed}']['skill_macro_repair_rate']=float(np.mean(rates)) if rates else None
        for source in ('learned','tool'):
            for baseline in (source+'_S0_R1',source+'_S1_R0','tool_S1_rule_R1'):
                treatment=source+'_S1_R1'
                a={r['item_id']:r for r in subset if r['arm']==treatment and r['variant'] in ('logical_error','execution_slip')}
                b={r['item_id']:r for r in subset if r['arm']==baseline and r['variant'] in ('logical_error','execution_slip')}
                assert a.keys()==b.keys()
                groups=defaultdict(list)
                for iid,r in a.items():groups[r['family']].append(float(r['complete'])-float(b[iid]['complete']))
                diffs=np.array([np.mean(x) for x in groups.values()])
                if not len(diffs):continue
                rng=np.random.default_rng(4401);boot=np.mean(diffs[rng.integers(len(diffs),size=(2000,len(diffs)))],axis=1)
                paired.append(dict(seed=seed,treatment=treatment,baseline=baseline,unit='family',families=len(diffs),
                    repair_delta=float(np.mean(diffs)),ci95=np.quantile(boot,[.025,.975]).tolist()))
    state={}
    for seed in sorted({r['seed'] for r in state_rows}):
        for key in ('S0','S1'):
            counts=Counter()
            selected=[r for r in state_rows if r['seed']==seed]
            for r in selected:counts.update(r['metrics'][key])
            n=len(selected);d=dict(counts)
            for prefix in ('step','active'):
                tp,tn,fp,fn=(counts[prefix+'_'+x] for x in ('tp','tn','fp','fn'))
                d[prefix+'_f1']=ratio(2*tp,2*tp+fp+fn)
                tpr=ratio(tp,tp+fn);tnr=ratio(tn,tn+fp)
                d[prefix+'_balanced_accuracy']=(tpr+tnr)/2 if tpr is not None and tnr is not None else None
            d['n']=n;d['final_set_exact_rate']=ratio(counts['final_set_exact'],n)
            d['prefix_exact_rate']=ratio(counts['prefix_exact'],counts['prefix_sets'])
            state[f'{key}/seed{seed}']=d
    return dict(panels=panels,state=state,paired=paired)

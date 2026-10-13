"""Public-text-only PRM adapter. No labels, solver, dataset IDs, or tuning."""
import re

VERSION = 'scoped_public_judgments_v3'
BOX = re.compile(r'\\boxed\{\s*(correct|incorrect)\s*\}', re.I)
HEAD = re.compile(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:(?:Starting with|Moving on to|Now,?|Finally,)\s+)?Step\s+(\d+)\s*(?:\*\*)?\s*[:.)]')
GLOBAL = re.compile(r'(?im)^[ \t]*(?:#{1,6}\s*|\*\*)?(?:Final Answer|Overall (?:Conclusion|Verdict)|Conclusion)\b[ \t]*:?(?:\*\*)?[ \t]*:?')
DIRECT_BOX = re.compile(r'(?i)\bStep\s+(\d+)\s*(?:\*\*)?\s+is\s*(?:\*\*)?\s*$')
# Accept only direct categorical assertions, not "may", "seems", quotations,
# assumed implications, or arbitrary sentiment in a rationale paragraph.
DIRECT_WORD = re.compile(r'(?im)^(?:[ \t]*(?:So|Therefore|Thus|Hence|Clearly),?\s+)?[ \t]*(?:\*\*)?Step\s+(\d+)\s*(?:\*\*)?\s+is\s+(correct|incorrect)\s*[.!]?(?:\*\*)?[ \t]*(?:\n|$)')


def parse(text, count, finish):
    if count<1:raise ValueError('Positive count required')
    events=[];issues=[];labels=[None]*count;by={}
    headers=list(HEAD.finditer(text));global_headers=list(GLOBAL.finditer(text))
    previous=-1
    for b in BOX.finditer(text):
        direct=DIRECT_BOX.search(text[:b.start()]);pos=None;kind='unscoped_box'
        hs=[h for h in headers if previous<h.start()<b.start()]
        gs=[g for g in global_headers if previous<g.start()<b.start()]
        if direct and direct.start()>previous:
            pos=int(direct[1]);kind='direct_numbered_box'
        elif hs and (not gs or hs[-1].start()>gs[-1].start()):
            pos=int(hs[-1][1]);kind='section_box'
        elif gs:kind='global_box_not_a_node'
        e=dict(position=pos,label=b[1].lower()=='correct',kind=kind,span=[b.start(),b.end()])
        events.append(e);previous=b.end()
    for m in DIRECT_WORD.finditer(text):
        events.append(dict(position=int(m[1]),label=m[2].lower()=='correct',kind='direct_numbered_word',span=[m.start(),m.end()]))
    for e in events:
        p=e['position']
        if p is None:continue
        if not 1<=p<=count:issues.append('number_out_of_range');continue
        by.setdefault(p,[]).append(e['label'])
    conflicts=[]
    for p,vs in by.items():
        if len(set(vs))==1:labels[p-1]=vs[0]
        else:conflicts.append(p)
    if conflicts:issues.append('contradictory_direct_judgments')
    end=re.search(r'Is the solution correct\?\s*(Yes|No)\s*[.!]?\s*$',text)
    verdict=(end[1]=='Yes') if end and finish=='eos' else None
    # Universal Yes means all supplied steps were accepted; No never identifies
    # an individual node. Conflicting local judgments forbid universal filling.
    global_filled=[]
    if verdict is True and not conflicts and False not in labels:
        global_filled=[i+1 for i,v in enumerate(labels) if v is None]
        labels=[True]*count
    elif verdict is True and (conflicts or False in labels):issues.append('global_yes_local_conflict')
    if finish!='eos':
        labels=[None]*count;global_filled=[];issues.append('truncated_unavailable')
    first=dict(status='unavailable',position=None)
    for i,v in enumerate(labels,1):
        if v is None:break
        if v is False:first=dict(status='error',position=i);break
    else:first=dict(status='no_error',position=None)
    return dict(version=VERSION,labels=labels,events=events,conflicts=conflicts,
        issues=sorted(set(issues)),prefix_verdict=verdict,first_error=first,
        mapped_count=sum(v is not None for v in labels),global_yes_filled=global_filled,
        complete_labels=all(v is not None for v in labels))


def evidence(public_case, parsed, prepared):
    math_ids={n['node_id'] for n in prepared['nodes'] if n['claim']['kind'] not in ('given','cue')}|{'answer'}
    values={t:v for t,v in zip(public_case['targets'],parsed['labels']) if t in math_ids}
    if set(values)!=math_ids:raise ValueError('PRM target identities differ')
    # Binary markers are NOT calibrated probabilities. Unknowns remain None.
    return {t:None if v is None else float(v) for t,v in values.items()}

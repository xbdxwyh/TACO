"""Public, bounded, shared repair actions. No solver, labels, or case metadata.

Menus are built once. Rules expose public algebraic rewrites, including branch
completion; they do not insert a reference solution. Unknowns remain explicit.
"""
import copy
import re
from fractions import Fraction as F
from .math12_syntax_v2 import Unsupported, parse_public, claim, plain, variables, signature
from .math12_state_v2 import menu as surface_menu, state_trace

MENU_CAP = 48


def render(a):
    k = a[0]
    if k in ('num', 'var'): return a[1]
    if k == 'neg': return '-(' + render(a[1]) + ')'
    if k == 'abs': return '|' + render(a[1]) + '|'
    if k == 'call': return a[1] + '(' + render(a[2]) + ')'
    return '(' + render(a[1]) + k + render(a[2]) + ')'


def constant_fold(a):
    if a[0] in ('num', 'var'): return a
    if a[0] == 'call': return ('call', a[1], constant_fold(a[2]))
    args = [constant_fold(x) for x in a[1:]]
    k = a[0]
    if all(x[0] == 'num' for x in args):
        xs = [F(x[1]) for x in args]
        try:
            if k == 'neg': value = -xs[0]
            elif k == 'abs': value = abs(xs[0])
            elif k == '+': value = xs[0] + xs[1]
            elif k == '-': value = xs[0] - xs[1]
            elif k == '*': value = xs[0] * xs[1]
            elif k == '/': value = xs[0] / xs[1]
            elif k == '^' and xs[1].denominator == 1 and abs(xs[1]) <= 16: value = xs[0] ** int(xs[1])
            else: return (k, *args)
            if max(value.numerator.bit_length(), value.denominator.bit_length()) <= 128:
                return ('num', str(value))
        except ZeroDivisionError: pass
    return (k, *args)


def normalized(text):
    c = claim(text)
    if c['kind'] == 'or': return ' or '.join(normalized_branch(x) for x in c['branches'])
    return normalized_branch(c)


def normalized_branch(c):
    if 'asts' not in c: raise Unsupported('NO_REPAIR_CLAIM')
    asts = [constant_fold(a) for a in c['asts']]
    # Recompute a written arithmetic RHS chain from its first RHS expression.
    # This is a syntax rule, not a check against the problem's solution.
    if len(asts) > 2: asts = asts[:2]
    text = (' ' + c['op'] + ' ').join(map(render, asts)) if c['kind'] == 'inequality' else ' = '.join(map(render, asts))
    if c.get('operation'): text = '(1)' + c['operation'] + '(2): ' + text
    return text


def syntax_role(q, c):
    """Scope/completion shape without evaluating expressions or solving q."""
    k = c['kind']; scope = 'goal'
    if k in ('cue', 'given'): return dict(scope=k, completes=False)
    a = c.get('asts', [None])[0]
    if q['kind'] == 'system' and a is not None:
        names = variables(a); scope = next(iter(names)) if len(names) == 1 else 'system'
    elif q['kind'] == 'function' and a == q['target'][2]: scope = 'inner'
    elif q['kind'] == 'weighted' and a == ('var', 'S'): scope = 'total'
    given = q.get('claim')
    restated = given is not None and signature(c) == signature(given)
    return dict(scope=scope, completes=not restated and (scope == 'goal' or (q['kind'] == 'system' and scope in ('x', 'y'))), restated_given=restated)


def public_rules(q):
    """Small algebraic rewrite library with alternatives, shared by every arm."""
    result = []
    if q['kind'] == 'equation':
        c = q['claim']
        if c['kind'] == 'chain' and c['asts'][0][0] == 'abs':
            inner, r = c['asts'][0][1], c['asts'][1]
            result.append(('branch_completion', render(inner) + '=' + render(r) + ' or ' + render(inner) + '=-(' + render(r) + ')'))
            if inner[0] in ('-', '+') and inner[1] == ('var', 'x'):
                center = inner[2] if inner[0] == '-' else ('neg', inner[2])
                result.append(('branch_completion', 'x=' + render(constant_fold(('+', center, r))) + ' or x=' + render(constant_fold(('-', center, r)))))
    if q['kind'] == 'expression':
        a = q['claim']['asts'][0]
        if a[0] == '*' and a[1][0] == '^' and a[2][0] == '^' and a[1][1] == a[2][1]:
            for op in ('+', '*', '-'):
                result.append(('power_rule', render(( '^', a[1][1], constant_fold((op, a[1][2], a[2][2]))))))
        if a[0] == '^' and a[1][0] == '^':
            for op in ('*', '+', '-'):
                result.append(('power_rule', render(('^', a[1][1], constant_fold((op, a[1][2], a[2]))))))
        if a[0] == '*' and a[1][0] in ('+', '-') and a[2][0] in ('+', '-'):
            for op in ('+', '-'):
                result.append(('binomial_rule', render((op, ('*', a[1][1], a[2][1]), constant_fold(('*', a[1][2], a[2][2]))))))
    if q['kind'] == 'weighted':
        n, a, m, b = q['values']
        for denominator in (n+m, 2, n, m):
            result.append(('weighted_rule', f'mean=({n}*{a}+{m}*{b})/{denominator}'))
        result.append(('weighted_total', f'S={n}*{a}+{m}*{b}'))
    if q['kind'] == 'function':
        target = q['target']; inner = target[2]
        def substitute(a, replacement):
            if a == ('var', 'x'): return replacement
            if a[0] in ('num', 'var'): return a
            return (a[0], *[substitute(x, replacement) if isinstance(x, tuple) else x for x in a[1:]])
        for call in (inner, target, ('call', inner[1], ('call', target[1], inner[2]))):
            def expand(a):
                if a[0] == 'call': return substitute(q['defs'][a[1]], expand(a[2]))
                return a
            result.append(('definition_substitution', render(call) + '=' + render(constant_fold(expand(call)))))
    return result


def answer_from_step(q, text, old_answer):
    """Copy a visible completed expression, never calculate a hidden solution."""
    c = claim(text)
    if c['kind'] in ('or', 'inequality'): return text
    a = c.get('asts', [])
    if not a: return old_answer
    if q['kind'] == 'system':
        if len(a) < 2 or a[0] not in (('var', 'x'), ('var', 'y')): return old_answer
        key = a[0][1]; parts = old_answer.split(',')
        return ', '.join(key + '=' + render(a[-1]) if x.strip().startswith(key+'=') else x.strip() for x in parts)
    if q['kind'] == 'equation' and (len(a) < 2 or a[0] != ('var', 'x')): return old_answer
    if q['kind'] == 'function' and a[0] != q['target'] and a[0][0] == 'call': return old_answer
    if q['kind'] == 'weighted' and a[0] == ('var', 'S'): return old_answer
    return render(a[-1])


def build(payload):
    p = parse_public(payload); q = p['question']; rules = public_rules(q)
    menus = {}; generation = dict(surface_candidates=0, normalization_attempts=0, public_rule_candidates=len(rules))
    targets = [(n['node_id'], n['display']) for n in p['nodes'] if n['claim']['kind'] not in ('cue', 'given')]
    targets.append(('answer', p['answer']))
    for target, original in targets:
        lanes = [[], [], []]; seen = {original}
        def push(lane, text, op):
            if text not in seen and len(text) <= 1024:
                try:
                    if target == 'answer' and q['kind'] == 'system':
                        for part in text.split(','): claim(part)
                    else: claim(text if target != 'answer' or q['kind'] != 'equation' or '=' in text else 'x='+text)
                except Unsupported: return
                seen.add(text); lanes[lane].append(dict(text=text, operation=op))
        for c in surface_menu(original):
            generation['surface_candidates'] += 1
            push(0, c['text'], c['operation'])
            generation['normalization_attempts'] += 1
            try: push(1, normalized(c['text']), 'normalize_after_'+c['operation'])
            except Unsupported: pass
        try: push(1, normalized(original), 'constant_fold')
        except Unsupported: pass
        for op, text in rules:
            if target == 'answer':
                try: text = answer_from_step(q, text, original)
                except Unsupported: continue
            push(2, text, op)
        # Public earlier expressions can supply a subsequent displayed value.
        for n in p['nodes']:
            if n['claim']['kind'] in ('cue', 'given'): continue
            try:
                text = answer_from_step(q, normalized(n['display']), p['answer'])
                if target != 'answer' and q['kind'] == 'equation' and '=' not in text: text = 'x='+text
                push(2, text, 'visible_value_copy')
            except Unsupported: pass
        merged = []
        for i in range(max(map(len, lanes), default=0)):
            for lane in lanes:
                if i < len(lane): merged.append(lane[i])
        # Deterministic surface-distance rule, not correctness or hidden labels.
        merged = merged[:MENU_CAP]  # reserve interleaved slots before rule ordering
        merged.sort(key=lambda c: (edit_distance(original, c['text']), c['operation'], c['text']))
        menus[target] = [dict(c, candidate_id=f'c{i}', target=target) for i, c in enumerate(merged)]
    return dict(prepared=p, menus=menus, generation_cost=generation)


def edit_distance(a, b):
    # Cheap deterministic edit proxy; not claimed to be Levenshtein distance.
    return abs(len(a)-len(b)) + sum(x != y for x,y in zip(a,b))


def apply(payload, action):
    result = copy.deepcopy(payload); target = action['target']; text = action['text']
    if target == 'answer': result['student_answer'] = text
    else:
        index = int(target[1:])-1
        if index <= 0 or index >= len(result['steps']): raise ValueError('Invalid edit target')
        result['steps'][index]['display'] = text
        # Both channels carry the same public expression after an explicit edit.
        result['steps'][index]['latex'] = text
    return result


def learned_state(prepared, probabilities, stateful):
    judgments=[]
    for n in prepared['nodes']:
        shape=syntax_role(prepared['question'], n['claim'])
        valid=True if shape['scope'] in ('cue','given') else probabilities[n['node_id']] >= .5
        judgments.append(dict(shape, admissible=valid, semantic_valid=valid, task_aligned=True))
    if stateful: return state_trace(prepared['nodes'], judgments, prepared['question']['kind'])
    pending=[n['node_id'] for n,j in zip(prepared['nodes'],judgments) if not j['admissible']]
    return dict(pending=pending, retired=[], snapshots=[], open_obligations=[])

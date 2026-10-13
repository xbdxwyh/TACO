"""Bounded public mathematical syntax. No eval, sympify(string), or gold access."""
import re
from fractions import Fraction


class Unsupported(ValueError):
    pass


def plain(s):
    if not isinstance(s, str) or len(s) > 1024:
        raise Unsupported('TEXT_LIMIT')
    s = s.strip().replace('−', '-').replace('×', '*').replace('÷', '/')
    s = s.replace('，', ',').replace('；', ';').replace('：', ':')
    s = s.replace(r'\begin{cases}', '').replace(r'\end{cases}', '').replace('\\\\', ';')
    s = re.sub(r'\\text\{\s*(Solve|Expand|Simplify)\s*\}', '', s)
    s = re.sub(r'\\text\{\s*(?:or|或)\s*\}', ' or ', s)
    s = re.sub(r'\\bar\s*x', 'mean', s)
    for a, b in [(r'\times', '*'), (r'\cdot', '*'), (r'\leq', '<='),
                 (r'\geq', '>='), ('≤', '<='), ('≥', '>='),
                 (r'\left', ''), (r'\right', ''), (r'\;', ' '), (r'\,', ' ')]:
        s = s.replace(a, b)
    def group(pos):
        if pos >= len(s) or s[pos] != '{': raise Unsupported('BAD_LATEX_GROUP')
        end, depth = pos+1, 1
        while end < len(s) and depth:
            depth += (s[end] == '{') - (s[end] == '}'); end += 1
        if depth: raise Unsupported('UNBALANCED_LATEX')
        return s[pos+1:end-1], end
    for _ in range(64):
        pos = s.find(r'\frac')
        if pos < 0: break
        top, end = group(pos+5); bottom, end = group(end)
        s = s[:pos] + '((' + top + ')/(' + bottom + '))' + s[end:]
    if '\\' in s: raise Unsupported('UNKNOWN_LATEX_COMMAND')
    return s.replace('{', '(').replace('}', ')').strip()


def expression(text):
    s = plain(text)
    ts = []; pos = 0
    while pos < len(s):
        if s[pos].isspace(): pos += 1; continue
        m = re.match(r'(?:\d+(?:\.\d+)?|mean|[xySfg]|[+*/^()|\-])', s[pos:])
        if not m: raise Unsupported('EXPRESSION_TOKEN')
        ts.append(m.group()); pos += len(m.group())
    if not ts or len(ts) > 192: raise Unsupported('TOKEN_LIMIT')
    out = []
    for t in ts:
        if out and (out[-1][0].isdigit() or out[-1] in ('x','y','S','mean',')')) and (
                t in ('x','y','S','mean','f','g','(') or (out[-1] == ')' and t[0].isdigit())):
            out.append('*')
        out.append(t)
    ts = out; pos = 0
    def parse(level=0, depth=0):
        nonlocal pos
        if depth > 24 or pos >= len(ts): raise Unsupported('EXPRESSION_DEPTH_OR_END')
        t = ts[pos]; pos += 1
        if t in ('+', '-'):
            a = parse(30, depth+1); node = ('neg', a) if t == '-' else a
        elif t == '(':
            node = parse(0, depth+1)
            if pos >= len(ts) or ts[pos] != ')': raise Unsupported('UNBALANCED_PARENTHESES')
            pos += 1
        elif t == '|':
            node = ('abs', parse(0, depth+1))
            if pos >= len(ts) or ts[pos] != '|': raise Unsupported('UNBALANCED_ABS')
            pos += 1
        elif t in ('f','g'):
            if pos >= len(ts) or ts[pos] != '(': raise Unsupported('FUNCTION_PARENTHESES')
            pos += 1; a = parse(0, depth+1)
            if pos >= len(ts) or ts[pos] != ')': raise Unsupported('FUNCTION_CLOSE')
            pos += 1; node = ('call', t, a)
        elif t in ('x','y','S','mean'): node = ('var', t)
        elif t[0].isdigit(): node = ('num', str(Fraction(t)))
        else: raise Unsupported('EXPECTED_OPERAND')
        while pos < len(ts) and ts[pos] in ('+','-','*','/','^'):
            op = ts[pos]; prec = {'+':10,'-':10,'*':20,'/':20,'^':40}[op]
            if prec < level: break
            pos += 1; node = (op, node, parse(prec if op == '^' else prec+1, depth+1))
        return node
    node = parse()
    if pos != len(ts): raise Unsupported('TRAILING_EXPRESSION')
    return node


def variables(ast):
    if ast[0] == 'var': return {ast[1]}
    return set().union(*(variables(x) for x in ast[1:] if isinstance(x, (tuple,list))))


def strip_question(s):
    s = plain(s)
    for prefix in ('解方程组:', '解不等式:', '解方程:', '解比例:', '展开并化简:', '化简:'):
        if s.startswith(prefix): return s[len(prefix):].strip()
    return s


def claim(text):
    s = plain(text)
    if s in ('=>','→','⇒','重算','重新计算') or s.startswith(('检查:',)):
        return {'kind':'cue'}
    operation = None
    m = re.match(r'^\(1\)([+-])\(2\)\s*:\s*(.*)$', s)
    if m: operation, s = m.groups()
    # Explicit task-level disjunction, never interpret a missing branch as complete.
    parts = re.split(r'\s+or\s+|或', s)
    if len(parts) > 1:
        if len(parts) > 4: raise Unsupported('BRANCH_LIMIT')
        return {'kind':'or', 'branches':[claim(p) for p in parts]}
    inequalities = re.split(r'(<=|>=|<|>)', s)
    if len(inequalities) == 3:
        rhs = inequalities[2].split('=')
        return {'kind':'inequality','op':inequalities[1],
                'asts':[expression(inequalities[0]), *[expression(p) for p in rhs]]}
    if len(inequalities) != 1: raise Unsupported('COMPARISON_CHAIN')
    continuation = s.startswith('=')
    if continuation: s = s[1:]
    parts = s.split('=')
    if len(parts) > 8 or any(not p.strip() for p in parts): raise Unsupported('EQUALITY_CHAIN')
    return {'kind':'chain' if len(parts)>1 else 'expression', 'asts':[expression(p) for p in parts],
            'continuation':continuation, 'operation':operation}


def signature(c):
    # The system-operation annotation has no LaTeX equivalent; validated separately.
    return {k:v for k,v in c.items() if k != 'operation'}


def question(text):
    raw = plain(text); s = strip_question(text)
    if raw.startswith('甲组'):
        m = re.fullmatch(r'甲组\s*(\d+)\s*人,平均\s*(\d+)\s*分;乙组\s*(\d+)\s*人,平均\s*(\d+)\s*分。求总平均分。', raw)
        if not m: raise Unsupported('WEIGHTED_QUESTION')
        return {'kind':'weighted','values':list(map(int,m.groups()))}
    if raw.startswith('n_1='):
        m = re.fullmatch(r'n_1=(\d+),\s*mean_1=(\d+);\s*n_2=(\d+),\s*mean_2=(\d+)',raw)
        if not m: raise Unsupported('WEIGHTED_LATEX')
        return {'kind':'weighted','values':list(map(int,m.groups()))}
    if s.startswith('f(x)='):
        m = re.fullmatch(r'f\(x\)=(.+),\s*g\(x\)=(.+),\s*(?:求\s*)?(.+)',s)
        if not m: raise Unsupported('FUNCTION_DEFINITIONS')
        f,g,target = m.groups()
        return {'kind':'function','defs':{'f':expression(f),'g':expression(g)},'target':expression(target)}
    if raw.startswith('解方程组:') or ';' in s:
        ps = re.split(r'[,;]',s)
        if len(ps)!=2: raise Unsupported('SYSTEM_ARITY')
        return {'kind':'system','equations':[claim(p) for p in ps]}
    c = claim(s)
    kind = ('inequality' if c['kind']=='inequality' else
            'equation' if c['kind'] in ('chain','or') else 'expression')
    return {'kind':kind,'claim':c}


def parse_public(payload):
    q = question(payload['question']['text'])
    if q != question(payload['question']['latex']): raise Unsupported('QUESTION_TEXT_LATEX_DISAGREE')
    nodes = []
    for i, s in enumerate(payload['steps'],1):
        if s['node_id']!=f's{i}' or s['position']!=i: raise Unsupported('STEP_IDENTITIES')
        if i==1:
            # A given may omit Solve/Expand in display, but its task must match.
            given = question(s['display'])
            if given != q or question(s['latex']) != q: raise Unsupported('GIVEN_MISMATCH')
            c={'kind':'given'}
        else:
            c=claim(s['display'])
            if c['kind']!='cue' and signature(c)!=signature(claim(s['latex'])):
                raise Unsupported('STEP_TEXT_LATEX_DISAGREE')
        nodes.append({'node_id':s['node_id'],'claim':c,'display':s['display']})
    if not nodes: raise Unsupported('EMPTY_TRACE')
    if payload['student_answer'] is None: raise Unsupported('MISSING_ANSWER')
    return {'question':q,'nodes':nodes,'answer':payload['student_answer']}

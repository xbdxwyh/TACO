"""Exact bounded algebra TOOL. This is not a neural state prediction.

Sparse rational polynomials retain every denominator constraint. Domains and
branches are checked rather than reducing all tasks to one matching root.
"""
from fractions import Fraction as F
from math import isqrt
from .math12_syntax_v2 import Unsupported, expression, claim, variables

ZERO=(0,0)


def poly(p):
    p={k:F(v) for k,v in p.items() if v}
    if len(p)>128 or any(sum(k)>10**13 or max(v.numerator.bit_length(),v.denominator.bit_length())>512 for k,v in p.items()):
        raise Unsupported('ALGEBRA_SIZE_LIMIT')
    return p


def add(a,b,sign=1):
    p=dict(a)
    for k,v in b.items(): p[k]=p.get(k,F(0))+sign*v
    return poly(p)


def mul(a,b):
    p={}
    for (x,y),v in a.items():
        for (u,w),z in b.items(): p[(x+u,y+w)]=p.get((x+u,y+w),F(0))+v*z
    return poly(p)


def number(n): return (poly({ZERO:F(n)}),{ZERO:F(1)},[])


def domain_zeros(a):
    excluded=set()
    for p in a[2]+[a[1]]:
        if len(p)==1:
            i,j=next(iter(p))
            if i:excluded.add(('x',F(0)))
            if j:excluded.add(('y',F(0)))
        else:
            r=roots(p)
            if r is None:raise Unsupported('EMPTY_DOMAIN')
            excluded.update(('x',x) for x in r)
    return excluded


def equal(a,b):
    return mul(a[0],b[1])==mul(b[0],a[1]) and domain_zeros(a)==domain_zeros(b)


def constant(a):
    if set(a[0])- {ZERO} or set(a[1])-{ZERO} or not a[1].get(ZERO):
        raise Unsupported('EXPECTED_CONSTANT')
    return a[0].get(ZERO,F(0))/a[1][ZERO]


def value(ast, env=None, defs=None, depth=0):
    env=env or {}; defs=defs or {}
    if depth>28: raise Unsupported('EVALUATION_DEPTH')
    k=ast[0]
    if k=='num': return number(ast[1])
    if k=='var':
        if ast[1] in env: return number(env[ast[1]])
        if ast[1] not in ('x','y'): raise Unsupported('UNBOUND_VARIABLE')
        return ({(1,0) if ast[1]=='x' else (0,1):F(1)},{ZERO:F(1)},[])
    if k=='call':
        if ast[1] not in defs: raise Unsupported('UNDEFINED_FUNCTION')
        arg=constant(value(ast[2],env,defs,depth+1))
        return value(defs[ast[1]],dict(env,x=arg),{},depth+1)
    a=value(ast[1],env,defs,depth+1)
    if k=='neg': return ({d:-v for d,v in a[0].items()},a[1],a[2])
    if k=='abs': return number(abs(constant(a)))
    b=value(ast[2],env,defs,depth+1)
    domains=a[2]+b[2]
    if k=='+': return add(mul(a[0],b[1]),mul(b[0],a[1])),mul(a[1],b[1]),domains
    if k=='-': return add(mul(a[0],b[1]),mul(b[0],a[1]),-1),mul(a[1],b[1]),domains
    if k=='*': return mul(a[0],b[0]),mul(a[1],b[1]),domains
    if k=='/':
        if not b[0]: raise Unsupported('DIVISION_BY_ZERO')
        return mul(a[0],b[1]),mul(a[1],b[0]),domains+[b[0]]
    if k=='^':
        n=constant(b)
        if n.denominator!=1 or abs(n)>10**13: raise Unsupported('POWER_OUTSIDE_DOMAIN')
        n=int(n); numerator,denominator=a[:2]
        if n<0:
            if not numerator:raise Unsupported('ZERO_NEGATIVE_POWER')
            domains=domains+[numerator];numerator,denominator=denominator,numerator;n=-n
        def power_poly(p):
            if not n:return {ZERO:F(1)}
            if not p:return {}
            if len(p)==1:
                (i,j),coeff=next(iter(p.items()))
                size=max(abs(coeff.numerator).bit_length(),coeff.denominator.bit_length())
                if abs(coeff)!=1 and n*max(1,size-1)>500:raise Unsupported('COEFFICIENT_POWER_SIZE')
                return poly({(i*n,j*n):coeff**n})
            if n>32:raise Unsupported('DENSE_POWER_SIZE')
            result={ZERO:F(1)}
            for _ in range(n):result=mul(result,p)
            return result
        top=power_poly(numerator);bottom=power_poly(denominator)
        return top,bottom,domains
    raise Unsupported('UNKNOWN_EXPRESSION')


def nonzero_at(p,x):
    if len(p)==1:
        (i,j),v=next(iter(p.items()))
        if j:raise Unsupported('MULTIVARIATE_DOMAIN')
        return bool(v) and (not i or x!=0)
    if any(i>64 or j for i,j in p):raise Unsupported('DENSE_DOMAIN_EVALUATION')
    return sum(v*x**i for (i,j),v in p.items())!=0


def roots(p):
    if any(y for x,y in p): raise Unsupported('MULTIVARIATE_ROOT')
    degree=max((x for x,y in p),default=0)
    if degree>2: raise Unsupported('ROOT_DEGREE')
    c=p.get(ZERO,F(0)); b=p.get((1,0),F(0)); a=p.get((2,0),F(0))
    if not a:
        if b: return {-c/b}
        if c: return set()
        return None
    disc=b*b-4*a*c
    if disc<0:return set()
    n,d=isqrt(disc.numerator),isqrt(disc.denominator)
    if n*n!=disc.numerator or d*d!=disc.denominator: raise Unsupported('IRRATIONAL_ROOT')
    sq=F(n,d); return {(-b+sq)/(2*a),(-b-sq)/(2*a)}


def equation_roots(c):
    if c['kind']=='or':
        result=set()
        for sub in c['branches']:
            r=equation_roots(sub)
            if r is None: raise Unsupported('UNBOUNDED_DISJUNCTION')
            result.update(r)
        return result
    if c['kind']!='chain': raise Unsupported('EXPECTED_EQUATION')
    asts=c['asts']
    if len(asts)==2 and asts[0][0]=='abs':
        radius=constant(value(asts[1]))
        if radius<0:return set()
        inner=asts[0][1]
        return equation_roots({'kind':'or','branches':[
            {'kind':'chain','asts':[inner,('num',str(radius))]},
            {'kind':'chain','asts':[inner,('num',str(-radius))]}]})
    vs=[value(a) for a in asts]; candidate=None; domains=[]
    for a in vs: domains += a[2]+[a[1]]
    for a,b in zip(vs,vs[1:]):
        r=roots(add(mul(a[0],b[1]),mul(b[0],a[1]),-1))
        if r is not None: candidate=r if candidate is None else candidate&r
    if candidate is None:
        if any(set(p)-{ZERO} for p in domains): raise Unsupported('INFINITE_DOMAIN_WITH_HOLES')
        return None
    return {r for r in candidate if all(nonzero_at(p,r) for p in domains)}


def inequality(c):
    if c['kind']!='inequality': raise Unsupported('EXPECTED_INEQUALITY')
    vals=[value(a) for a in c['asts']]
    if any(not equal(vals[1],v) for v in vals[2:]): return ('empty',)
    left,right=vals[:2]
    if any(set(v[1])-{ZERO} for v in (left,right)): raise Unsupported('RATIONAL_INEQUALITY')
    p=add({k:v/left[1][ZERO] for k,v in left[0].items()},
          {k:v/right[1][ZERO] for k,v in right[0].items()},-1)
    if set(p)-{ZERO,(1,0)}: raise Unsupported('NONLINEAR_INEQUALITY')
    a,b=p.get((1,0),F(0)),p.get(ZERO,F(0)); op=c['op']
    if not a: return ('all',) if {'<':b<0,'>':b>0,'<=':b<=0,'>=':b>=0}[op] else ('empty',)
    if a<0: op={'<':'>','>':'<','<=':'>=','>=':'<='}[op]
    return (op,-b/a)


def system(q):
    rows=[]
    for c in q['equations']:
        if c['kind']!='chain' or len(c['asts'])!=2: raise Unsupported('SYSTEM_EQUATIONS')
        a,b=[value(t) for t in c['asts']]
        if any(set(t[1])-{ZERO} for t in (a,b)): raise Unsupported('SYSTEM_DENOMINATOR')
        p=add({k:v/a[1][ZERO] for k,v in a[0].items()},{k:v/b[1][ZERO] for k,v in b[0].items()},-1)
        if set(p)-{ZERO,(1,0),(0,1)}: raise Unsupported('NONLINEAR_SYSTEM')
        rows.append((p.get((1,0),F(0)),p.get((0,1),F(0)),-p.get(ZERO,F(0))))
    (a,b,c),(d,e,f)=rows; det=a*e-b*d
    if not det: raise Unsupported('NONUNIQUE_SYSTEM')
    return {'x':(c*e-b*f)/det,'y':(a*f-c*d)/det}


def context(q):
    k=q['kind']; ctx={'kind':k,'public_question':q}
    if k=='equation':
        ctx['roots']=equation_roots(q['claim'])
        if not ctx['roots']: raise Unsupported('NO_FINITE_NONEMPTY_SOLUTION')
    elif k=='inequality': ctx['interval']=inequality(q['claim'])
    elif k=='expression': ctx['expected']=value(q['claim']['asts'][0])
    elif k=='system': ctx['env']=system(q)
    elif k=='function':
        ctx['defs']=q['defs']; ctx['target']=q['target']; ctx['expected']=value(q['target'],defs=q['defs'])
        if q['target'][0]!='call' or q['target'][2][0]!='call': raise Unsupported('EXPECTED_NESTED_TARGET')
        ctx['inner']=q['target'][2]
    elif k=='weighted':
        n,a,m,b=q['values']
        if min(n,m)<=0: raise Unsupported('NONPOSITIVE_GROUP_SIZE')
        ctx['env']={'S':F(n*a+m*b),'mean':F(n*a+m*b,n+m)}
    else: raise Unsupported('TASK_KIND')
    return ctx


def judge(ctx,c):
    """Global mathematical/task checks. Not an immediate-premise local proof."""
    kind=c['kind']; task=ctx['kind']; scope='goal'; aligned=True
    if kind in ('cue','given'):
        return {'semantic_valid':True,'task_aligned':True,'admissible':True,
                'scope':kind,'completes':False,'reason':kind.upper()}
    if task=='equation': valid=equation_roots(c)==ctx['roots']
    elif task=='inequality': valid=inequality(c)==ctx['interval']
    elif task=='expression':
        if kind not in ('chain','expression'): raise Unsupported('EXPRESSION_CLAIM')
        valid=all(equal(value(a),ctx['expected']) for a in c['asts'])
    elif task=='system':
        if kind not in ('chain','expression') or len(c['asts'])<2: raise Unsupported('SYSTEM_CLAIM')
        names=variables(c['asts'][0]); scope=next(iter(names)) if len(names)==1 else 'system'
        if names-{'x','y'}: raise Unsupported('SYSTEM_VARIABLE')
        vs=[value(a,ctx['env']) for a in c['asts']]
        valid=all(equal(vs[0],v) for v in vs[1:]); aligned=bool(names)
        if c.get('operation'):
            # Verify the claimed LHS and RHS combination, not just at the solution.
            e1,e2=ctx['public_question']['equations']; op=c['operation']
            wanted=[(op,e1['asts'][i],e2['asts'][i]) for i in range(2)]
            valid=valid and all(equal(value(c['asts'][i]),value(wanted[i])) for i in range(2))
    elif task=='function':
        if kind not in ('chain','expression'): raise Unsupported('FUNCTION_CLAIM')
        asts=c['asts']; vs=[value(a,defs=ctx['defs']) for a in asts]
        if asts[0][0]=='call':
            valid=all(equal(vs[0],v) for v in vs[1:])
            aligned=asts[0] in (ctx['inner'],ctx['target'])
            scope='inner' if asts[0]==ctx['inner'] else 'goal'
        else: valid=all(equal(v,ctx['expected']) for v in vs)
    elif task=='weighted':
        if kind not in ('chain','expression'): raise Unsupported('WEIGHTED_CLAIM')
        asts=c['asts']; vs=[value(a,ctx['env']) for a in asts]
        scope='total' if asts[0]==('var','S') else 'goal'
        target=number(ctx['env']['S' if scope=='total' else 'mean'])
        valid=all(equal(v,target) for v in vs)
    else: raise Unsupported('JUDGMENT_KIND')
    return {'semantic_valid':bool(valid),'task_aligned':bool(aligned),'admissible':bool(valid and aligned),
            'scope':scope,'completes':scope=='goal' or (task=='system' and scope in ('x','y')),
            'reason':'PASS' if valid and aligned else ('OFF_TARGET_WORK' if valid else 'MATHEMATICAL_MISMATCH')}


def answer_claim(ctx,text):
    kind=ctx['kind']
    if kind=='equation' and '=' not in text: text='x='+text
    if kind=='system':
        parts=text.split(',')
        if len(parts)!=2: raise Unsupported('SYSTEM_ANSWER_ARITY')
        cs=[claim(s) for s in parts]
        if {tuple(c['asts'][0]) for c in cs}!={('var','x'),('var','y')}:
            raise Unsupported('INCOMPLETE_SYSTEM_ANSWER')
        return cs
    return [claim(text)]


def answer_ok(ctx,text): return all(judge(ctx,c)['admissible'] for c in answer_claim(ctx,text))

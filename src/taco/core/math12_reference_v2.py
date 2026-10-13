"""Scoring-only SymPy arithmetic cross-check; never imported by prediction.

Shares the bounded syntax and declared task contract, NOT the Fraction algebra.
This cross-check plus original annotations does not prove the shared parser.
"""
import sympy as sp
from .math12_syntax_v2 import Unsupported, claim, variables

X,Y=sp.symbols('x y',real=True)


def expression_ref(a,env=None,defs=None):
    env=env or {}; defs=defs or {}; k=a[0]
    if k=='num': return sp.Rational(a[1])
    if k=='var':
        if a[1] in env:return env[a[1]]
        if a[1] in ('x','y'):return {'x':X,'y':Y}[a[1]]
        raise Unsupported('REF_UNBOUND')
    if k=='call':
        if a[1] not in defs: raise Unsupported('REF_FUNCTION')
        return expression_ref(defs[a[1]],dict(env,x=expression_ref(a[2],env,defs)),{})
    left=expression_ref(a[1],env,defs)
    if k=='neg':return -left
    if k=='abs':return sp.Abs(left)
    right=expression_ref(a[2],env,defs)
    if k=='/':
        if right==0:raise Unsupported('REF_ZERO_DIVISOR')
        return left/right
    if k=='^':
        if not right.is_Integer or abs(right)>10**13:raise Unsupported('REF_POWER')
        n=abs(int(right))
        if n>32:
            num,den=sp.fraction(left)
            for expr in (num,den):
                terms=sp.Poly(expr,X,Y).terms()
                if len(terms)>1:raise Unsupported('REF_DENSE_POWER')
                coeff=terms[0][1]
                if abs(coeff)!=1 and n>500:raise Unsupported('REF_POWER_COEFFICIENT')
        if left==0 and right<0:raise Unsupported('REF_ZERO_POWER')
        return left**right
    return {'+':lambda:left+right,'-':lambda:left-right,'*':lambda:left*right}[k]()


def same_ref(a,b):
    # expand stays sparse for allowed monomials; do not call polynomial GCD on
    # x**(12**12). Large coefficients/dense powers are rejected before here.
    return sp.expand(a-b)==0


def domains_ref(a):
    result=[]
    if a[0]=='/': result.append(expression_ref(a[2]))
    if a[0]=='^' and expression_ref(a[2])<0:result.append(expression_ref(a[1]))
    for t in a[1:]:
        if isinstance(t,(tuple,list)):result+=domains_ref(t)
    return result


def excluded_ref(a):
    result=sp.S.EmptySet
    for den in domains_ref(a):
        # Zero sets of a monomial are known without enumerating a huge degree.
        p=sp.Poly(den,X)
        if len(p.terms())==1:
            degree=p.degree();z=sp.FiniteSet(0) if degree>0 else sp.S.EmptySet
        else:z=sp.solveset(den,X,domain=sp.S.Reals)
        result=sp.Union(result,z)
    return result


def solutions_ref(c):
    if c['kind']=='or':return sp.Union(*(solutions_ref(x) for x in c['branches']))
    if c['kind']!='chain':raise Unsupported('REF_NOT_EQUATION')
    vals=[expression_ref(a) for a in c['asts']]
    result=sp.S.Reals
    for a,b in zip(vals,vals[1:]): result=sp.Intersection(result,sp.solveset(a-b,X,domain=sp.S.Reals))
    for a in c['asts']:
        for den in domains_ref(a):result=sp.Complement(result,sp.solveset(den,X,domain=sp.S.Reals))
    if result.has(sp.ConditionSet):raise Unsupported('REF_UNRESOLVED_SET')
    return result


def inequality_ref(c):
    if c['kind']!='inequality':raise Unsupported('REF_NOT_INEQUALITY')
    vals=[expression_ref(a) for a in c['asts']]
    if any(not same_ref(vals[1],v) for v in vals[2:]):return sp.S.EmptySet
    op={'<':sp.Lt,'>':sp.Gt,'<=':sp.Le,'>=':sp.Ge}[c['op']]
    return sp.solve_univariate_inequality(op(*vals[:2]),X,relational=False)


def context_ref(q):
    k=q['kind']; r={'kind':k,'q':q}
    if k=='equation':r['answer']=solutions_ref(q['claim'])
    elif k=='inequality':r['answer']=inequality_ref(q['claim'])
    elif k=='expression':r['answer']=expression_ref(q['claim']['asts'][0])
    elif k=='function':r['answer']=expression_ref(q['target'],defs=q['defs'])
    elif k=='weighted':
        n,a,m,b=q['values']; r['env']={'S':sp.Integer(n*a+m*b),'mean':sp.Rational(n*a+m*b,n+m)}
    elif k=='system':
        eqs=[expression_ref(c['asts'][0])-expression_ref(c['asts'][1]) for c in q['equations']]
        answer=list(sp.linsolve(eqs,(X,Y)))
        if len(answer)!=1 or any(z.free_symbols for z in answer[0]):raise Unsupported('REF_SYSTEM')
        r['env']=dict(zip(('x','y'),answer[0]))
    else:raise Unsupported('REF_TASK')
    return r


def judge_ref(r,c):
    if c['kind'] in ('given','cue'):return True
    k=r['kind']; q=r['q']
    if k=='equation':return solutions_ref(c)==r['answer']
    if k=='inequality':return inequality_ref(c)==r['answer']
    asts=c.get('asts')
    if not asts:raise Unsupported('REF_CLAIM')
    if k=='expression':
        domain=excluded_ref(q['claim']['asts'][0])
        return all(same_ref(expression_ref(a),r['answer']) and excluded_ref(a)==domain for a in asts)
    if k=='function':
        vals=[expression_ref(a,defs=q['defs']) for a in asts]
        if asts[0][0]=='call':
            aligned=asts[0] in (q['target'],q['target'][2])
            return aligned and all(same_ref(vals[0],v) for v in vals[1:])
        return all(same_ref(v,r['answer']) for v in vals)
    if k=='weighted':
        target=r['env']['S' if asts[0]==('var','S') else 'mean']
        return all(same_ref(expression_ref(a,r['env']),target) for a in asts)
    if k=='system':
        if len(asts)<2:raise Unsupported('REF_SYSTEM_CLAIM')
        names=variables(asts[0]); vals=[expression_ref(a,r['env']) for a in asts]
        valid=bool(names) and all(same_ref(vals[0],v) for v in vals[1:])
        if c.get('operation'):
            e1,e2=q['equations']; sign=1 if c['operation']=='+' else -1
            valid=valid and all(same_ref(expression_ref(asts[i]),
                expression_ref(e1['asts'][i])+sign*expression_ref(e2['asts'][i])) for i in range(2))
        return valid
    raise Unsupported('REF_TASK')


def answer_ref(r,text):
    if r['kind']=='equation' and '=' not in text:text='x='+text
    if r['kind']=='system':
        parts=text.split(',')
        if len(parts)!=2:return False
        cs=[claim(p) for p in parts]
        if {tuple(c['asts'][0]) for c in cs}!={('var','x'),('var','y')}:return False
        return all(judge_ref(r,c) for c in cs)
    return judge_ref(r,claim(text))

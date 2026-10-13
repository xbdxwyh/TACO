"""V2 restricted math: Python comparison semantics and explicit math capabilities.

V1 is frozen. Generated code still never reaches exec/eval/compile. The caller
retains the isolated process, five-second limit and Windows memory/job bounds.
"""
import ast
import json
import operator
import re
import string
import sys
from pathlib import Path
import sympy as sp
# -I intentionally removes the working directory; import only our fixed sibling.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from genprm_math_sandbox_v1 import Interpreter as Legacy, Function, Module, Unsupported, contains_string


def math_call(fn):
    def call(*args, **kw):
        if contains_string(args) or any(contains_string(v) for v in kw.values()):
            raise Unsupported('String-to-symbolic coercion is unavailable; use Symbol/symbols or sympify')
        return fn(*args, **kw)
    return call


class Interpreter(Legacy):
    def __init__(self):
        super().__init__()
        sym=self.modules['sympy'].values
        for name in ('solve_univariate_inequality reduce_inequalities Pow expand_mul latex '
                     'Interval FiniteSet Union Intersection').split():
            sym[name]=Function(name,math_call(getattr(sp,name)))
        sym['S'].values.update(Reals=sp.S.Reals,Integers=sp.S.Integers,Rationals=sp.S.Rationals,
                               Complexes=sp.S.Complexes,EmptySet=sp.S.EmptySet)
        def symbolic_function(name):
            self.symbol(name)  # validate identifier without invoking a string parser
            return Function(name,math_call(sp.Function(name)))
        sym['Function']=Function('Function',symbolic_function)
        # Only this math namespace is exposed, not the actual imported module.
        self.modules['sympy.solvers']=Module({'solvers':Module({
            n:sym[n] for n in ('solve','solveset','solve_univariate_inequality')})})
        self.modules['re']=Module({n:Function('re.'+n,getattr(re,n))
                                  for n in ('compile','match','search','fullmatch','findall','split','sub','escape')})
        def mapped(fn,*iters):
            if not isinstance(fn,Function):raise Unsupported('map requires an allowlisted function')
            result=[]
            for values in zip(*iters):
                self.tick()
                if len(result)>=10000:raise Unsupported('Iteration limit reached')
                result.append(fn.call(*values))
            return result
        self.env['map']=Function('map',mapped)

    def expr(self,node,env):
        if isinstance(node,ast.Compare):
            self.tick();a=self.expr(node.left,env)
            for i,(op,n) in enumerate(zip(node.ops,node.comparators)):
                b=self.expr(n,env)
                fn={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,
                    ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b,
                    ast.NotIn:lambda a,b:a not in b,ast.Is:operator.is_,ast.IsNot:operator.is_not}.get(type(op))
                if fn is None:raise Unsupported('Comparison')
                value=fn(a,b)
                # Python returns the final comparison object WITHOUT bool().
                # Intermediate chained comparisons do require truth conversion.
                if i==len(node.ops)-1 or not value:return value
                a=b
        if isinstance(node,ast.BoolOp):
            self.tick()
            for i,n in enumerate(node.values):
                value=self.expr(n,env)
                if i==len(node.values)-1:return value
                if isinstance(node.op,ast.And) and not value:return value
                if isinstance(node.op,ast.Or) and value:return value
        if isinstance(node,ast.Lambda):
            self.tick();args=node.args
            if args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs:raise Unsupported('Lambda signature')
            names=[a.arg for a in args.args];defaults=[self.expr(x,env) for x in args.defaults]
            def call(*values,**kw):
                if self.depth>=32:raise Unsupported('Recursion limit')
                if len(values)>len(names) or set(kw)-set(names) or set(names[:len(values)])&set(kw):
                    raise TypeError('Lambda arguments')
                bound=dict(zip(names[len(names)-len(defaults):],defaults));bound.update(zip(names,values));bound.update(kw)
                if set(bound)!=set(names):raise TypeError('Missing arguments')
                self.depth+=1
                try:return self.expr(node.body,dict(env,**bound))
                finally:self.depth-=1
            return Function('lambda',call)
        return super().expr(node,env)

    def attribute(self,obj,name):
        if name.startswith('_'):raise Unsupported('Private attribute unavailable')
        if isinstance(obj,str) and name=='format':
            def format_string(*args,**kw):
                for _,field,spec,_ in string.Formatter().parse(obj):
                    if field is not None and (not re.fullmatch(r'[A-Za-z0-9_]*',field) or field.startswith('_')):
                        raise Unsupported('Compound/private format field')
                    if '{' in spec or '}' in spec or len(spec)>30 or any(int(x)>1000 for x in re.findall(r'\d+',spec)):
                        raise Unsupported('Format size limit')
                return obj.format(*args,**kw)
            return Function(name,format_string)
        allowed=()
        if isinstance(obj,str):allowed=('split','rsplit','join','replace','strip','lstrip','rstrip','startswith','endswith')
        elif isinstance(obj,(set,frozenset)):allowed=('union','intersection','difference','issubset','issuperset')
        elif isinstance(obj,float):allowed=('as_integer_ratio',)
        elif isinstance(obj,sp.Basic):
            allowed=('as_coefficients_dict','as_set','union','intersection')
        elif isinstance(obj,re.Pattern):allowed=('match','search','fullmatch','findall','split','sub')
        elif isinstance(obj,re.Match):allowed=('group','groups','start','end','span')
        if name in allowed and hasattr(obj,name):
            fn=getattr(obj,name)
            return Function(name,math_call(fn) if isinstance(obj,sp.Basic) else fn)
        return super().attribute(obj,name)


def worker():
    engine=Interpreter();print(json.dumps(dict(ready=True)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request=={'reset':True}:
            engine=Interpreter();result=dict(status='ok',output='RESET')
        else:result=engine.execute(request['code'])
        print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':worker()

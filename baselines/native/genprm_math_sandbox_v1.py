"""Capability-limited mathematical Python interpreter for GenPRM on Windows.

Generated code is parsed, NEVER passed to Python exec/eval/compile. Only explicit
AST constructs and mathematical operations below are interpreted. No host files,
network, dynamic imports, arbitrary attributes, or foreign Python callables are
available. Unsupported programs return an explicit error to the verifier.
The parent additionally enforces a 5-second wall limit and a Windows memory job.
This is a disclosed restricted-math execution adaptation, not general Python.
"""
import ast
import math
import operator
import re
import sys
import json
import sympy as sp
from fractions import Fraction

class Unsupported(Exception): pass
class Returned(Exception):
    def __init__(self,value): self.value=value
class Broken(Exception): pass
class Continued(Exception): pass

class Function:
    def __init__(self,name,call): self.name,self.call=name,call
class Module:
    def __init__(self,values): self.values=values

def contains_string(x):
    if isinstance(x,str): return True
    if isinstance(x,(list,tuple,set)): return any(contains_string(v) for v in x)
    if isinstance(x,dict): return any(contains_string(v) for pair in x.items() for v in pair)
    return False

class Interpreter:
    def __init__(self):
        self.output=[];self.ticks=0;self.depth=0
        def safe(fn):
            def call(*args,**kwargs):
                # SymPy string coercion can invoke its parser: disallow it.
                if any(contains_string(x) for x in args) or any(contains_string(x) for x in kwargs.values()):
                    raise Unsupported('String-to-symbolic coercion is unavailable; use Symbol/symbols or sympify')
                return fn(*args,**kwargs)
            return call
        symbolic={}
        for name in ('Eq Ne Lt Le Gt Ge solve solveset linsolve nonlinsolve simplify expand factor cancel together '
                     'apart collect diff integrate limit summation product sqrt root Abs sin cos tan asin acos atan '
                     'atan2 exp log floor ceiling Min Max N gcd lcm binomial factorial Rational Integer Float '
                     'Matrix Poly degree roots real_root re im conjugate trigsimp powsimp expand_trig nsimplify '
                     'isprime primefactors divisors And Or Not Implies Equivalent').split():
            symbolic[name]=Function(name,safe(getattr(sp,name)))
        symbolic.update(pi=sp.pi,E=sp.E,I=sp.I,oo=sp.oo,true=sp.true,false=sp.false)
        symbolic['symbols']=Function('symbols',self.symbols)
        symbolic['Symbol']=Function('Symbol',self.symbol)
        symbolic['sympify']=Function('sympify',self.sympify)
        # Only scalar attributes of S are offered, never S(string).
        symbolic['S']=Module(dict(Infinity=sp.oo,Zero=sp.S.Zero,One=sp.S.One))
        math_values={n:Function('math.'+n,getattr(math,n)) for n in
            ('sqrt sin cos tan asin acos atan atan2 exp log log10 log2 pow floor ceil fabs factorial gcd lcm '
             'isclose isfinite isinf isnan radians degrees hypot comb perm trunc prod fsum').split()}
        math_values.update(pi=math.pi,e=math.e,tau=math.tau,inf=math.inf)
        self.modules={'sympy':Module(symbolic),'math':Module(math_values),
                      'fractions':Module({'Fraction':Function('Fraction',self.fraction)})}
        builtins={n:Function(n,fn) for n,fn in dict(abs=abs,round=round,len=len,sum=sum,min=min,max=max,
            sorted=sorted,list=list,tuple=tuple,set=set,dict=dict,enumerate=enumerate,zip=zip,
            all=all,any=any,reversed=reversed,int=int,float=float,str=str,bool=bool).items()}
        builtins.update(print=Function('print',self.print),range=Function('range',self.range),
                        pow=Function('pow',lambda a,b:self.binop(ast.Pow(),a,b)))
        self.env=builtins

    def tick(self):
        self.ticks+=1
        if self.ticks>50000: raise Unsupported('Instruction limit reached')

    def print(self,*args,sep=' ',end='\n'):
        text=sep.join(str(v) for v in args)+end
        if sum(map(len,self.output))+len(text)>16384: raise Unsupported('Output limit reached')
        self.output.append(text)

    def range(self,*args):
        r=range(*args)
        if len(r)>10000: raise Unsupported('Iteration limit reached')
        return r

    def symbol(self,name,**kw):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,40}',name):
            raise Unsupported('Invalid symbol name')
        if any(k not in ('real','positive','negative','nonzero','integer','nonnegative','finite') or
               not isinstance(v,bool) for k,v in kw.items()): raise Unsupported('Unsupported symbol assumption')
        return sp.Symbol(name,**kw)

    def symbols(self,names,**kw):
        if not isinstance(names,str) or len(names)>1024 or not re.fullmatch(r'[A-Za-z0-9_ ,:]+',names):
            raise Unsupported('Invalid symbols')
        # Validate assumptions with the same boundary as Symbol.
        self.symbol('validation',**kw)
        if any(int(n)>100 for n in re.findall(r':(\d+)',names)): raise Unsupported('Too many symbols')
        return sp.symbols(names,**kw)

    def fraction(self,*args):
        if any(isinstance(v,str) and not re.fullmatch(r'[+\-0-9./ eE]+',v) for v in args):
            raise Unsupported('Invalid fraction literal')
        return Fraction(*args)

    def sympify(self,value):
        if not isinstance(value,str):
            if contains_string(value): raise Unsupported('Nested symbolic strings unavailable')
            return sp.sympify(value)
        if len(value)>4096: raise Unsupported('Expression too long')
        tree=ast.parse(value.replace('^','**'),mode='eval')
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        env=dict(self.modules['sympy'].values)
        for name in names:
            if name not in env: env[name]=self.symbol(name)
        return self.expr(tree.body,env)

    def assign(self,target,value,env):
        if isinstance(target,ast.Name):
            if target.id.startswith('_'): raise Unsupported('Private name unavailable')
            env[target.id]=value
        elif isinstance(target,(ast.Tuple,ast.List)):
            values=list(value)
            if len(values)!=len(target.elts): raise ValueError('Unpack length mismatch')
            for t,v in zip(target.elts,values): self.assign(t,v,env)
        elif isinstance(target,ast.Subscript):
            obj=self.expr(target.value,env)
            if not isinstance(obj,(list,dict)): raise Unsupported('Assignment only to list/dict')
            obj[self.expr(target.slice,env)]=value
        else: raise Unsupported('Assignment '+type(target).__name__)

    def binop(self,op,a,b):
        if isinstance(op,ast.Pow) and (not isinstance(b,sp.Basic) and abs(b)>10000):
            raise Unsupported('Exponent limit')
        if isinstance(op,ast.Mult):
            if isinstance(a,(str,list,tuple)) and isinstance(b,int) and len(a)*max(0,b)>100000: raise Unsupported('Size limit')
            if isinstance(b,(str,list,tuple)) and isinstance(a,int) and len(b)*max(0,a)>100000: raise Unsupported('Size limit')
        f={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
           ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,
           ast.BitAnd:operator.and_,ast.BitOr:operator.or_}.get(type(op))
        if f is None: raise Unsupported('Operator '+type(op).__name__)
        v=f(a,b)
        if isinstance(v,(str,list,tuple,dict,set)) and len(v)>100000: raise Unsupported('Size limit')
        return v

    def attribute(self,obj,name):
        if name.startswith('_'): raise Unsupported('Private attribute unavailable')
        if isinstance(obj,Module):
            if name not in obj.values: raise Unsupported('Module member '+name)
            return obj.values[name]
        methods=()
        if isinstance(obj,sp.Basic):
            if name in ('lhs','rhs','is_real','is_integer','is_number','is_zero','free_symbols','args'): return getattr(obj,name)
            methods=('subs','evalf','simplify','expand','factor','diff','integrate','equals','as_numer_denom','coeff','doit')
        elif isinstance(obj,sp.MatrixBase): methods=('det','inv','transpose','rref','rank','eigenvals','tolist','subs','evalf')
        elif isinstance(obj,list): methods=('append','extend','count','index','pop','sort')
        elif isinstance(obj,dict): methods=('items','keys','values','get','update')
        elif isinstance(obj,(tuple,str)): methods=('count','index')
        if name not in methods: raise Unsupported('Attribute '+name)
        fn=getattr(obj,name)
        def call(*args,**kw):
            if isinstance(obj,(sp.Basic,sp.MatrixBase)) and (contains_string(args) or contains_string(kw)):
                # Keyword names are not coercion inputs; inspect values only.
                if contains_string(args) or any(contains_string(v) for v in kw.values()): raise Unsupported('Symbolic string argument')
            return fn(*args,**kw)
        return Function(name,call)

    def expr(self,node,env):
        self.tick()
        if isinstance(node,ast.Constant):
            if isinstance(node.value,(str,int,float,complex,bool,type(None))) and len(str(node.value))<=16384: return node.value
            raise Unsupported('Literal')
        if isinstance(node,ast.Name):
            if node.id.startswith('_'): raise Unsupported('Private name unavailable')
            if node.id not in env: raise NameError(node.id)
            return env[node.id]
        if isinstance(node,(ast.List,ast.Tuple,ast.Set)):
            values=[self.expr(x,env) for x in node.elts]
            return {ast.List:list,ast.Tuple:tuple,ast.Set:set}[type(node)](values)
        if isinstance(node,ast.Dict): return {self.expr(k,env):self.expr(v,env) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.BinOp): return self.binop(node.op,self.expr(node.left,env),self.expr(node.right,env))
        if isinstance(node,ast.UnaryOp):
            op={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_,ast.Invert:operator.invert}.get(type(node.op))
            if op is None: raise Unsupported('Unary operation')
            return op(self.expr(node.operand,env))
        if isinstance(node,ast.BoolOp):
            for n in node.values:
                v=self.expr(n,env)
                if isinstance(node.op,ast.And) and not v or isinstance(node.op,ast.Or) and v: return v
            return v
        if isinstance(node,ast.Compare):
            a=self.expr(node.left,env)
            for op,n in zip(node.ops,node.comparators):
                b=self.expr(n,env)
                f={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,
                   ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b,ast.NotIn:lambda a,b:a not in b,
                   ast.Is:operator.is_,ast.IsNot:operator.is_not}.get(type(op))
                if f is None: raise Unsupported('Comparison')
                if not f(a,b): return False
                a=b
            return True
        if isinstance(node,ast.IfExp): return self.expr(node.body if self.expr(node.test,env) else node.orelse,env)
        if isinstance(node,ast.Attribute): return self.attribute(self.expr(node.value,env),node.attr)
        if isinstance(node,ast.Subscript): return self.expr(node.value,env)[self.expr(node.slice,env)]
        if isinstance(node,ast.Slice): return slice(*(self.expr(x,env) if x else None for x in (node.lower,node.upper,node.step)))
        if isinstance(node,ast.Call):
            fn=self.expr(node.func,env)
            if not isinstance(fn,Function): raise Unsupported('Only allowlisted functions are callable')
            if any(isinstance(x,ast.Starred) for x in node.args) or any(k.arg is None for k in node.keywords): raise Unsupported('Argument expansion unavailable')
            return fn.call(*[self.expr(x,env) for x in node.args],**{k.arg:self.expr(k.value,env) for k in node.keywords})
        if isinstance(node,(ast.ListComp,ast.SetComp,ast.GeneratorExp,ast.DictComp)):
            result=[];local=dict(env)
            def walk(index):
                self.tick()
                if len(result)>10000: raise Unsupported('Comprehension limit')
                if index==len(node.generators):
                    result.append((self.expr(node.key,local),self.expr(node.value,local)) if isinstance(node,ast.DictComp) else self.expr(node.elt,local));return
                g=node.generators[index]
                if g.is_async: raise Unsupported('Async comprehension')
                for v in self.expr(g.iter,local):
                    self.assign(g.target,v,local)
                    if all(self.expr(n,local) for n in g.ifs): walk(index+1)
            walk(0)
            return dict(result) if isinstance(node,ast.DictComp) else set(result) if isinstance(node,ast.SetComp) else result
        if isinstance(node,ast.JoinedStr):
            parts=[]
            for x in node.values:
                if not isinstance(x,ast.FormattedValue): parts.append(self.expr(x,env));continue
                value=self.expr(x.value,env)
                if x.conversion==114: value=repr(value)
                elif x.conversion==115: value=str(value)
                elif x.conversion==97: value=ascii(value)
                spec=self.expr(x.format_spec,env) if x.format_spec else ''
                if len(spec)>30 or any(int(z)>1000 for z in re.findall(r'\d+',spec)): raise Unsupported('Format size limit')
                parts.append(format(value,spec))
            return ''.join(parts)
        raise Unsupported('Expression '+type(node).__name__)

    def block(self,nodes,env):
        for n in nodes: self.statement(n,env)

    def statement(self,n,env):
        self.tick()
        if isinstance(n,ast.Expr): self.expr(n.value,env)
        elif isinstance(n,ast.Assign):
            v=self.expr(n.value,env)
            for target in n.targets: self.assign(target,v,env)
        elif isinstance(n,ast.AugAssign): self.assign(n.target,self.binop(n.op,self.expr(n.target,env),self.expr(n.value,env)),env)
        elif isinstance(n,ast.Import):
            for a in n.names:
                if a.name not in self.modules: raise Unsupported('Import '+a.name)
                env[a.asname or a.name]=self.modules[a.name]
        elif isinstance(n,ast.ImportFrom):
            if n.level or n.module not in self.modules: raise Unsupported('Import '+str(n.module))
            module=self.modules[n.module]
            for a in n.names:
                if a.name=='*': env.update(module.values)
                else: env[a.asname or a.name]=self.attribute(module,a.name)
        elif isinstance(n,ast.If): self.block(n.body if self.expr(n.test,env) else n.orelse,env)
        elif isinstance(n,(ast.For,ast.While)):
            iterator=iter(self.expr(n.iter,env)) if isinstance(n,ast.For) else None
            broke=False
            while True:
                self.tick()
                if iterator is not None:
                    try: v=next(iterator)
                    except StopIteration: break
                    self.assign(n.target,v,env)
                elif not self.expr(n.test,env): break
                try: self.block(n.body,env)
                except Broken: broke=True;break
                except Continued: continue
            if not broke: self.block(n.orelse,env)
        elif isinstance(n,ast.FunctionDef):
            if n.decorator_list or n.args.vararg or n.args.kwarg or n.args.kwonlyargs or n.args.posonlyargs: raise Unsupported('Function signature')
            names=[a.arg for a in n.args.args];defaults=[self.expr(x,env) for x in n.args.defaults]
            def call(*args,**kw):
                if self.depth>=32: raise Unsupported('Recursion limit')
                if len(args)>len(names) or set(kw)-set(names): raise TypeError('Function arguments')
                local=dict(env);values=dict(zip(names[len(names)-len(defaults):],defaults));values.update(zip(names,args));values.update(kw)
                if set(values)!=set(names): raise TypeError('Missing arguments')
                local.update(values);self.depth+=1
                try: self.block(n.body,local)
                except Returned as r: return r.value
                finally: self.depth-=1
            self.assign(ast.Name(id=n.name),Function(n.name,call),env)
        elif isinstance(n,ast.Return): raise Returned(self.expr(n.value,env) if n.value else None)
        elif isinstance(n,ast.Assert):
            if not self.expr(n.test,env): raise AssertionError(self.expr(n.msg,env) if n.msg else '')
        elif isinstance(n,ast.Pass): pass
        elif isinstance(n,ast.Break): raise Broken()
        elif isinstance(n,ast.Continue): raise Continued()
        else: raise Unsupported('Statement '+type(n).__name__)

    def execute(self,code):
        self.output=[];self.ticks=0;self.depth=0
        try:
            if len(code)>20000: raise Unsupported('Code size limit')
            self.block(ast.parse(code).body,self.env)
            return dict(status='ok',output=''.join(self.output).strip())
        except Exception as e:
            return dict(status='unsupported' if isinstance(e,Unsupported) else 'error',
                        output=f'Code execute Error: {type(e).__name__}: {str(e)[:1000]}')

def worker():
    engine=Interpreter()
    print(json.dumps(dict(ready=True)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        print(json.dumps(engine.execute(request['code']),ensure_ascii=True),flush=True)

if __name__=='__main__': worker()

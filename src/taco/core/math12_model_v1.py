"""Shared small Transformer: learned local validity and contextual repair ranking.

No solved values, labels, residuals or dataset identities enter encode().
The state controller is explicit and separate from this learned semantic model.
"""
import hashlib
import math
from functools import lru_cache
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .math12_syntax_v2 import Unsupported, claim

MAX_TOKENS = 384
WORDS = ['PAD','CLS','OPEN','CLOSE','NUMBER','STRING','question','kind','claim','asts',
         'defs','target','values','equations','continuation','operation','op','branches',
         'equation','inequality','expression','system','function','weighted','chain',
         'or','given','cue','num','var','neg','abs','call','x','y','S','mean','f','g',
         '+','-','*','/','^','<','>','<=','>=','True','False','None']
VOCAB = {s:i for i,s in enumerate(WORDS)}


def candidate_claim(q, target, text):
    if target == 'answer' and q['kind'] == 'system':
        return dict(kind='system', equations=[claim(x) for x in text.split(',')])
    if target == 'answer' and q['kind'] == 'equation' and '=' not in text: text = 'x='+text
    return claim(text)


def encode(q, target, text, original=None, previous=None):
    tokens=[VOCAB['CLS']]; roles=[0]; nums=[[0.,0.,0.]]
    def push(word, role, number=None):
        if word not in VOCAB: raise Unsupported('ENCODING_SYMBOL:'+str(word))
        tokens.append(VOCAB[word]); roles.append(role)
        if number is None: nums.append([0.,0.,0.])
        else:
            v=Fraction(number); value=float(v)
            nums.append([max(-1e4,min(1e4,value))/256., math.copysign(math.log1p(abs(value)),value)/8., math.log1p(v.denominator)/8.])
    def visit(obj, role):
        if isinstance(obj, dict):
            push('OPEN',role)
            for k in sorted(obj): push(k,role); visit(obj[k],role)
            push('CLOSE',role)
        elif isinstance(obj,(list,tuple)):
            if len(obj)==2 and obj[0]=='num': push('NUMBER',role,obj[1]); return
            push('OPEN',role)
            for x in obj: visit(x,role)
            push('CLOSE',role)
        elif isinstance(obj,bool) or obj is None: push(str(obj),role)
        elif isinstance(obj,(int,float)): push('NUMBER',role,str(obj))
        else: push(obj,role)
    visit(q,0)
    if previous is not None: visit(previous,3)
    if original is not None: visit(candidate_claim(q,target,original),1)
    visit(candidate_claim(q,target,text),2)
    if len(tokens)>MAX_TOKENS: raise Unsupported('ENCODING_LENGTH')
    return dict(tokens=tokens,roles=roles,numbers=nums)


def record(prepared, target, text, rank=False):
    if target=='answer': original=prepared['answer']; before=prepared['nodes']
    else:
        index=int(target[1:])-1; original=prepared['nodes'][index]['display']; before=prepared['nodes'][:index]
    previous=next((n['claim'] for n in reversed(before) if n['claim']['kind'] not in ('given','cue')), None)
    return encode(prepared['question'],target,text,original if rank else None,previous if rank else None)


def batch(records, device):
    n=max(len(r['tokens']) for r in records); b=len(records)
    token=np.zeros((b,n),np.int64); role=token.copy(); number=np.zeros((b,n,3),np.float32)
    for i,r in enumerate(records):
        m=len(r['tokens']); token[i,:m]=r['tokens']; role[i,:m]=r['roles']; number[i,:m]=r['numbers']
    return [torch.as_tensor(x,device=device) for x in (token,role,number)]


class Block(nn.Module):
    def __init__(self,w=192,heads=6):
        super().__init__(); self.heads=heads
        self.norm1=nn.LayerNorm(w); self.norm2=nn.LayerNorm(w)
        self.qkv=nn.Linear(w,w*3); self.out=nn.Linear(w,w)
        self.ff=nn.Sequential(nn.Linear(w,w*4),nn.GELU(),nn.Linear(w*4,w))
    def forward(self,h,mask):
        b,n,w=h.shape
        q,k,v=self.qkv(self.norm1(h)).reshape(b,n,3,self.heads,w//self.heads).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
        h=h+self.out(a.transpose(1,2).reshape(b,n,w))
        return h+self.ff(self.norm2(h))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); w=192
        self.token=nn.Embedding(len(WORDS),w,padding_idx=0); self.role=nn.Embedding(4,w)
        self.position=nn.Embedding(MAX_TOKENS,w); self.number=nn.Linear(3,w,bias=False)
        self.blocks=nn.ModuleList(Block() for _ in range(4)); self.norm=nn.LayerNorm(w)
        self.valid=nn.Linear(w,1); self.rank=nn.Linear(w,1)
    def forward(self,records,device,head='valid'):
        t,r,n=batch(records,device)
        h=(self.token(t)+self.role(r)+self.number(n))/math.sqrt(3)
        h=h+self.position(torch.arange(t.shape[1],device=t.device))[None]
        for block in self.blocks: h=block(h,t!=0)
        return getattr(self,head)(self.norm(h[:,0])).squeeze(-1)


@torch.no_grad()
def scores(model,records,device,head,chunk=16):
    if not records:return []
    model.eval(); output=[]
    for start in range(0,len(records),chunk):output.extend(model(records[start:start+chunk],device,head).cpu().tolist())
    if not all(math.isfinite(x) for x in output):raise ValueError('Nonfinite model score')
    return output


def initialization(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def list_gradient(logits,labels):
    known=torch.tensor([x is not None for x in labels],device=logits.device)
    positive=torch.tensor([x is True for x in labels],device=logits.device)
    if not positive.any() or not (known & ~positive).any():raise ValueError('No ranking contrast')
    den=logits.masked_fill(~known,float('-inf')); num=logits.masked_fill(~positive,float('-inf'))
    return torch.logsumexp(den,0)-torch.logsumexp(num,0), torch.softmax(den,0)-torch.softmax(num,0)

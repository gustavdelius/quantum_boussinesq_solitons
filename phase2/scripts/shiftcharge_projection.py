#!/usr/bin/env python3
"""(c) Exact two-soliton collision: instantaneous measure and endpoint test.

This evaluates the 4x4 symplectic Gram determinant, NOT a spacetime
functional determinant. The latter remains explicitly null in the cache.
"""
import json
import time
from pathlib import Path
import mpmath as mp

start=time.monotonic(); mp.mp.dps=65
a1,a2=mp.mpf('.3'),mp.mpf('.6')
k1,k2=mp.sin(a1),mp.sin(a2)
v1,v2=mp.cos(a1),mp.cos(a2)
def interaction(a,b):
    k,l=mp.sin(a),mp.sin(b); v,w=mp.cos(a),mp.cos(b)
    aa,bb=(v/mp.sqrt(3)+k)/2,(v/mp.sqrt(3)-k)/2
    cc,dd=(w/mp.sqrt(3)+l)/2,(w/mp.sqrt(3)-l)/2
    return (aa-cc)*(bb-dd)/((aa-dd)*(bb-cc))
A=interaction(a1,a2)
assert A>0
la1=mp.diff(lambda z:mp.log(interaction(z,a2)),a1)
la2=mp.diff(lambda z:mp.log(interaction(a1,z)),a2)
Ks=[mp.mpf(0),k1,k2,k1+k2]
Ws=[mp.mpf(0),k1*v1,k2*v2,k1*v1+k2*v2]

def tangents(x):
    logs=[mp.mpf(0),k1*x,k2*x,mp.log(A)+(k1+k2)*x]
    top=max(logs); ww=[mp.exp(t-top) for t in logs]; norm=sum(ww)
    ww=[t/norm for t in ww]
    mean=lambda f:sum(p*z for p,z in zip(ww,f))
    Kbar,Wbar=mean(Ks),mean(Ws)
    KW=mean([k*w for k,w in zip(Ks,Ws)])
    specs=[([0,v1,0,v1],[0,mp.cos(2*a1),0,mp.cos(2*a1)],[0,v1*x,0,v1*x+la1]),
           ([0,0,0,0],[0,0,0,0],[0,-k1,0,-k1]),
           ([0,0,v2,v2],[0,0,mp.cos(2*a2),mp.cos(2*a2)],[0,0,v2*x,v2*x+la2]),
           ([0,0,0,0],[0,0,0,0],[0,0,-k2,-k2])]
    out=[]
    for dk,dw,dl in specs:
        lbar=mean(dl)
        Km=mean(dk)+mean([k*l for k,l in zip(Ks,dl)])-Kbar*lbar
        Wm=mean(dw)+mean([w*l for w,l in zip(Ws,dl)])-Wbar*lbar
        KWm=mean([dk[i]*Ws[i]+Ks[i]*dw[i] for i in range(4)])+mean([Ks[i]*Ws[i]*dl[i] for i in range(4)])-KW*lbar
        out.append((2*Km,-2*(KWm-Km*Wbar-Kbar*Wm)))
    return out

intervals=[-600,-150,-40,-10,0,10,40,150,600]
W=mp.matrix(4)
for i in range(4):
    for j in range(i+1,4):
        def integrand(x):
            z=tangents(x)
            return z[i][1]*z[j][0]-z[j][1]*z[i][0]
        W[i,j]=mp.quad(integrand,intervals)
        W[j,i]=-W[i,j]
Pprime=lambda a:mp.mpf(2)/3*mp.sin(a)*mp.sin(3*a)
pp1,pp2=Pprime(a1),Pprime(a2)
assert abs(W[0,1]-pp1)<mp.mpf('1e-50')
assert abs(W[2,3]-pp2)<mp.mpf('1e-50')
assert abs(W[0,3])<mp.mpf('1e-50')
assert abs(W[1,2])<mp.mpf('1e-50')
assert abs(W[1,3])<mp.mpf('1e-50')
det=mp.det(W)
assert abs(det-(pp1*pp2)**2)<mp.mpf('1e-50')
# Project a genuinely localized fluctuation eta=(0, exp(-x^2)).
bvec=mp.matrix([mp.quad(lambda x:-mp.exp(-x*x)*tangents(x)[i][0],[-20,-5,0,5,20]) for i in range(4)])
coeff=mp.lu_solve(W,bvec)
right_tail=2*v1*coeff[0]+2*v2*coeff[2]
assert abs(right_tail)>mp.mpf('.01')
assert mp.norm(W*coeff-bvec)<mp.mpf('1e-60')
evaluated_tail=sum(coeff[i]*tangents(mp.mpf(600))[i][0] for i in range(4))
assert abs(evaluated_tail-right_tail)<mp.mpf('1e-50')
out=dict(claim_label='c, instantaneous geometry only',precision_dps=65,alpha1=str(a1),alpha2=str(a2),t='0',X1='0',X2='0',A=str(A),
         symplectic_gram=[[str(W[i,j]) for j in range(4)] for i in range(4)],
         gram_determinant=str(det),expected_gram_determinant=str((pp1*pp2)**2),det_error=str(abs(det-(pp1*pp2)**2)),
         projected_right_tail=str(right_tail),projected_left_tail='0',
         full_two_soliton_projected_functional_determinant=None,
         missing=['Boundary-compatible time-dependent collective/radiation domain.',
                  'Constraint evolution and its Schur complement.',
                  'Incoming/outgoing normalization and time-boundary determinant phase.'],
         failures_recorded=['At the requested single kinematic point the instantaneous projector maps a localized fluctuation to a nonzero potential endpoint.'],
         runtime_seconds=time.monotonic()-start)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Collision measure checked; projected right endpoint =',mp.nstr(right_tail,18))
print('ALL CHECKS PASSED')

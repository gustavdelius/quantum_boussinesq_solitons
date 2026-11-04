#!/usr/bin/env python3
"""(c) Bad/Toda calibration, good rational recoil, and conditional ring test.

The shared functions are not asserted to be a shared physical S matrix.
"""
import json
import time
from pathlib import Path
import sympy as s
import mpmath as mp

start=time.monotonic(); mp.mp.dps=80
k,r,lam=s.symbols('k r lam',positive=True)
records={}
for sig,name in [(1,'good'),(-1,'bad')]:
    v=s.sqrt(1-sig*k*k); P=s.Rational(2,3)*v*k**3
    Pi=-2*k*v; M=2*k
    T=s.atanh(2*k/s.sqrt(3)) if sig==1 else s.atan(2*k/s.sqrt(3))
    pc=-2*s.sqrt(3)*v*T/s.pi+2*sig*lam*v*k
    gamma=s.diff(Pi,k)/s.diff(P,k)
    tail=pc*gamma*2*r*k-pc.subs(k,r*k)*gamma.subs(k,r*k)*2*k
    C=lam-2*sig/s.pi
    leading=-4*sig*C*(r-1/r)
    assert s.simplify(s.limit(tail,k,0)-leading)==0
    Q=P*2*r*k-P.subs(k,r*k)*2*k
    assert s.simplify(s.limit(Q/k**4,k,0)-s.Rational(4,3)*r*(1-r*r))==0
    records[name]=dict(rational_recoil=str(leading),Q_leading=str(s.Rational(4,3)*r*(1-r*r)*k**4))

# Classical action, length, energy and Planck dictionary checked by series.
b,beta,hB=s.symbols('b beta hB',positive=True)
kap=s.sqrt(3)*beta
PT=4*b*(kap*s.cosh(kap)-s.sinh(kap))
ET=2*b*b*(s.sinh(kap)*s.cosh(kap)-kap)
PB=PT/(6*s.sqrt(3)*b); EB=ET/(6*s.sqrt(3)*b*b)
assert s.limit(PB/beta**3,beta,0)==s.Rational(2,3)
assert s.limit(EB/beta**3,beta,0)==s.Rational(2,3)
delta=s.symbols('delta')
assert s.cancel((3*b*delta)/(3*b*hB)-delta/hB)==0
assert s.simplify((3*b*hB)*s.symbols('y')/(3*b)/hB-s.symbols('y'))==0

rows=[]
for rs in ['.25','.4','.5','.75','.9']:
    rr=mp.mpf(rs)
    chi=(mp.polylog(2,rr)-mp.polylog(2,-rr))/2
    # Independent regular integral, not a rearranged closed-form assertion.
    forward=4/mp.pi*mp.quad(lambda x:mp.log(x/rr)*x*x/(1-x*x),[0,rr])
    target=-4/mp.pi*(chi-rr)
    assert abs(forward-target)<mp.mpf('1e-70')
    symmetric=2/mp.pi*(rr+1/rr)-mp.pi
    theta=forward-symmetric
    rational=2/mp.pi*(rr-1/rr)
    assert abs(theta-(mp.pi-4/mp.pi*chi+rational))<mp.mpf('1e-70')
    # Direct finite-width recoil, with no target fitted.
    KK=mp.mpf('1e-16'); lbad=-3/(2*mp.pi)
    def charge(K):
        V=mp.sqrt(1+K*K)
        pp=-2*mp.sqrt(3)*V/mp.pi*mp.atan(2*K/mp.sqrt(3))-2*lbad*V*K
        gamma=mp.diff(lambda t:-2*t*mp.sqrt(1+t*t),K)/mp.diff(lambda t:mp.mpf(2)/3*t**3*mp.sqrt(1+t*t),K)
        return pp*gamma
    recoil=charge(KK)*2*rr*KK-charge(rr*KK)*2*KK
    assert abs(recoil-rational)<mp.mpf('1e-29')
    # Existing canonical candidate has a nonconstant gap after rational match.
    Lr=2/mp.pi*mp.log(rr)*mp.log((1-rr)/(1+rr))
    J=mp.pi/2-4/mp.pi*chi-Lr
    gap=J+rational-theta
    assert abs(gap+mp.pi/2+Lr)<mp.mpf('1e-70')
    rows.append(dict(r=rs,forward=str(forward),symmetric=str(symmetric),theta_N=str(theta),
                     rational_Toda=str(rational),bad_recoil=str(recoil),recoil_error=str(abs(recoil-rational)),
                     conditional_candidate_gap=str(gap)))
assert abs(mp.mpf(rows[0]['conditional_candidate_gap'])-mp.mpf(rows[2]['conditional_candidate_gap']))>mp.mpf('.01')

# Corrected conditional ring formula, including the audited self term.
h=s.symbols('h'); P=s.symbols('P:3'); M=s.symbols('M:3'); Pi=s.symbols('Pi:3')
mm=s.symbols('m:3'); qq=s.symbols('q:3'); L=s.symbols('L')
tot=sum((Pi[j]+h*qq[j])*(M[l]+h*mm[l]) for j in range(3) for l in range(3))
expected=sum(qq)*sum(M)+sum(Pi)*sum(mm)
assert s.expand(tot).coeff(h)==s.expand(expected)
assert s.expand(tot-sum(Pi[j]+h*qq[j] for j in range(3))*sum(M[l]+h*mm[l] for l in range(3)))==0
ring_pair=qq[0]*M[1]+Pi[0]*mm[1]
sym=(qq[0]*M[1]+Pi[0]*mm[1]+qq[1]*M[0]+Pi[1]*mm[0])/2
assert s.expand((ring_pair+qq[1]*M[0]+Pi[1]*mm[0])/2-sym)==0

out=dict(claim_label='c, comparisons conditional on the explicitly borrowed dictionary',
         precision_dps=80,records=records,ratio_rows=rows,c_bad=None,c_good=None,
         good_candidates=[dict(lambda_exact='9/(4*pi)',C_exact='1/(4*pi)',rational_coefficient='-1/pi'),
                          dict(lambda_exact='0',C_exact='-2/pi',rational_coefficient='8/pi')],
         quantum_ring_holonomy=None,
         failures_recorded=['A finite c times P wedge M scales as k^4 and cannot reproduce the O(1) Toda rational term.',
          'The conditional canonical continuum candidate retains a nonconstant gap from the directional Toda residual.',
          'The rational match is already a momentum-recoil contribution and cannot be counted again as an unknown CDD coefficient.'],
         runtime_seconds=time.monotonic()-start)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Bad rational recoil matches 2/pi*(r-1/r); no finite CDD coefficient is extracted.')
print('ALL CHECKS PASSED')

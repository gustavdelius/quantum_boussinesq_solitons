#!/usr/bin/env python3
"""Potential-field Jacobi and continuum modes, with 60-digit residuals.

Checks exact solution families and static kernel identities, not completeness
of the nonzero-frequency spectral pencil or its functional determinant.
"""
import json
from pathlib import Path
import time
import mpmath as mp
import sympy as s

started=time.perf_counter()
mp.mp.dps=60
x,k,om,q,v=s.symbols('x k om q v',real=True)
t=s.symbols('t',real=True)
assert s.diff(t,t,2)-2*v*s.diff(t,x,t)+v*v*s.diff(t,x,2)+s.diff(t,x,4)-s.diff(t,x,2)==0
f=s.Function('f')(x)
assert s.expand((-s.I*om)**2*f+2*s.I*v*om*s.diff(f,x)+v*v*s.diff(f,x,2)
                -(-om*om*f+2*s.I*v*om*s.diff(f,x)+v*v*s.diff(f,x,2)))==0
quartic=q**4+(1-v*v)*q*q-2*v*om*q-om*om
assert s.expand(s.diff(quartic,q,2)-(12*q*q+2*(1-v*v)))==0
assert s.expand(quartic-(-(om+v*q)**2+q*q+q**4))==0
hh=s.Function('h')(x); jj=s.Function('j')(x); V=s.Function('V')(x)
assert s.expand(s.diff(hh*s.diff(jj,x)-jj*s.diff(hh,x),x)-
                (hh*(s.diff(jj,x,2)-V*jj)-jj*(s.diff(hh,x,2)-V*hh)))==0

rows=[]; worst=mp.mpf(0); wrong=[]
for astr in ['0.3','0.8','1.2']:
    a=mp.mpf(astr); kk=mp.sin(a)
    for eps in [1,-1]:
        vv=eps*mp.cos(a)
        u=lambda X:kk**2/(2*mp.cosh(kk*X/2)**2)
        phi=lambda X:kk*mp.tanh(kk*X/2)
        def Kop(F,X):
            return mp.diff(F,X,4)-(1-vv*vv)*mp.diff(F,X,2)+6*mp.diff(lambda Y:u(Y)*mp.diff(F,Y),X)
        def pencil(F,X,lam):
            return Kop(F,X)-2*vv*lam*mp.diff(F,X)+lam*lam*F(X)
        def phiv(X):
            return -vv/kk*(mp.tanh(kk*X/2)+kk*X/(2*mp.cosh(kk*X/2)**2))
        for X in map(mp.mpf,['-1.1','0.37']):
            errs=[abs(Kop(u,X)),abs(Kop(lambda Y:mp.mpf(1),X)),
                  abs(Kop(phiv,X)+2*vv*mp.diff(u,X))]
            # f_v=Phi_v-t u: its mixed time-space term cancels K Phi_v.
            assert max(errs)<mp.mpf('1e-50'); worst=max(worst,*errs)
        for qq in map(mp.mpf,['0.4','-1.1']):
            omega=abs(qq)*mp.sqrt(1+qq*qq)
            pa=(kk+vv/mp.sqrt(3))/2; pb=(vv/mp.sqrt(3)-kk)/2
            qa=mp.sign(qq)*mp.sqrt(1+qq*qq)/(2*mp.sqrt(3))+1j*qq/2
            qb=mp.conj(qa)
            A=(pa-qa)*(pb-qb)/((pa-qb)*(pb-qa))
            F=lambda X:mp.diff(lambda Y:(1+A*mp.exp(kk*Y))/(1+mp.exp(kk*Y))*mp.exp(1j*qq*Y),X)
            res=abs(pencil(F,mp.mpf('0.37'),-1j*(omega-vv*qq)))
            assert res<mp.mpf('1e-49') and abs(abs(A)-1)<mp.mpf('1e-55')
            worst=max(worst,res)
        if a>mp.pi/3:
            gamma=mp.sqrt(3)*(kk*kk-mp.mpf(3)/4)
            Ksmall=mp.sin(a-mp.pi/3); Klarge=mp.sin(a+mp.pi/3)
            assert abs(Ksmall+Klarge-kk)<mp.mpf('1e-55')
            assert 0<Ksmall<kk and 0<Klarge<kk
            for kap,lam in [(Ksmall,eps*gamma),(Klarge,-eps*gamma)]:
                F=lambda X:mp.diff(lambda Y:mp.exp(kap*Y)/(1+mp.exp(kk*Y)),X)
                res=abs(pencil(F,mp.mpf('0.37'),lam))
                assert res<mp.mpf('1e-49'); worst=max(worst,res)
            Fsmall=lambda X:mp.diff(lambda Y:mp.exp(Ksmall*Y)/(1+mp.exp(kk*Y)),X)
            bad=abs(pencil(Fsmall,mp.mpf('0.37'),-eps*gamma))
            assert bad>mp.mpf('1e-4')
            wrong.append({'alpha':astr,'epsilon':eps,'same_profile_opposite_rate_residual':str(bad)})
        rows.append({'alpha':astr,'epsilon':eps})

result={'precision':mp.mp.dps,'points':rows,'max_PDE_residual':str(worst),
        'failures_recorded':wrong,
        'implementation_failures':['Initial SymPy structural equality test for quartic second derivative failed; checking the expanded difference fixes the test without changing the identity or a tolerance.'],
        'scope':'No claim of spectral completeness or absolute determinant normalization.',
        'runtime_seconds':time.perf_counter()-started}
Path(__file__).with_name('det_spectrum_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('ALL CHECKS PASSED')

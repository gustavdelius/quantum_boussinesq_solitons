#!/usr/bin/env python3
"""Fresh dictionary algebra, arcsine integrals and finite-sea endpoint checks.
Does not import or modify any previous project script.
"""
from pathlib import Path
import json
import sympy as s
import mpmath as m
m.mp.dps=60
a,b,k,B,hB,PT,ET=s.symbols('a b k B hB PT ET',positive=True)
ft,fx,fxx=s.symbols('ft fx fxx')
lx,lt,lf=2*s.sqrt(3),2*s.sqrt(3)*b,-s.sqrt(3)
LT=(lf*lt*ft)**2/2-b*b*(lf*lx*fx)**2/2+b*b*(lf*lx*fx)**3/6+b*b*(lf*lx*lx*fxx)**2/24
LB=ft*ft/2-fx*fx/2-fx**3+fxx*fxx/2
assert s.simplify(LT/(lx*lt)-3*b*LB)==0
assert s.simplify(lf**2*lt/lx)==3*b
P0=4*b*(k*s.cosh(k)-s.sinh(k))/a
E0=2*b*b*(s.sinh(k)*s.cosh(k)-k)
PB=2*s.sinh(B)**3*s.cosh(B)/3
EB=2*s.sinh(B)**3/3+8*s.sinh(B)**5/15
mappedP=P0.subs(k,s.sqrt(3)*B)*a/(6*s.sqrt(3)*b)
mappedE=E0.subs(k,s.sqrt(3)*B)/(6*s.sqrt(3)*b*b)
for numerator,denominator in ((mappedP,PB),(mappedE,EB),(mappedE-mappedP,EB-PB)):
    assert s.limit(numerator/denominator,B,0)==1
assert s.simplify(s.diff(E0,k)/s.diff(P0,k)-a*b*s.sinh(k)/k)==0
assert s.limit((-2*s.sqrt(3)*B)/(-s.sqrt(3)*2*s.sinh(B)),B,0)==1
assert s.simplify((ET/(6*s.sqrt(3)*b*b))/(hB*a*PT/(6*s.sqrt(3)*b))-(3/a)*ET/((3*b*hB)*PT))==0

# Independently integrate the classical arcsine Bethe density, rather than
# merely substitute the proposed dressed momentum definition.
arcsine=[]
for aa in map(m.mpf,['0.7','1.4']):
    bb=m.exp(-aa/2)
    for kk in map(m.mpf,['0.15','0.6']):
        pp=2*bb*m.cosh(kk)
        theta=lambda d:2*d*(m.log(d)-1)
        pdr=pp+m.quad(lambda t:theta(pp-2*bb*m.cos(t)),[0,m.pi/4,m.pi])/(aa*m.pi)
        expected=4*bb/aa*(kk*m.cosh(kk)-m.sinh(kk))
        assert abs(pdr-expected)<m.mpf('1e-53')
        arcsine.append({'a':str(aa),'kappa':str(kk),'error':str(abs(pdr-expected))})

# Two-soliton phase check: both channels, exact Hirota polynomials, no leading
# head-on formula assumed as input.
phase=[]
bb=m.exp(-m.mpf('.5'))
for sign in (1,-1):
    errors=[]
    for scale in map(m.mpf,['0.001','0.0001','0.00001']):
        be1,be2=scale*m.mpf('1.7'),scale
        k1,k2=m.sqrt(3)*be1,m.sqrt(3)*be2
        polyT=lambda K,W:W*W-4*bb*bb*m.sinh(K/2)**2
        AT=-polyT(2*(k1-k2),2*bb*(m.sinh(k1)-sign*m.sinh(k2)))/polyT(2*(k1+k2),2*bb*(m.sinh(k1)+sign*m.sinh(k2)))
        dt=2*bb*(m.cosh(k1)-sign*m.cosh(k2))*m.log(AT)+4*k1*k2*bb*(m.sinh(k1)/k1-sign*m.sinh(k2)/k2)
        u1,u2=m.sinh(be1),m.sinh(be2)
        w1,w2=u1*m.cosh(be1),sign*u2*m.cosh(be2)
        polyB=lambda K,W:W*W-K*K-K**4
        AB=-polyB(u1-u2,w1-w2)/polyB(u1+u2,w1+w2)
        db=m.mpf(2)/9*(m.cosh(3*be1)-sign*m.cosh(3*be2))*m.log(AB)+m.mpf(8)/3*u1*u2*(m.cosh(be1)-sign*m.cosh(be2))
        errors.append(abs(dt/(3*bb*db)-1))
    assert errors[2]<m.mpf('1e-8') and errors[2]<errors[1]/90 and errors[1]<errors[0]/90
    phase.append({'orientation':sign,'errors':list(map(str,errors))})

# Fresh high-precision finite sea. LU solve, with endpoint evaluation separate
# from Gauss nodes. Differences are independent sea solves at q +/- epsilon.
def sea(N,q,h):
    nodes,weights=m.gauss_quadrature(N,'legendre')
    x=[q*z for z in nodes];w=[q*z for z in weights]
    K=lambda d:2*m.re(m.digamma(1+1j*d/h))+2*m.log(h)
    Th=lambda d:2*h*m.im(m.loggamma(1+1j*d/h))+2*d*m.log(h)
    A=m.matrix([[2*m.pi*h*(i==j)-K(x[i]-x[j])*w[j] for j in range(N)] for i in range(N)])
    rho=m.lu_solve(A,m.ones(N,1));eta=m.lu_solve(A,m.matrix(x))
    D=lambda p:1+sum(wi*K(p-xi)*ri for wi,xi,ri in zip(w,x,rho))
    V=lambda p:p+sum(wi*K(p-xi)*ei for wi,xi,ei in zip(w,x,eta))
    P=lambda p:p+sum(wi*Th(p-xi)*ri for wi,xi,ri in zip(w,x,rho))
    n=sum(wi*ri for wi,ri in zip(w,rho))
    return n,D,V,P
q=m.mpf(1);hh=m.mpf('.8');eps=m.mpf('1e-15');rows=[]
for N in (32,48):
    n,D,V,P=sea(N,q,hh)
    np,dp,vp,pp=sea(N,q+eps,hh)
    nm,dm,vm,pm=sea(N,q-eps,hh)
    aspace=1/n;vs=V(q)/D(q)
    endpoint=abs(D(q)*V(q)-m.pi*hh*n)
    pedge=abs(P(q)-m.pi*hh*n)
    maxwell=abs((pp(m.mpf('2.2'))/np-pm(m.mpf('2.2'))/nm)/(1/np-1/nm)-V(m.mpf('2.2')))
    speed=abs(vs+aspace/((1/np-1/nm)/(2*eps)))
    rows.append(dict(N=N,endpoint=str(endpoint),momentum_origin=str(pedge),maxwell=str(maxwell),speed=str(speed)))
    assert max(endpoint,pedge,maxwell,speed)<(m.mpf('1e-17') if N==32 else m.mpf('1e-27'))
nhp,_,_,_=sea(48,q,hh+eps)
nhm,_,_,_=sea(48,q,hh-eps)
aq=((1/np)-(1/nm))/(2*eps)
ah=((1/nhp)-(1/nhm))/(2*eps)
q_h_at_a=-ah/aq
scaling=abs(vs-aspace*(q-hh*q_h_at_a)/2)
assert scaling<m.mpf('1e-27')

out={'arcsine':arcsine,'phase':phase,'finite_sea':rows,'sound_scaling_error':str(scaling),
     'dictionary_verdict':'Six leading classical factors are one; this does not fix quantum finite counterterms.',
     'correction':'b=exp(-a/2), not pi/a.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
print('ALL CHECKS PASSED')

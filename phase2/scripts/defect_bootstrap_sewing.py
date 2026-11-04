#!/usr/bin/env python3
"""(c) Two CZ defects in series, checked from the variational action.
Uses independently constructed tau fields. No legacy code imported or executed.
Finite high-precision checks are not a proof of a complete quantum spectrum."""
from pathlib import Path
import json,time,itertools
import sympy as S
import mpmath as m
start=time.perf_counter();m.mp.dps=70;r=m.sqrt(3);tol=m.mpf('1e-52')
# Independent variational equations in CZ units, with U_CZ=-3 phi.
U,V,UX,VX,UT,VT,UXT,VXT,UXX,VXX,UXXX,VXXX,s,z=S.symbols('U V UX VX UT VT UXT VXT UXX VXX UXXX VXXX s z')
args=[U,UX,UXX,UXXX,UT,UXT,V,VX,VXX,VXXX,VT,VXT,s]
sew={}
for eta in [-1,1]:
    al=S.Integer(eta)/S.sqrt(3);be=2*al;mu=al
    sig=s+mu*(U-V)
    rhoz=(z-z**3/3-s+s**3/3)/(2*mu)
    rho=rhoz.subs(z,sig)
    tau=-S.integrate(z*rhoz,(z,s,sig))
    f=-sig**2/(4*mu)
    F=al*UX+be*VX+f;G=be*UX+al*VX+f
    H=sig*(be*UX**2+2*al*UX*VX+be*VX**2)/2+rho*(UX+VX)+tau
    D=UT*F+VT*G+H
    dt=lambda g:sum(S.diff(g,j)*jj for j,jj in [(U,UT),(V,VT),(UX,UXT),(VX,VXT)])
    equations=[-UX-UX**2+UXXX+S.diff(D,U)-dt(F),-UXX+S.diff(D,UX),
               VX+VX**2-VXXX+S.diff(D,V)-dt(G),VXX+S.diff(D,VX)]
    sew[eta]=S.lambdify(args,[S.expand(j) for j in equations],'mpmath')

def wave(a,ep):
    k=m.sin(a);v=ep*m.cos(a);return (k,k*v)
def pol(k,w):return w*w-k*k+k**4
def interaction(wi,wj):return -pol(wi[0]-wj[0],wi[1]-wj[1])/pol(wi[0]+wj[0],wi[1]+wj[1])
def factor(w,Q,eta):
    k,ww=w;v=ww/k
    return ((v/r-k-2*Q)/(v/r+k-2*Q))**eta

def makefield(waves,weights):
    # Store exponential coefficients; the rational logarithmic derivative
    # avoids numerical differencing of log tau to construct the field.
    terms=[]
    for mask in itertools.product([0,1],repeat=len(waves)):
        coeff=m.mpf(1);K=W=m.mpf(0)
        for j,on in enumerate(mask):
            if on:coeff*=weights[j];K+=waves[j][0];W+=waves[j][1]
        for i in range(len(waves)):
            for j in range(i+1,len(waves)):
                if mask[i] and mask[j]:coeff*=interaction(waves[i],waves[j])
        terms.append((coeff,K,W))
    def field(x,t):
        vals=[(c*m.exp(k*x-w*t),k) for c,k,w in terms]
        return -6*sum(v*k for v,k in vals)/sum(v for v,k in vals)
    return field,terms

def jet(f,x,t):return [f(x,t),m.diff(lambda xx:f(xx,t),x),m.diff(lambda xx:f(xx,t),x,2),
    m.diff(lambda xx:f(xx,t),x,3),m.diff(lambda tt:f(x,tt),t),m.diff(lambda xx:m.diff(lambda tt:f(xx,tt),t),x)]
def residual(waves,base,Q1,Q2,e1,e2,which,x,t):
    z1=[factor(w,Q1,e1) for w in waves];z2=[factor(w,Q2,e2) for w in waves]
    left=base if which==1 else [c*z for c,z in zip(base,z1)]
    right=[c*z for c,z in zip(left,z1 if which==1 else z2)]
    f,_=makefield(waves,left);g,_=makefield(waves,right)
    return sew[e1 if which==1 else e2](*jet(f,x,t),*jet(g,x,t),-2*r*(Q1 if which==1 else Q2))
worst=m.mpf(0);n=0;ordererr=m.mpf(0);hirota=m.mpf(0)
waves=[wave(m.mpf('.2'),1),wave(m.mpf('.4'),1)]
for e1,e2 in itertools.product([-1,1],repeat=2):
    Q1=e1*m.mpf('.7');Q2=e2*m.mpf('.8')
    for tt in ['-2','.3','4']:
        for which,xx in [(1,m.mpf(0)),(2,m.mpf('1.2'))]:
            res=residual(waves,[m.mpf('.8'),m.mpf('1.3')],Q1,Q2,e1,e2,which,xx,m.mpf(tt))
            worst=max(worst,*[abs(j) for j in res]);n+=4
    weights=[factor(w,Q1,e1)*factor(w,Q2,e2) for w in waves]
    f,terms=makefield(waves,weights)
    f_rev,_=makefield(waves,[factor(w,Q2,e2)*factor(w,Q1,e1) for w in waves])
    ordererr=max(ordererr,abs(f(m.mpf('.33'),m.mpf('.27'))-f_rev(m.mpf('.33'),m.mpf('.27'))))
    # Full two-soliton Hirota coefficient sum independently from sewing.
    for x,t in [(m.mpf('.2'),m.mpf('-.3'))]:
        val=sum(ci*cj*m.exp((ki+kj)*x-(wi+wj)*t)*pol(ki-kj,wi-wj) for ci,ki,wi in terms for cj,kj,wj in terms)
        hirota=max(hirota,abs(val))
assert worst<tol,('nonlinear sewing',worst)
assert hirota<tol,('bulk Hirota',hirota)
assert ordererr<tol
# Linearised two-soliton+phonon mode across BOTH defects; differentiate the
# nonlinear sewing functional with respect to a wave amplitude at zero.
linear=m.mpf(0);modulus=m.mpf(0);nlin=0
for e1,e2 in [(1,1),(-1,1)]:
    Q1=e1*m.mpf('.7');Q2=e2*m.mpf('.8')
    for q in [m.mpf('-.6'),m.mpf('.6')]:
        ph=(1j*q,1j*abs(q)*m.sqrt(1+q*q));ww=waves+[ph]
        for which,xx in [(1,m.mpf(0)),(2,m.mpf('1.2'))]:
            for eq in range(4):
                fn=lambda amp:residual(ww,[m.mpf('.8'),m.mpf('1.3'),amp],Q1,Q2,e1,e2,which,xx,m.mpf('.25'))[eq]
                rr=m.diff(fn,m.mpf(0));linear=max(linear,abs(rr));nlin+=1
        total=interaction(waves[0],ph)*interaction(waves[1],ph)*factor(ph,Q1,e1)*factor(ph,Q2,e2)
        modulus=max(modulus,abs(abs(total)-1))
assert linear<tol,('linearised sewing',linear)
assert modulus<tol
out={'label':'(c) finite checks of exact tau/action expressions','digits':70,'tolerance':str(tol),
 'nonlinear_equations_tested':n,'linearised_equations_tested':nlin,
 'nonlinear_max':str(worst),'linearised_max':str(linear),'bulk_hirota_max':str(hirota),
 'order_error':str(ordererr),'phonon_modulus_error':str(modulus),'seconds':time.perf_counter()-start}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2));print('ALL CHECKS PASSED')

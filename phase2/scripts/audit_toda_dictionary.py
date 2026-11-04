#!/usr/bin/env python3
"""Exact-algebra and independent high-precision Toda dictionary audit.

The core high-precision solver is independent. An isolated final comparison
uses the original implementation, including its edge offset. Consequences of
the existing Fredholm equations are tested; no physical frame is chosen.
Run from any directory. Writes only audit_toda_dictionary_results.json.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = ['phase2/scripts/toda_edge_ba.py',
          'phase2/scripts/toda_edge_twoterm.py',
          'phase2/scripts/toda_ba_vs_label.py',
          'phase2/charge_frame_master.tex']
hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}

# Exact bilinear decomposition and chain rules, independently of quadrature.
fp,fm,gp,gm,ep,em,dp,dm,t = sp.symbols('fp fm gp gm ep em dp dm t')
af,bf=fp+fm-ep-em,fp-fm-ep+em
ag,bg=gp+gm-dp-dm,gp-gm-dp+dm
rhs=t*((fp-ep)*(gp-dp)-(fm-em)*(gm-dm))
assert sp.expand(rhs-t*(bf*ag+af*bg)/2)==0
a,b,k,h,y,p,rhoq,Dp,De=sp.symbols('a b k h y p rhoq Dp De', positive=True)
Pn=4*b*(k*sp.cosh(k)-sp.sinh(k))
assert sp.simplify(sp.diff(Pn/a,k)*h/(2*b*k)-2*h*sp.sinh(k)/a)==0
assert sp.simplify((2*h*sp.sinh(k)/(a*Dp))/h-2*sp.sinh(k)/(a*Dp))==0
# Classical momentum, viewed at fixed bare spectral p, and its pressure response.
Pcl=2*(p*sp.acosh(p/(2*sp.exp(-a/2)))-sp.sqrt(p*p-4*sp.exp(-a)))/a
Pc=Pn/a; Pi=2*b*sp.sinh(k)
Pprime=sp.diff(Pc,k)/(2*b*sp.sinh(k))
assert sp.simplify(Pprime-2*k/a)==0
pressure=sp.diff(Pcl,a).subs(p,2*sp.exp(-a/2)*sp.cosh(k))
# For k>0, sqrt(sinh(k)^2)=sinh(k), acosh(cosh(k))=k.
pressure=pressure.replace(sp.acosh(sp.cosh(k)),k)
pressure=sp.refine(sp.simplify(pressure),sp.Q.positive(k))
pressure=pressure.subs(sp.sqrt(sp.cosh(k)**2-1),sp.sinh(k))
assert sp.simplify(pressure-((Pi-Pc)/a).subs(b,sp.exp(-a/2)))==0
# The actual edge() implementation has a fixed relative offset; do not hide it.
q,eta=sp.symbols('q eta', positive=True)
assert sp.simplify((q*(1+eta)-q)/h-eta*q/h)==0
print('PASS: exact endpoint bilinear, momentum Jacobian, classical responses, edge offset.',flush=True)

class IndependentBA:
    """Independent mpmath Nystrom solver at fixed h,q; a follows from counting.

    Fixing h,q and calculating a=1/int rho is the same ground-state family as
    fixing h,a and solving for q, avoiding a second nonlinear solve here.
    """
    def __init__(self,N):
        self.N=N; self.h=mp.mpf('.4'); self.q=mp.mpf('1.5')
        self.t=2*mp.pi*self.h
        X,W=mp.gauss_quadrature(N,'legendre')
        self.x=self.q*X; self.w=self.q*W
        A=mp.matrix([[self.t*(i==j)-self.K(self.x[i]-self.x[j])*self.w[j]
                      for j in range(N)] for i in range(N)])
        self.inv=A**-1
        self.rho=self.inv*mp.ones(N,1)
        self.a=1/sum(self.w[i]*self.rho[i] for i in range(N))
        self.b=mp.exp(-self.a/2)
        self.rhoq=self.D(self.q)/self.t
    def K(self,d):
        return 2*mp.re(mp.digamma(1+mp.j*d/self.h))+2*mp.log(self.h)
    def Th(self,d):
        return 2*self.h*mp.im(mp.loggamma(1+mp.j*d/self.h))+2*d*mp.log(self.h)
    def Kprime(self,d):
        return -2*mp.im(mp.polygamma(1,1+mp.j*d/self.h))/self.h
    def dot(self,x,y):
        return sum(self.w[i]*x[i]*y[i] for i in range(self.N))
    @lru_cache(None)
    def kv(self,p):return mp.matrix([self.K(p-x) for x in self.x])
    @lru_cache(None)
    def tv(self,p):return mp.matrix([self.Th(p-x) for x in self.x])
    @lru_cache(None)
    def F(self,p):return self.inv*self.tv(p)
    @lru_cache(None)
    def Fprime(self,p):return self.inv*self.kv(p)
    def R(self,p,z):return self.K(p-z)+self.dot(self.kv(p),self.Fprime(z))
    def P(self,p):return p+self.dot(self.tv(p),self.rho)
    def D(self,p):return 1+self.dot(self.kv(p),self.rho)
    def Phi(self,p,z):return self.Th(p-z)-self.dot(self.kv(p),self.F(z))
    @lru_cache(None)
    def ends(self,p):
        return tuple((self.Th(p-x)+self.dot(self.kv(x),self.F(p)))/self.t
                     for x in [self.q,-self.q])
    def Ptarget(self,k):return 4*self.b*(k*mp.cosh(k)-mp.sinh(k))/self.a
    def invert(self,k):
        target=self.Ptarget(k); P0=self.P(self.q)
        return mp.findroot(lambda p:self.P(p)-P0-target,
                           self.q+target/self.D(self.q),
                           df=self.D, solver='newton',tol=mp.mpf('1e-50'))
    def responses(self,p,e):
        f,g=self.ends(p),self.ends(e)
        aa=f[0]+f[1]-g[0]-g[1]
        bb=f[0]-f[1]-g[0]+g[1]
        return self.t*self.a*self.rhoq*bb, aa/(2*self.a*self.rhoq)
    def four(self,p,z,e1,e2):
        return self.Phi(p,z)-self.Phi(p,e2)-self.Phi(e1,z)+self.Phi(e1,e2)
    def two(self,p,z,e2):return self.Phi(p,z)-self.Phi(p,e2)

rows=[]
for N in [40,64,88]:
    begin=time.monotonic(); m=IndependentBA(N)
    pairs=[(m.q+mp.mpf('.7'),m.q+mp.mpf('.2'),m.q,m.q),
           (m.q+mp.mpf('.7'),-m.q-mp.mpf('.2'),m.q,-m.q)]
    errors={'edge_momentum':abs(m.P(m.q)-mp.pi*m.h/m.a),
            'reciprocity':mp.mpf(0),'fourterm_symmetric':mp.mpf(0),
            'two_vs_four':mp.mpf(0),'response_derivative':mp.mpf(0),
            'backflow_momentum':mp.mpf(0),
            'intermediate_R_identities':mp.mpf(0),
            'sea_density_derivative':abs(m.D(m.x[N//2])-m.t*m.rho[N//2])}
    for p,z,e1,e2 in pairs:
        f,g=m.ends(p),m.ends(z)
        errors['backflow_momentum']=max(errors['backflow_momentum'],abs(m.P(p)-p-sum(m.w[i]*m.F(p)[i] for i in range(m.N))))
        Kpprime=mp.matrix([m.Kprime(p-x) for x in m.x])
        dfirst=m.K(p-z)-m.dot(Kpprime,m.F(z))
        dsecond=-m.K(p-z)-m.dot(m.kv(p),m.Fprime(z))
        boundary=m.R(p,m.q)*g[0]-m.R(p,-m.q)*g[1]
        ends_derivative=[(m.K(p-e)+m.dot(m.kv(e),m.Fprime(p)))/m.t for e in [m.q,-m.q]]
        intermediate=max(abs(dsecond+m.R(p,z)),abs(m.R(p,z)-m.R(z,p)),
                         abs(dfirst+dsecond-boundary),
                         abs(ends_derivative[0]-m.R(m.q,p)/m.t),
                         abs(ends_derivative[1]-m.R(-m.q,p)/m.t))
        errors['intermediate_R_identities']=max(errors['intermediate_R_identities'],intermediate)
        sym=m.Phi(p,z)+m.Phi(z,p)
        end=m.t*(f[0]*g[0]-f[1]*g[1])
        errors['reciprocity']=max(errors['reciprocity'],abs(sym-end))
        M1,Q1=m.responses(p,e1);M2,Q2=m.responses(z,e2)
        full=m.four(p,z,e1,e2)+m.four(z,p,e2,e1)
        errors['fourterm_symmetric']=max(errors['fourterm_symmetric'],abs(full-M1*Q2-M2*Q1))
        onebody=m.Phi(e1,z)-m.Phi(e1,e2)+m.Phi(e2,p)-m.Phi(e2,e1)
        errors['two_vs_four']=max(errors['two_vs_four'],abs(m.two(p,z,e2)+m.two(z,p,e1)-full-onebody))
        errors['response_derivative']=max(errors['response_derivative'],abs(M1+m.a*(m.D(p)-m.D(e1))))
    # Actual implemented classical frame, expressed at fixed dressed momenta.
    k1s=list(map(mp.mpf,['.8','1.0'])); k2s=list(map(mp.mpf,['.35','.5']))
    points={}; responses={}; inversions=[]
    for k in k1s+k2s:
        p=m.invert(k); points[k]=p; responses[k]=m.responses(p,m.q)
        inversions.append(dict(kappa=str(k),p=str(p),u=str((p-m.q)/m.h),
                               target_momentum=str(m.Ptarget(k)),
                               inversion_error=str(abs(m.P(p)-m.P(m.q)-m.Ptarget(k))),
                               quantum_M=str(responses[k][0]),quantum_Q=str(responses[k][1]),
                               classical_M=str(-2*k),classical_Q=str(m.Ptarget(k)-2*m.b*mp.sinh(k))))
    def classframe(k,z):return (-2*k)*(m.Ptarget(z)-2*m.b*mp.sinh(z))
    def Sret(k,z):
        p,r=points[k],points[z]
        return m.two(p,r,m.q)+m.two(r,p,m.q)-classframe(k,z)-classframe(z,k)
    def rectangle(fun):
        return fun(k1s[1],k2s[1])-fun(k1s[0],k2s[1])-fun(k1s[1],k2s[0])+fun(k1s[0],k2s[0])
    actual=rectangle(Sret)
    def response_difference(k,z):
        M1,Q1=responses[k];M2,Q2=responses[z]
        return M1*Q2+M2*Q1-classframe(k,z)-classframe(z,k)
    factored=rectangle(response_difference)
    errors['rectangle_factorization']=abs(actual-factored)
    rows.append(dict(N=N,a=str(m.a),q=str(m.q),h=str(m.h),
                     errors={k:str(v) for k,v in errors.items()},
                     classical_frame_rectangle=str(actual),
                     rectangle_in_phase_units=str(actual/m.h),
                     inversions=inversions,seconds=time.monotonic()-begin))
    print('N=',N,'a=',mp.nstr(m.a,16),'max identity error=',mp.nstr(max(errors.values()),5),
          'classical-frame rectangle / h=',mp.nstr(actual/m.h,16),flush=True)
    assert max(errors.values()) < {40:mp.mpf('1e-8'),64:mp.mpf('1e-13'),88:mp.mpf('1e-19')}[N]
    assert abs(actual/m.h)>mp.mpf('1e-4'), 'The claimed nonseparable frame discrepancy disappeared'
assert abs(mp.mpf(rows[-1]['classical_frame_rectangle'])-mp.mpf(rows[-2]['classical_frame_rectangle']))<mp.mpf('1e-13')
# Isolated original-code comparison; this supplies no input to the independent
# high-precision derivation or solver above, and retains the actual edge offset.
from toda_edge_ba import EdgeBA, Pn, Pi
import numpy as np
old=EdgeBA(float(rows[-1]['a']),.4,npts=120)
ks=[.8,1.,.35,.5]
pp={k:old.p_of_P(Pn(k,1,old.b)/old.a,1) for k in ks}
edge=old.edge(1)
def old_s(k,z):
    p,r=pp[k],pp[z]
    N12=old.Phi(p,r)-old.Phi(p,edge)
    N21=old.Phi(r,p)-old.Phi(r,edge)
    C12=(-2*k)*(Pn(z,1,old.b)/old.a-Pi(z,1,old.b))
    C21=(-2*z)*(Pn(k,1,old.b)/old.a-Pi(k,1,old.b))
    return N12+N21-C12-C21
old_rectangle=(old_s(1.,.5)-old_s(.8,.5)-old_s(1.,.35)+old_s(.8,.35))/old.h
comparison_error=abs(old_rectangle-float(rows[-1]['rectangle_in_phase_units']))
assert comparison_error<2e-10
original_comparison=dict(nodes=120,actual_edge_offset=float(edge-old.q),
                         rectangle_in_phase_units=float(old_rectangle),
                         difference_from_independent=float(comparison_error))
print('Original code including edge offset: rectangle/h=',old_rectangle,
      'difference=',comparison_error,flush=True)
assert hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
out=dict(precision=mp.mp.dps,input_sha256=hashes,rows=rows,original_comparison=original_comparison,
         scope='Consequences of existing Toda Fredholm equations; no physical frame convention chosen.',
         failures_recorded=['Exploratory N=24 and N=40 meshes gave endpoint reciprocity errors about 3.5e-7 and 6.3e-11; refinement was required, not a change of identity.',
                            'Classical frame fails to remove the finite-hbar nonseparable symmetric two-body part; this is not a fitted edge-limit assertion.'])
HERE.joinpath('audit_toda_dictionary_results.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""Independent 60-digit finite-sea checks and exact symbolic dictionary checks.
No import of earlier project solvers; writes frame_identities.json only.
"""
import time,json
from pathlib import Path
import mpmath as mp
import sympy as s
mp.mp.dps=60
START=time.monotonic(); HERE=Path(__file__).resolve().parent
# Fixed-observable chain rules, exact classical scale conversion.
k,b,a,h,y,be=s.symbols('k b a h y be',positive=True)
PT=4*b*(k*s.cosh(k)-s.sinh(k))/a
ET=2*b*b*(s.sinh(k)*s.cosh(k)-k)
assert s.simplify(s.diff(ET,k)/s.diff(PT,k)-a*b*s.sinh(k)/k)==0
assert s.simplify(s.diff(PT,k)*h/(2*b*k)-2*h*s.sinh(k)/a)==0
PB=s.Rational(2,3)*s.sinh(be)**3*s.cosh(be)
EB=s.Rational(2,15)*s.sinh(be)**3*(5+4*s.sinh(be)**2)
assert s.limit(PT.subs(k,s.sqrt(3)*be)*a/(6*s.sqrt(3)*b*PB),be,0)==1
assert s.limit(ET.subs(k,s.sqrt(3)*be)/(6*s.sqrt(3)*b*b*EB),be,0)==1
assert s.limit((ET-a*b*PT)/(b*b*k**5),k,0)==s.Rational(2,15)
assert s.limit((EB-PB)/be**5,be,0)==s.Rational(1,5)
# Orientation and scale derivatives, and exact phase-reference decomposition.
al=s.symbols('al',real=True)
assert s.diff(s.cos(al),al)==-s.sin(al)
assert s.diff(s.cosh(be),be)==s.sinh(be)
T12,T1q,Tq2,Tqq=s.symbols('T12 T1q Tq2 Tqq')
assert s.expand((T12-T1q)-(T12-T1q-Tq2+Tqq)-(Tq2-Tqq))==0
# Generic formal O(h) fixed-P inversion and fixed-P energy/phase conversion.
P0p,P1,E0p,E1,shift=s.symbols('P0p P1 E0p E1 shift')
assert s.expand(P0p*(-P1/P0p)+P1)==0
assert s.expand(E0p*(-P1/P0p)+E1-(E1-E0p*P1/P0p))==0
x,z=s.symbols('x z',positive=True)
A=x*z*(x-z)/(x+z)**3
assert s.simplify(A+A.xreplace({x:z,z:x}))==0
assert s.limit(A,z,0)==0
Y,r=s.symbols('Y r',positive=True)
assert s.simplify(A.subs({x:Y,z:r*r*Y})-r*r*(1-r*r)/(1+r*r)**3)==0
assert s.diff(A,x,z).subs({x:2,z:1})!=0
# Endpoint bilinear and derivative diagnostics.
f,g,j,l,t,R,V=s.symbols('f g j l t R V',nonzero=True)
M=t*a*R*(f-g);Q=(f+g)/(2*a*R)
assert s.expand(M*(j+l)/(2*a*R)+t*a*R*(j-l)*Q-t*(f*j-g*l))==0
# WH factor product only: no use of its inverse transform/asymptotics.
def wh(v):return mp.sqrt(2*mp.pi/v)*mp.gamma(1+v/(2*mp.pi))*mp.exp(-v/(2*mp.pi)*mp.log(v/(2*mp.pi*mp.e)))
whrows=[]
for w in map(mp.mpf,['.2','1','3']):
    actual=wh(mp.j*w)*wh(-mp.j*w)
    corrected=mp.pi*mp.exp(w/2)/mp.sinh(w/2)
    printed=2*mp.pi**2*mp.exp(mp.pi*w)/mp.sinh(mp.pi*w)
    assert abs(actual-corrected)<mp.mpf('1e-55')
    assert abs(actual-printed)>1
    whrows.append(dict(omega=str(w),product=str(mp.re(actual)),printed=str(printed)))
print('PASS symbolic dictionary, ambiguity, and independently checked WH product',flush=True)

class Sea:
    def __init__(self,N,q=mp.mpf('1.5')):
        self.N=N;self.h=mp.mpf('.4');self.t=2*mp.pi*self.h;self.q=q
        z,w=mp.gauss_quadrature(N,'legendre');self.x=q*z;self.w=q*w
        A=mp.matrix([[self.t*(i==j)-self.K(self.x[i]-self.x[j])*self.w[j] for j in range(N)] for i in range(N)])
        self.inv=A**-1;self.rho=self.inv*mp.ones(N,1);self.eta=self.inv*self.x
        self.n=sum(self.w[i]*self.rho[i] for i in range(N));self.a=1/self.n
        self.R=self.D(q)/self.t;self.V=self.dE(q)/self.t
    def K(self,d):return 2*mp.re(mp.digamma(1+mp.j*d/self.h))+2*mp.log(self.h)
    def Th(self,d):return 2*self.h*mp.im(mp.loggamma(1+mp.j*d/self.h))+2*d*mp.log(self.h)
    def dot(self,f,g):return sum(self.w[i]*f[i]*g[i] for i in range(self.N))
    def tv(self,p):return mp.matrix([self.Th(p-x) for x in self.x])
    def kv(self,p):return mp.matrix([self.K(p-x) for x in self.x])
    def P(self,p):return p+self.dot(self.tv(p),self.rho)
    def E(self,p):return p*p/2+self.dot(self.tv(p),self.eta)
    def D(self,p):return 1+self.dot(self.kv(p),self.rho)
    def dE(self,p):return p+self.dot(self.kv(p),self.eta)
    def ends(self,p):
        F=self.inv*self.tv(p)
        return [(self.Th(p-e)+self.dot(self.kv(e),F))/self.t for e in (self.q,-self.q)]

rows=[]
for N in (40,64):
    st=time.monotonic();m=Sea(N);err={}
    for p in (mp.mpf('1.5'),mp.mpf('2.2'),mp.mpf('-2.2')):
        f,g=m.ends(p)
        err[str(p)]=max(abs(m.dE(p)-m.P(p)+m.t*m.V*(f+g)),
                       abs(m.D(p)-1+m.t*m.R*(f-g)),
                       abs(m.E(p)-p*p/2-m.dot(m.x,m.inv*m.tv(p))),
                       abs(mp.diff(m.E,p)-m.dE(p)),abs(mp.diff(m.P,p)-m.D(p)))
    # q derivative obtained by differentiating the discrete matrix exactly.
    # At continuum level n_q=2t rho(q)^2, P_q=t rho(q)(F+ + F-).
    qp=mp.mpf('1e-10'); plus=Sea(N,m.q+qp);minus=Sea(N,m.q-qp)
    nq=(plus.n-minus.n)/(2*qp);p=mp.mpf('2.2');f,g=m.ends(p)
    pq=(plus.P(p)-minus.P(p))/(2*qp)
    err['count_q']=abs(nq-2*m.t*m.R*m.R)
    err['momentum_q']=abs(pq-m.t*m.R*(f+g))
    # derivative wrt a at fixed bare p follows from a=1/n.
    Pa=(plus.P(p)-minus.P(p))/(plus.a-minus.a)
    err['pressure_response']=abs(-m.a*Pa-(f+g)/(2*m.a*m.R))
    row=dict(N=N,errors={k:str(v) for k,v in err.items()},a=str(m.a),
             sound=str(m.V/m.R),seconds=time.monotonic()-st)
    rows.append(row); print('N',N,'max error',mp.nstr(max(err.values()),6),'seconds',row['seconds'],flush=True)
    assert max(err.values()) < (mp.mpf('1e-8') if N==40 else mp.mpf('1e-13'))
# Conditional continuum nonuniformity, verified algebraically; not a new determinant.
alpha,hh,c=s.symbols('alpha hh c',positive=True)
assert s.simplify((hh*alpha*(s.log(alpha)+c)/s.pi)/(s.Rational(2,3)*alpha**3)-3*hh*(s.log(alpha)+c)/(2*s.pi*alpha**2))==0
out=dict(precision=mp.mp.dps,rows=rows,wh_product=whrows,seconds=time.monotonic()-START,
         failures_recorded=['The printed Phase 2i Fourier symbol differs from the product of its factors at all three tested nonzero frequencies. No other Phase 2i result is used.'],
         scope='Finite-sea identities and formal chain rules; no physical quantum frame selected.')
(HERE/'frame_identities.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL CHECKS PASSED')

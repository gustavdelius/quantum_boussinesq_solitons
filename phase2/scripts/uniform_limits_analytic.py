#!/usr/bin/env python3
"""Symbolic WH reversion, kernel boundary layer, and nonuniform phase Taylor tests.
All checks are of stated algebra/conditional asymptotics, not a resolvent theorem.
"""
from pathlib import Path
import json
import mpmath as m
import sympy as s
m.mp.dps=70
HERE=Path(__file__).resolve().parent
L,alpha,z=s.symbols('L alpha z',real=True)
A=1/(2*s.pi);B=1+s.log(2*s.pi)-s.EulerGamma
c=(1-s.log(8*s.pi))/(2*s.pi);d=s.Rational(1,48)-1/(4*s.pi**2)
e=(4+6*s.zeta(3)-s.pi**2)/(96*s.pi**3)
polys=[1,A*(B-z),s.Rational(1,48)+A*A*(B-z)**2/2,-s.zeta(3)/(24*s.pi**3)+A*(B-z)/48+A**3*(B-z)**3/6]
def ilog(al,j):
    ex=(-1)**j*s.diff(s.exp((alpha-1)*L)/s.gamma(alpha),alpha,j)
    return s.simplify(s.expand_func(ex.subs(alpha,al))/s.exp((al-1)*L))
f=[]
for n in range(1,4):
    p=s.Poly(polys[n],z)
    f.append(s.simplify(3*s.sqrt(s.pi)/4*sum(co*ilog(s.Rational(5,2)-n,j[0]) for j,co in p.terms())))
a1=s.simplify(2*f[0]/3)
a2=s.simplify(2*f[1]/3-f[0]**2/9)
a3=s.simplify(2*f[2]/3-2*f[0]*f[1]/9+4*f[0]**3/81)
b0=-a1
b1=s.simplify(-s.diff(a1,L)*b0-a2)
b2=s.simplify(-s.diff(a1,L)*(b1-b0*b0/2)-s.diff(a1,L,2)*b0*b0/2-s.diff(a2,L)*b0+a2*b0-a3)
# SymPy leaves negative-half-integer polygammas unevaluated; use recurrence.
rec={s.polygamma(1,-s.Rational(1,2)):s.pi**2/2+4,s.polygamma(2,-s.Rational(1,2)):-14*s.zeta(3)+16}
for j,value in [(1,m.pi**2/2+4),(2,-14*m.zeta(3)+16)]:
    assert abs(m.polygamma(j,-m.mpf('.5'))-value)<m.mpf('1e-65')
b2=s.simplify(s.expand_log(b2.subs(rec),force=True))
assert s.simplify(b0-(-A*L+c))==0
assert s.simplify(b1-d)==0
assert s.simplify(b2-e)==0
# Check energy/velocity powers and allowed integration constant independently.
y,K=s.symbols('y K',positive=True)
u=y-A*s.log(y)+c+d/y+e/y**2
F=4*y**s.Rational(3,2)/3
J=8*y**s.Rational(5,2)/15-8*A*y**s.Rational(3,2)/9-8*d*s.sqrt(y)/3+16*e/(3*s.sqrt(y))+K
assert s.simplify(s.diff(J,y)-F*s.diff(u,y))==0
CE=3/(4*s.pi)+3*J/(4*F)-3*y/10
assert s.simplify(CE-(1/(2*s.pi)-3*d/(2*y)+9*K/(16*y**s.Rational(3,2))+3*e/y**2))==0
CV=3/(4*s.pi)+y*(s.diff(u,y)-1)/2
assert s.simplify(CV-(1/(2*s.pi)-d/(2*y)-e/y**2))==0
# Low-y: F~sqrt(pi)u -> u~4y^(3/2)/(3sqrt(pi)).
assert s.limit(3/(4*s.pi)+y*(2*s.sqrt(y)/s.sqrt(s.pi)-1)/2,y,0)==3/(4*s.pi)
assert s.simplify(3/(4*s.pi)-1/(2*s.pi))==1/(4*s.pi)
# Kernel primitive and nonuniform layer, with genuine high precision quadrature.
def R(x):return 2*(m.re(m.digamma(1+1j*x))-m.log(x))
def primitive(x):return 2*m.im(m.loggamma(1+1j*x))-2*x*m.log(x)+2*x
xx=m.mpf('1e-6')
assert abs((R(xx)+2*m.log(xx)+2*m.euler)/(xx*xx)-2*m.zeta(3))<m.mpf('1e-9')
kernel=[]
for x in map(m.mpf,['1','10','100','1000']):
    integ=m.quad(R,[0,min(1,x),x]) if x>1 else m.quad(R,[0,x])
    assert abs(integ-primitive(x))<m.mpf('1e-60')
    if x>=10: assert abs((m.pi-2*primitive(x))*x-m.mpf(1)/3)<m.mpf('.001')
    kernel.append(dict(L=str(x),mass=str(2*integ),tail_scaled=str(x*(m.pi-2*integ))))
for x in map(m.mpf,['20','100','1000']):
    assert abs(R(x)-1/(6*x*x)-1/(60*x**4))<1/(100*x**6)
logmoment=2*m.quad(lambda x:x*R(x),[1000,2000])
assert abs(logmoment-m.log(2)/3)<m.mpf('1e-7')
# Classical homogeneous phase Hessian, including the log-squared term.
p,t,r,Y,d1,d2,T,cc=s.symbols('p t r Y d1 d2 T cc',positive=True)
Phi=2*(p*p-t*t)*s.log((p-t)/(p+t))+4*p*t
g1=s.diff(Phi,p)/(2*p);g2=s.diff(Phi,t)/(2*t)
H=s.Matrix([[s.diff(g1,p)/(2*p),s.diff(g1,t)/(2*t)],[s.diff(g2,p)/(2*p),s.diff(g2,t)/(2*t)]])
Q=s.simplify((s.Matrix([d1,d2]).T*H*s.Matrix([d1,d2]))[0]/2)
assert s.simplify(Q.subs(t,r*p).subs(p*p,Y)-(d2-r*r*d1)**2/(r*(1-r*r)*Y))==0
logcoef=s.expand((d2-r*r*d1)**2/(r*(1-r*r))).subs({d1:-T/(2*s.pi)+cc,d2:-T/(2*s.pi)+cc-s.log(r)/s.pi}).expand().coeff(T,2)
assert s.simplify(logcoef-(1-r*r)/(4*s.pi**2*r))==0
# Scaling: kappa^2=hY/b and fixed P equivalently fixed kappa (monotone).
k,b,h=s.symbols('k b h',positive=True)
P=4*b*(k*s.cosh(k)-s.sinh(k))
assert s.simplify(s.diff(P,k)-4*b*k*s.sinh(k))==0
assert s.limit(P/k**3,k,0)==4*b/3
# Inherited vacuum and physical-charge normalisations.
a,hh,hB,ccp,pp=s.symbols('a hh hB ccp pp',positive=True)
bb=s.exp(-a/2);ev=2*bb/s.pi
assert s.simplify(a*s.diff(ev,a,2)/(2*bb)-a/(4*s.pi))==0
omega=2*bb*s.sin(k/2)
assert s.integrate(omega,(k,0,2*s.pi))/(4*s.pi)==2*bb/s.pi
PB=pp/(6*s.sqrt(3)*b)
assert s.simplify((hh*pp*ccp/(6*s.sqrt(3)*b*b)).subs(hh,3*b*hB)/(hB*PB)-3*ccp)==0
E=2*b*b*(s.sinh(k)*s.cosh(k)-k)
assert s.simplify(s.diff(E,k)/s.diff(P,k)-b*s.sinh(k)/k)==0
assert s.limit((E-b*P)/(b*k*k*P),k,0)==s.Rational(1,10)
# Two-term frame removal versus the unsubtracted WH convolution.
rr,YY=s.symbols('rr YY',positive=True)
assert s.limit(-2*YY*(4*rr*k*s.cosh(rr*k)-6*s.sinh(rr*k))/k,k,0)==4*rr*YY
assert s.simplify(s.limit(2*YY*(s.cosh(k)-s.cosh(rr*k))/k**2,k,0)-YY*(1-rr**2))==0
assert s.limit(4*YY*(rr*k*s.sinh(k)-k*s.sinh(rr*k))/k**2,k,0)==0
# A formal endpoint-region O(Y^-3/2 log^2 Y) scale is integrable, not O(Y^-2).
assert s.integrate(t*t,(t,0,1/s.sqrt(Y)))==1/(3*Y**s.Rational(3,2))
# Factorization at nonzero frequency.
def g(w):return m.exp(m.loggamma(1+w/(2*m.pi))-w/(2*m.pi)*m.log(w/(2*m.pi*m.e)))/m.sqrt(w)
for w in map(m.mpf,['.05','.5','2','10']):
    assert abs(2*m.pi*g(1j*w)*g(-1j*w)-2*m.pi/(-m.expm1(-w)))<m.mpf('1e-64')
out=dict(precision=70,u_coefficients=[str(b0),str(b1),str(b2)],energy=str(s.simplify(CE)),velocity=str(s.simplify(CV)),kernel=kernel,log_moment_increment=str(logmoment),log_squared_Taylor_coefficient=str(s.simplify(logcoef)),C_T=str(1/(2*m.pi)),small_y_limit=str(3/(4*m.pi)),scope='Conditional half-line algebra and direct kernel estimates; no uniform finite-sea resolvent or physical frame theorem.')
(HERE/'uniform_limits_analytic.json').write_text(json.dumps(out,indent=2)+'\n')
print('WH reversion d,e:',m.nstr(m.mpf(str(s.N(d,65))),16),m.nstr(m.mpf(str(s.N(e,65))),16))
print('kernel mass, Lipschitz logarithm, phase Hessian: checked at 70 digits')
print('ALL CHECKS PASSED')

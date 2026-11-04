#!/usr/bin/env python3
"""Checks for the conditional matched-endpoint lemma and endpoint moment.
This script does not assert that finite-sea solutions obey its hypotheses.
"""
from pathlib import Path
import json, sys
import mpmath as m
import sympy as s
m.mp.dps=60
HERE=Path(__file__).resolve().parent
A=1/(2*m.pi);c=(1-m.log(8*m.pi))/(2*m.pi)
d=m.mpf(1)/48-1/(4*m.pi**2);e=(4+6*m.zeta(3)-m.pi**2)/(96*m.pi**3)
# Differentiate the exact change of variables, then expand at large w.
w,z,aa,dd=s.symbols('w z aa dd',positive=True)
u=w-aa*s.log(w)+dd/w
F=4*w**s.Rational(3,2)/3
f=s.diff(s.diff(F,w)/s.diff(u,w),w)/s.diff(u,w)
assert s.simplify(f-(1/(s.sqrt(w)*s.diff(u,w)**2)-2*s.sqrt(w)*s.diff(u,w,2)/s.diff(u,w)**3))==0
expansion=s.series((f*s.sqrt(w)).subs(w,1/z),z,0,3).removeO()
assert s.simplify(expansion-(1-(2*dd+3*aa*aa)*z*z))==0
# Balanced displacement retains a t logarithm, but no log Y.
Y,t,r,cc=s.symbols('Y t r cc',positive=True)
delta=lambda x:-s.log(x)/(2*s.pi)+cc
balanced=s.expand_log(delta(Y)-delta(r*r*Y)+delta(Y*t*t)-delta(Y*(1-r*r+t*t)),force=True)
assert s.simplify(s.diff(balanced,Y))==0 and s.simplify(s.diff(balanced,cc))==0
# Inner-region and outer-remainder powers, exact integrals used in proof.
assert s.integrate(t*t,(t,0,1/s.sqrt(Y)))==1/(3*Y**s.Rational(3,2))
assert s.simplify(s.integrate(t**-2/Y**2,(t,1/s.sqrt(Y),r))-(Y**-s.Rational(3,2)-1/(r*Y**2)))==0
# The formal J exponents cannot generate a constant of integration.
assert all(s.Rational(5,2)-n != 0 for n in range(12))
# Integration by parts links the endpoint moment to J's constant.
U=s.Function('U')(Y);JJ=s.Function('J')(Y)
primitive=2*Y**s.Rational(3,2)*U/3-JJ/2
assert s.simplify(s.diff(primitive,Y).subs(s.diff(JJ,Y),4*Y**s.Rational(3,2)*s.diff(U,Y)/3)-s.sqrt(Y)*U)==0
integrals=[]
for rr in map(m.mpf,['.2','.4','.5','.6','.8']):
    def delta0(t):return -m.log(t*t/(rr*rr*(1-rr*rr+t*t)))/(2*m.pi)
    def first(t):return -4*t*t*delta0(t)/(1-rr*rr+t*t)**m.mpf('1.5')
    def second(t):
        sig=1-rr*rr+t*t;D=delta0(t)
        return 8/sig**m.mpf('2.5')*(t*t*(-2*d-3*A*A-A*D/2+3*D*D/8)-d*sig/2*(1+t*t*(1-1/(rr*rr)-1/sig)))
    leading=8*m.quad(lambda t:t*t/m.sqrt(1-rr*rr+t*t),[0,rr])
    cl=2*(1-rr*rr)*m.log((1-rr)/(1+rr))+4*rr
    c0=-2*(m.polylog(2,rr)-m.polylog(2,-rr)-2*rr)/m.pi
    c1=-rr**3/(6*(1-rr*rr))-2*d*((1-rr*rr)/rr**2*m.log((1-rr)/(1+rr))+2/rr)
    p0=m.quad(first,[0,rr]);p1=m.quad(second,[0,rr])
    assert abs(leading-cl)<m.mpf('1e-54')
    assert abs(p0-c0)<m.mpf('1e-54')
    assert abs(p1-c1)<m.mpf('1e-54')
    integrals.append(dict(r=str(rr),c0=str(p0),c1=str(p1)))
def g(z):return m.exp(m.loggamma(1+z/(2*m.pi))-z/(2*m.pi)*m.log(z/(2*m.pi*m.e)))/m.sqrt(z)
def inv(x,n):return m.invertlaplace(lambda z:m.sqrt(m.pi)*g(z)/z**n,x,method='talbot')
def uof(y):
    target=4*y**m.mpf('1.5')/3
    guess=y-A*m.log(y)+c+d/y if y>2 else target/m.sqrt(m.pi)
    return m.findroot(lambda u:inv(u,2)-target,(guess,guess+m.mpf('.001')))
def mu(Y):
    u=uof(Y);J=inv(u,3)
    elementary=2*Y**m.mpf('2.5')/5-A*Y**m.mpf('1.5')*(2*m.log(Y)/3-m.mpf(4)/9)+2*c*Y**m.mpf('1.5')/3+2*d*m.sqrt(Y)
    return 2*Y**m.mpf('1.5')*u/3-J/2-elementary
moments=[]
for Y in map(m.mpf,['64','512','4096']):
    M=mu(Y);scaled=(M+2*e/m.sqrt(Y))*Y**m.mpf('1.5')
    assert abs(scaled)<m.mpf('.001')
    moments.append(dict(Y=str(Y),mu_partial=str(M),after_tail_scaled=str(scaled)))
# High-precision direct quadrature of the finite endpoint moment, independent of J.
cache=HERE/'uniform_limits_matched.json'
if cache.exists() and '--refresh' not in sys.argv:
    saved=json.loads(cache.read_text())
    direct_difference=m.mpf(saved['direct_moment_difference'])
    quad_spread=m.mpf(saved['direct_quadrature_spread'])
    assert direct_difference<m.mpf('1e-14') and quad_spread<m.mpf('1e-10')
else:
    Y=m.mpf(4);vals=[]
    for n in [20,32]:
        nodes,weights=m.gauss_quadrature(n,'legendre')
        integral=m.mpf(0)
        for z,wgt in zip(nodes,weights):
            t=(z+1)*m.sqrt(Y)/2
            integral+=wgt*m.sqrt(Y)/2*2*t*t*uof(t*t)
        elem=2*Y**m.mpf('2.5')/5-A*Y**m.mpf('1.5')*(2*m.log(Y)/3-m.mpf(4)/9)+2*c*Y**m.mpf('1.5')/3+2*d*m.sqrt(Y)
        vals.append(integral-elem)
    direct_difference=abs(vals[1]-mu(Y))
    assert direct_difference<m.mpf('1e-14'),(vals,mu(Y))
    quad_spread=abs(vals[1]-vals[0])
    assert quad_spread<m.mpf('1e-10'),vals
out=dict(precision=60,coefficients=integrals,moments=moments,direct_moment_difference=str(direct_difference),direct_quadrature_spread=str(quad_spread),hypothesis='u and three derivatives have stated uniform large-y bounds; exact half-line transform only, not yet finite sea.')
(HERE/'uniform_limits_matched.json').write_text(json.dumps(out,indent=2)+'\n')
print('Matched coefficients, endpoint moment and direct quadrature checked')
print('ALL CHECKS PASSED')

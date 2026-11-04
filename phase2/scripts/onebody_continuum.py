#!/usr/bin/env python3
"""Independent conditional continuum checks; no existing module is imported.

The real stable-band trace and the formal continuation are different objects.
Neither is asserted to be a completed projected quantum determinant.
"""
from pathlib import Path
import json
import time
import sympy as s
import mpmath as m

start = time.monotonic()
m.mp.dps = 65
b, k, v, c = s.symbols('beta k v c', positive=True)
ut, ux, uxx = s.symbols('phi_t phi_x phi_xx')
# sigma=c^2, phi_sigma=phi_good/c, (x,t)_sigma=c*(x,t)_good.
Ls = (ut/c**2)**2/2 - (ux/c**2)**2/2 + c**2*(ux/c**2)**3 - c**2*(uxx/c**3)**2/2
Lg = ut**2/2-ux**2/2+ux**3-uxx**2/2
assert s.simplify(c**4*Ls-Lg) == 0
# S_sigma=S_good/c^2; h_good=c^2*h_sigma; E1_sigma=E1_good/c.
P = s.Rational(2,3)*s.sinh(b)**3*s.cosh(b)
E = s.Rational(2,3)*s.sinh(b)**3+s.Rational(8,15)*s.sinh(b)**5
assert s.simplify(s.diff(E,b)-s.cosh(b)*s.diff(P,b)) == 0
assert s.simplify(s.cosh(b)*P-E-s.Rational(2,15)*s.sinh(b)**5) == 0
z = s.symbols('z', real=True)
u = k**2*(1-z*z)/2
dx = lambda f: s.diff(f,z)*k*(1-z*z)/2
assert s.simplify(dx(dx(u))-k*k*u+3*u*u) == 0
measure=2/(k*(1-z*z))
assert s.integrate(u*measure,(z,-1,1))==2*k
assert s.integrate(u*u*measure,(z,-1,1))==2*k**3/3
Hb=((2+k*k)*u*u/2+u**3-dx(u)**2/2)*measure
assert s.simplify(s.integrate(Hb,(z,-1,1))-2*k**3/3-8*k**5/15)==0
ee,pp,delta,L=s.symbols('e p delta L')
assert s.expand((ee+v*delta)-v*(pp+delta)-(ee-v*pp))==0
assert s.expand((v*pp-L)-v*pp+L)==0
F = s.sqrt(3)/s.pi*(k*k+s.Rational(3,4))*s.atan(2*k/s.sqrt(3))-3*k/(2*s.pi)
Fb = F.subs(k,s.sinh(b))
R = -s.sqrt(3)/2*(s.sinh(b)**2+s.Rational(3,4))
assert s.simplify(s.diff(F,k)-2*s.sqrt(3)*k*s.atan(2*k/s.sqrt(3))/s.pi) == 0
assert s.simplify(-s.diff(Fb,b)/s.sinh(b)+2*s.sqrt(3)*s.cosh(b)*s.atan(2*s.sinh(b)/s.sqrt(3))/s.pi)==0
assert s.simplify(s.diff(R.subs(b,s.acosh(v)),v,2)+s.sqrt(3)) == 0
series = s.series(Fb,b,0,7)
assert s.simplify(series.removeO()-4*b**3/(3*s.pi)-14*b**5/(45*s.pi)) == 0
# Origin/chemical-potential invariant: annihilates A+B*v+C*M exactly.
ann = lambda f: s.diff(f,b,3)-s.diff(f,b)
for f in (s.Integer(1),s.cosh(b),2*s.sinh(b)):
    assert s.simplify(ann(f)) == 0
assert s.limit(ann(P),b,0) == 4
assert s.limit(ann(Fb),b,0)/4 == 2/s.pi
assert s.limit(ann(R),b,0) == 0
# Analytic reduction of the stable-band integral: z=cos(t), q=sin(t).
Z=1-2*k*k-2*z*z; B=-3-2*k*k+2*z*z
Dp=9*k*k*(1-z*z)+(Z+v*z)**2
Dm=9*k*k*(1-z*z)+(Z-v*z)**2
num,den=s.fraction(s.together((z-v)*(v+z*B)/Dp+(z+v)*(-v+z*B)/Dm))
num=s.rem(num,v*v-1-k*k,v);den=s.rem(den,v*v-1-k*k,v)
rational=2*(-4*k*k+8*z**4-10*z*z-1)/(16*k**4+16*k*k*z*z+8*k*k+16*z**4-8*z*z+1)
assert s.factor(num/den-rational)==0
A=z/(s.sqrt(3)*k)
Bangle=s.sqrt(3)*z*(4*z*z+8*k*k-1)/(3*k*(4*k*k+1))
primitive=z-s.sqrt(3)*(4*k*k+3)/(12*k)*(s.atan(A)+s.atan(Bangle))
assert s.simplify(s.diff(primitive,z)-rational)==0
# At z=1 both angles are positive and their sum is in (0,pi).
# Its tangent equals that of pi-2 atan(2k/sqrt(3)), also in (0,pi).
A1=A.subs(z,1);B1=Bangle.subs(z,1);t=2*k/s.sqrt(3)
assert s.simplify(s.factor((A1+B1)/(1-A1*B1)+2*t/(1-t*t)))==0
assert primitive.subs(z,0)==0
# Wick sign: bad L=-u^3, bad H=+u^3, :u^3:=u^3-3Cu.
C,U,noise=s.symbols('C U noise')
pol=s.Poly(s.expand((U+noise)**3-3*C*(U+noise)),noise)
assert s.expand(sum(coef*{0:1,1:0,2:C,3:0}[n[0]] for n,coef in pol.terms())-U**3)==0

def formal(beta):
    kk=m.sinh(beta)
    return m.sqrt(3)/m.pi*(kk*kk+m.mpf(3)/4)*m.atan(2*kk/m.sqrt(3))-3*kk/(2*m.pi)
def good(a):
    # Holomorphic branches agreeing with the real light-branch det formula.
    roots=(a+2*m.pi/3,a-2*m.pi/3)
    logs=(m.log(m.tan(roots[0]/2)),m.log(-m.tan(roots[1]/2)))
    L=(3*m.sin(a)+sum(m.sin(f)*(m.cos(f)-m.cos(a))*l for f,l in zip(roots,logs)))/(2*m.pi)
    return -L
def transmission(kk,vv,q):
    w=abs(q)*m.sqrt(1-q*q)
    poly=lambda K,W:W*W-K*K-K**4
    return -poly(kk-1j*q,kk*vv-1j*w)/poly(kk+1j*q,kk*vv+1j*w)

rows=[]; maxres=m.mpf(0)
for bs in ('0.01','0.05','0.2','0.7'):
    bb=m.mpf(bs);kk=m.sinh(bb);vv=m.cosh(bb)
    assert abs(-1j*good(1j*bb)-formal(bb))<m.mpf('1e-59')
    assert abs(m.sin(1j*bb)-1j*kk)<m.mpf('1e-60')
    assert abs(-m.cos(3j*bb)+m.cosh(3*bb))<m.mpf('1e-60')
    for eps in (1,-1):
        velocity=eps*vv
        def theta_t(t):
            qq=m.sin(t);cc=m.cos(t);sg=m.sign(t)
            X=3*kk*qq;Z=velocity*sg*cc-1-2*kk*kk+2*qq*qq
            return 2*(Z*3*kk*cc-X*(-velocity*sg*qq+4*qq*cc))/(Z*Z+X*X)
        for ts in ('-0.7','0.4'):
            tt=m.mpf(ts);qq=m.sin(tt)
            X=3*kk*qq;Z=velocity*m.sign(qq)*m.sqrt(1-qq*qq)-1-2*kk*kk+2*qq*qq
            A=transmission(kk,velocity,qq)
            assert abs(A-(Z+1j*X)/(Z-1j*X))<m.mpf('1e-58')
            assert abs(abs(A)-1)<m.mpf('1e-58')
            assert abs(m.im(m.diff(lambda t:m.log(transmission(kk,velocity,m.sin(t))),tt))-theta_t(tt))<m.mpf('1e-56')
        # Resolve both possible resonance locations and the narrow q=0 region.
        bounds=sorted(set([-m.pi/2,-m.pi/3,-bb,0,bb,m.pi/3,m.pi/2]+
                          [x for center in (-m.pi/3,m.pi/3) for x in (center-bb,center+bb) if abs(x)<m.pi/2]))
        integral=m.quad(lambda t:(abs(m.sin(t))*m.cos(t)-velocity*m.sin(t))*theta_t(t),bounds)/(4*m.pi)
        normal=-3*kk/m.pi
        exact=formal(bb)-m.sqrt(3)/2*(kk*kk+m.mpf(3)/4)
        err=abs(integral+normal-exact); maxres=max(maxres,err)
        assert err<m.mpf('1e-52'), (bs,eps,err)
        pos=[x for x in bounds if x>=0];neg=[x for x in bounds if x<=0]
        thplus=m.quad(theta_t,pos);thminus=-m.quad(theta_t,neg)
        assert abs(thplus+thminus)<m.mpf('1e-52')
        rows.append(dict(beta=bs,epsilon=eps,formal=str(formal(bb)),stable_band=str(exact),quadrature_error=str(err),endpoint_phase_sum=str(thplus+thminus)))
        # Independent potential pencil residual, using the Hirota ratio above.
        qq=m.mpf('0.43');om=qq*m.sqrt(1-qq*qq);nu=om-velocity*qq
        AA=transmission(kk,velocity,qq)
        def mode(x):
            return m.diff(lambda z:m.exp(1j*qq*z)*(1+AA*m.exp(kk*z))/(1+m.exp(kk*z)),x)
        def pulse(x):return kk*kk/(2*m.cosh(kk*x/2)**2)
        xx=m.mpf('.31')
        residual=-m.diff(mode,xx,4)+(velocity**2-1)*m.diff(mode,xx,2)-6*m.diff(lambda z:pulse(z)*m.diff(mode,z),xx)+2j*velocity*nu*m.diff(mode,xx)-nu*nu*mode(xx)
        assert abs(residual)<m.mpf('1e-54'), residual

# Stable vacuum contraction via q=sin(t), and an unstable vacuum sample.
assert abs(m.quad(lambda t:m.sin(t),[0,m.pi/2])/(2*m.pi)-1/(2*m.pi))<m.mpf('1e-60')
unstable=abs(transmission(m.sinh(m.mpf('.7')),m.cosh(m.mpf('.7')),m.mpf('1.3')))
assert abs(unstable-1)>m.mpf('.1')
for aa in map(m.mpf,['.15','.5','.85']):
    kg=m.sin(aa)
    fg=m.sqrt(3)/m.pi*(m.mpf(3)/4-kg*kg)*m.atanh(2*kg/m.sqrt(3))-3*kg/(2*m.pi)
    assert abs(good(aa)-fg)<m.mpf('1e-59')
qs=-m.sqrt(3)/2
om=lambda q:abs(q)*m.sqrt(1-q*q)
assert abs(om(qs)-m.sqrt(3)/4)<m.mpf('1e-60')
assert abs(m.diff(om,qs)-1)<m.mpf('1e-60')
assert abs(-(om(qs)-qs)/2+3*m.sqrt(3)/8)<m.mpf('1e-60')
tiny=m.mpf('1e-12');p0=lambda x:2*m.sinh(x)**3*m.cosh(x)/3
assert abs(formal(tiny)/p0(tiny)-2/m.pi)<m.mpf('1e-22')
assert abs(good(tiny)/(2*m.sin(tiny)**3*m.cos(tiny)/3)+2/m.pi)<m.mpf('1e-22')
out=dict(precision=m.mp.dps,rows=rows,max_quadrature_error=str(maxres),bad_formal_ratio=str(2/m.pi),good_ratio=str(-2/m.pi),unstable_transmission_modulus=str(unstable),stable_band_series=str(s.series(Fb+R,b,0,6)),formal_series=str(series),runtime_seconds=time.monotonic()-start,
         failures_recorded=['Scratch derivation initially inverted the Hirota ratio; corrected to -F(difference)/F(sum), then checked against the potential fluctuation pencil.',
         'Inherited bad-equation note used +3k/pi for the normal-ordering energy; bad Hamiltonian Wick ordering gives -3k/pi.',
         'A symbolic endpoint identity initially failed structural equality because factor returned unevaluated 0*sqrt(3); exact simplify reduced it to zero, without changing the formula or any tolerance.',
         'The full bad-equation real-frequency trace does not exist for |q|>1; continuation and stable-band reduction require extra assumptions.'])
Path(__file__).with_name('onebody_continuum_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
print('ALL CHECKS PASSED')

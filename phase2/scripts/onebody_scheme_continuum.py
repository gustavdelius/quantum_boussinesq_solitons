#!/usr/bin/env python3
"""Independent continuum and lattice vacuum matching checks (coefficients of hbar).

Only onebody_scheme_* output is written. Analytic continuation is a specified
prescription, not a construction of a stable bad-Boussinesq quantum theory.
"""
from pathlib import Path
import json
import sympy as s
import mpmath as m

m.mp.dps = 70
U, L, k, z, a, h, alpha = s.symbols('U L k z a h alpha', positive=True)
pi = s.pi

# Homogeneous good vacuum, at fixed coordinate/mode cutoff. Subtract vacuum
# and its first derivative (the specified normal-ordering tadpole).
bare = ((L**2+1-6*U)**s.Rational(3,2)-(1-6*U)**s.Rational(3,2))/(6*pi)
good = -((1-6*U)**s.Rational(3,2)-1+9*U)/(6*pi)
assert s.simplify(s.limit(bare-bare.subs(U,0)-U*s.diff(bare,U).subs(U,0),L,s.oo)-good)==0
assert s.simplify(s.diff(bare,U).subs(U,0)+3*(s.sqrt(1+L**2)-1)/(2*pi))==0
assert s.simplify(s.diff(bare,U,2).subs(U,0)-9*(1/s.sqrt(1+L**2)-1)/(2*pi))==0
# The spatial continuation consistent with F_b(k)=-i F_g(ik) gives W_b=-W_g(-U).
bad = -good.subs(U,-U)
assert s.simplify(bad-((1+6*U)**s.Rational(3,2)-1-9*U)/(6*pi))==0
sg = s.diff(good,U,2).subs(U,0)/2
sb = s.diff(bad,U,2).subs(U,0)/2
assert sg == -9/(4*pi) and sb == 9/(4*pi)

# Independent quadrature of the convergent good vacuum bubble/elastic modulus.
bubble = -9/(2*m.pi)*m.quad(lambda q:q/(1+q*q)**m.mpf('1.5'),[0,1,m.inf])
assert abs(bubble+9/(2*m.pi))<m.mpf('1e-65')
vacuum_rows=[]
for uu in (m.mpf('0.01'),m.mpf('-0.02')):
    # The subtractions are made inside the integral. A finite endpoint avoids
    # catastrophic cancellation at mpmath's arbitrarily large infinite nodes.
    cutoff=m.mpf('100')
    cc=1-6*uu
    integral=m.quad(lambda q:q*(m.sqrt(q*q+cc)-m.sqrt(q*q+1)+3*uu/m.sqrt(q*q+1)),[0,1,10,cutoff])/(2*m.pi)
    exact=s.lambdify((U,L),bare-bare.subs(U,0)-U*s.diff(bare,U).subs(U,0),'mpmath')(uu,cutoff)
    assert abs(integral-exact)<m.mpf('1e-62')
    vacuum_rows.append({'U':str(uu),'quadrature_error':str(abs(integral-exact))})

# Lattice harmonic mode sum at strain r-a=-6U, using reference b and coordinates.
b=s.exp(-a/2)
p=s.symbols('p',positive=True)
e1T=s.integrate(2*b*s.sin(p/2),(p,0,2*pi))/(4*pi)
assert s.simplify(e1T-2*b/pi)==0
assert s.simplify(a*s.diff(e1T,a,2)/(2*b)-a/(4*pi))==0
# E_B density = E_T per particle/(36 b^2), h_T=3 b h_B.
lat0=s.simplify(e1T*s.exp(3*U)/(12*b))
lat=s.simplify(lat0-lat0.subs(U,0)-U*s.diff(lat0,U).subs(U,0))
st=s.diff(lat,U,2).subs(U,0)/2
assert st==3/(4*pi)
matching=s.simplify(lat-bad)
lam=s.expand(s.series(matching,U,0,3).removeO()).coeff(U,2)
assert lam == -3/(2*pi)
# Exact exponential versus cubic Hamiltonian, including omitted quartic term.
Vlat=(s.exp(6*U)-1-6*U)/36
assert s.series(Vlat,U,0,5).removeO()==U**2/2+U**3+s.Rational(3,2)*U**4

# Soliton integrals by z=tanh(k*x/2): dx=2 dz/[k(1-z^2)].
moments={n:s.integrate((k*k*(1-z*z)/2)**n*2/(k*(1-z*z)),(z,-1,1)) for n in (1,2,3)}
assert moments[1]==2*k and moments[2]==2*k**3/3 and moments[3]==4*k**5/15
Fg=s.sqrt(3)/pi*(s.Rational(3,4)-k*k)*s.atanh(2*k/s.sqrt(3))-3*k/(2*pi)
Fb=s.sqrt(3)/pi*(s.Rational(3,4)+k*k)*s.atan(2*k/s.sqrt(3))-3*k/(2*pi)
assert s.simplify(-s.I*Fg.subs(k,s.I*k)-Fb)==0
Pg=2*k**3*s.sqrt(1-k*k)/3
Pb=2*k**3*s.sqrt(1+k*k)/3
Cg=s.limit(Fg/Pg,k,0); Cb=s.limit(Fb/Pb,k,0)
Cm=s.limit((Fb+lam*moments[2])/Pb,k,0)
assert Cg==-2/pi and Cb==2/pi and Cm==1/(2*pi)
assert s.simplify(Cg-sg)==1/(4*pi)
assert s.simplify(Cb-sb)==-1/(4*pi)
assert s.simplify(Cm-st)==-1/(4*pi)
for momentum,energy,velocity in ((Pg,2*k**3/3-8*k**5/15,s.sqrt(1-k*k)),(Pb,2*k**3/3+8*k**5/15,s.sqrt(1+k*k))):
    assert s.simplify(s.diff(energy,k)/s.diff(momentum,k)-velocity)==0
# A finite tadpole changes only M and destroys the finite ratio unless zero.
assert s.limit(k*k*moments[1]/Pb,k,0)==3
# Check the differential origin/mass annihilator does NOT remove this new u^2 term.
beta=s.symbols('beta',positive=True)
ann=lambda f:s.diff(f,beta,3)-s.diff(f,beta)
assert s.limit(ann((lam*moments[2]).subs(k,s.sinh(beta)))/ann(Pb.subs(k,s.sinh(beta))),beta,0)==lam

# What the light-root cancellation actually says: the logarithmic weight only.
phi=s.symbols('phi',real=True)
w=2*s.sin(phi)*(s.cos(alpha)-s.cos(phi))
assert s.trigsimp(w.subs(phi,alpha))==0
for sign in (-1,1):
    assert s.trigsimp(s.expand_trig(w.subs(phi,alpha+sign*2*pi/3))-sign*2*s.sqrt(3)*(s.Rational(3,4)-s.sin(alpha)**2))==0
assert -2*2*s.sin(alpha)/(4*pi)==-s.sin(alpha)/pi
# The pole location is NOT a zero co-moving frequency. At negative q the
# group velocity has the opposite sign to the positive-q formula.
qstar=-s.sqrt(3)/2
omega=-p*s.sqrt(1-p*p)
assert omega.subs(p,qstar)==s.sqrt(3)/4
assert s.diff(omega,p).subs(p,qstar)==1
assert (omega-p).subs(p,qstar)==3*s.sqrt(3)/4
assert s.diff(p*s.sqrt(1-p*p),p).subs(p,1/s.sqrt(2))==0

# Independent real stable-band quadratures via the rational z=cos(t) integral.
band_rows=[]
for kk in map(m.mpf,['0.03','0.2','0.5']):
    def R(zz):
        return 2*(-4*kk*kk+8*zz**4-10*zz*zz-1)/(16*kk**4+16*kk*kk*zz*zz+8*kk*kk+16*zz**4-8*zz*zz+1)
    # Derive the rational integrand independently from the Hirota polynomial
    # transmission and numerical logarithmic differentiation, point by point.
    vv=m.sqrt(1+kk*kk)
    def A(tt):
        qq=m.sin(tt);om=abs(qq)*m.cos(tt)
        poly=lambda K,W:W*W-K*K-K**4
        return -poly(kk-1j*qq,kk*vv-1j*om)/poly(kk+1j*qq,kk*vv+1j*om)
    for zz in map(m.mpf,['0.2','0.45','0.7','0.9']):
        tt=m.acos(zz);combined=m.mpf(0)
        for t0 in (-tt,tt):
            qq=m.sin(t0);nu=abs(qq)*m.cos(t0)-vv*qq
            combined+=nu*m.im(m.diff(A,t0)/A(t0))
        assert abs(combined/m.sqrt(1-zz*zz)-6*kk*R(zz))<m.mpf('1e-60')
    val=3*kk/(2*m.pi)*m.quad(R,[0,m.mpf('.5')-kk/2,m.mpf('.5'),m.mpf('.5')+kk/2,1])-3*kk/m.pi
    closed=m.sqrt(3)/m.pi*(kk*kk+m.mpf('.75'))*m.atan(2*kk/m.sqrt(3))-3*kk/(2*m.pi)-m.sqrt(3)/2*(kk*kk+m.mpf('.75'))
    assert abs(val-closed)<m.mpf('1e-60')
    band_rows.append({'k':str(kk),'error':str(abs(val-closed))})
band=Fb-s.sqrt(3)/2*(k*k+s.Rational(3,4))
assert s.limit(k**3*band/Pb,k,0)==-9*s.sqrt(3)/16

# Finite-cutoff endpoint: no finite k^3 freedom in its standard large-L tail.
# theta(L)-theta(-L)=-2 sum c_r atan(sin(phi_r)/L).
S1=-3*s.sin(alpha)
S3=sum(c*s.sin(alpha+shift)**3 for c,shift in [(1,2*pi/3),(1,-2*pi/3),(-2,0)])
assert s.trigsimp(s.expand_trig(S3)+9*s.sin(alpha)/4)==0
# B=-[L^2+1/2+O(L^-2)] [S1/L-S3/(3L^3)+...]/(2pi).
# Therefore B=3k L/(2pi)+O(L^-1), with no L^0 term.

out={'good_sound':str(sg),'continued_bad_sound':str(sb),'Toda_sound':str(st),
     'matching_H_density_per_hbar':str(lam*U**2),'matched_C':str(Cm),
     'good_sound_subtracted_C':str(s.simplify(Cg-sg)),
     'bad_sound_subtracted_C':str(s.simplify(Cb-sb)),
     'vacuum_quadratures':vacuum_rows,'stable_band_quadratures':band_rows,
     'limitations':['The bad vacuum determinant is formally continued, not a real stable-band determinant.',
       'Only the leading small-amplitude coefficient is matched; no full lattice soliton determinant is computed.',
       'No physical scheme is selected for the PI.']}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
print('ALL CHECKS PASSED')

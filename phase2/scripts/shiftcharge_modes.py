#!/usr/bin/env python3
"""(c) Conditional continuum tadpole calculation, independent of E'(P).

The linear shift charge has no Gaussian contraction. Its mean-profile term
is evaluated using the adjoint static Green equation; the direct momentum
insertion includes both the mean profile and Gaussian contraction. This is
not a construction of the missing boundary-compatible functional measure.
"""
import json
import time
from pathlib import Path
import sympy as s
import mpmath as mp

start=time.monotonic(); mp.mp.dps=80
k,y,q,a,b,v,R=s.symbols('k y q a b v R',real=True)
Dx=lambda f:k*(1-y*y)*s.diff(f,y)/2
u=k*k*(1-y*y)/2
uk=k*(1-y*y)*(1-y*s.atanh(y))
H=lambda f:-Dx(Dx(f))+(k*k-6*u)*f
w=1/k**2-3*uk/k**3
assert s.simplify(H(w)-1)==0
assert s.simplify(H(uk)+2*k*u)==0
assert s.simplify(H(Dx(u)))==0
assert s.simplify(H(Dx(Dx(u)))-6*Dx(u)**2)==0
AA,BB,ll=s.symbols('AA BB ll')
mean=(-3*AA/k**3+ll/k)*uk+2*BB/k**6*Dx(Dx(u))
assert s.simplify(H(mean)-3*(AA*(1-y*y)+BB*y*y*(1-y*y)**2)+2*ll*u)==0
measure=2/(k*(1-y*y))
moments={}
for name,expr,want in [
    ('w_s2',w*(1-y*y)*measure,-2/k**3),
    ('w_B',w*y*y*(1-y*y)**2*measure,0),
    ('uk_s2',uk*(1-y*y)*measure,2),
    ('uk_B',uk*y*y*(1-y*y)**2*measure,s.Rational(8,45))]:
    got=s.integrate(expr,(y,-1,1))
    assert s.simplify(got-want)==0,(name,got,want)
    moments[name]=str(want)
# Direct potential-field mode density, not a phase-density substitution.
hh=(1+y)/2; hp=k*(1-y*y)/4; hpp=-k*k*y*(1-y*y)/4
D=a+s.I*b; RR=1+D*hh
f=RR+D*hp/(s.I*q)
fx=s.I*q*RR+2*D*hp+D*hpp/(s.I*q)
def unit_reduce(expr):
    return s.factor(s.expand(s.expand_complex(expr)).subs(b*b,-2*a-a*a))
g=unit_reduce(fx*s.conjugate(fx)-q*q)
j=unit_reduce(s.re(s.I*f*s.conjugate(fx))-q)
Ac=a*(q*q-k*k)/2+q*b*k
Bc=-a*k**4/(8*q*q)
assert s.simplify(g-Ac*(1-y*y)-Bc*y*y*(1-y*y)**2)==0
J=-a*k/q+2*a*q/k+3*b
G=-a*k**3/(15*q*q)-2*a*k+2*a*q*q/k+4*b*q
assert s.simplify(s.integrate(j*measure,(y,-1,1))-J)==0
assert s.simplify(s.integrate(g*measure,(y,-1,1))-G)==0
assert s.simplify(G-3/k*(2*Ac+s.Rational(8,45)*Bc)+2*Ac/k)==0
# Factorization makes the integrated tadpole elementary and avoids UV loss.
for sign in [1,-1]:
    den=9*(1-v*v)*(R*R-1)+(2*R*R-1-2*(1-v*v)-sign*v*R)**2
    assert s.expand(den-(R-sign*v)**2*(4*R*R+4*sign*R*v+4*v*v-3))==0
# Primitive for the rational R integral used for A.
d=s.symbols('d',positive=True)
primitive=s.log((2*R+v-d)/(2*R+v+d))/(4*d)
assert s.simplify(s.diff(primitive,R)-1/((2*R+v)**2-d*d))==0
qp=s.symbols('qp',positive=True)
for direction in [1,-1]:
    tq=3*k*direction*qp
    bq=1-2*k*k+2*qp*qp-v*direction*s.sqrt(1+qp*qp)
    aq=-2*tq*tq/(tq*tq+bq*bq)
    soft=-aq*k**4/(8*qp**2)/(4*s.pi*qp*s.sqrt(1+qp*qp))
    assert s.simplify(s.limit(qp*soft,qp,0)-9*k**6/(16*s.pi*(1-2*k*k-direction*v)**2))==0

def spectral(K,eps):
    V=eps*mp.sqrt(1-K*K)
    def terms(Q):
        rad=mp.sqrt(1+Q*Q); om=abs(Q)*rad; nu=om-V*Q
        tt=3*K*Q; bb=1-2*K*K+2*Q*Q-V*mp.sign(Q)*rad
        den=tt*tt+bb*bb
        re=-2*tt*tt/den; im=2*tt*bb/den
        ac=re*(Q*Q-K*K)/2+Q*im*K
        jj=-re*K/Q+2*re*Q/K+3*im
        return ac/(4*mp.pi*om),(nu*jj-2*V*ac/K)/(4*mp.pi*om)
    AA=mp.quad(lambda Q:terms(Q)[0]+terms(-Q)[0],[0,K,1,mp.inf])
    pp=mp.quad(lambda Q:terms(Q)[1]+terms(-Q)[1],[0,K,1,mp.inf])
    return AA,pp

rows=[]
mode_residuals=[]
for K,V,Q in [(mp.mpf('.6'),mp.mpf('.8'),mp.mpf('.7')),
              (mp.mpf('.6'),mp.mpf('.8'),mp.mpf('-.7')),
              (mp.mpf('.6'),mp.mpf('-.8'),mp.mpf('.7'))]:
    om=abs(Q)*mp.sqrt(1+Q*Q); nu=om-V*Q
    tt=3*K*Q; bb=1-2*K*K+2*Q*Q-V*mp.sign(Q)*mp.sqrt(1+Q*Q)
    transmission=-(tt-1j*bb)/(tt+1j*bb)
    assert abs(abs(transmission)-1)<mp.mpf('1e-75')
    def potential_mode(x):
        yy=mp.tanh(K*x/2); hh=(1+yy)/2; hp=K*(1-yy*yy)/4
        return mp.exp(1j*Q*x)*(1+(transmission-1)*(hh+hp/(1j*Q)))
    def profile(x):
        return K*K/(2*mp.cosh(K*x/2)**2)
    X=mp.mpf('.37')
    residual=(mp.diff(potential_mode,X,4)-K*K*mp.diff(potential_mode,X,2)
              +6*mp.diff(lambda z:profile(z)*mp.diff(potential_mode,z),X)
              +2j*V*nu*mp.diff(potential_mode,X)-nu*nu*potential_mode(X))
    assert abs(residual)<mp.mpf('1e-65')
    mode_residuals.append(str(abs(residual)))
for ks,eps in [('.12',1),('.31',1),('.6',1),('.6',-1),('.8',1)]:
    K=mp.mpf(ks); V=eps*mp.sqrt(1-K*K)
    AA,pp=spectral(K,eps)
    T=mp.atanh(2*K/mp.sqrt(3))
    ac=mp.sqrt(3)*K*T/(4*mp.pi)
    pc=-2*mp.sqrt(3)*V*T/mp.pi
    assert abs(AA-ac)<mp.mpf('1e-65')
    assert abs(pp-pc)<mp.mpf('1e-65')
    # Independent check against differentiated audited energy, only AFTER
    # the direct field insertions above have been evaluated.
    F=lambda t:mp.sqrt(3)/mp.pi*(mp.mpf('.75')-t*t)*mp.atanh(2*t/mp.sqrt(3))-3*t/(2*mp.pi)
    pc_from_energy=V/K*mp.diff(F,K)
    assert abs(pp-pc_from_energy)<mp.mpf('1e-65')
    # Direct x integration of the solved, infrared-regulated mean profile.
    # Two different finite B values model different soft translation variances.
    for softB in [mp.mpf('0.2'),mp.mpf('2.7')]:
        def profile_mean(z):
            yy=mp.tanh(K*z/2); ss=1-yy*yy
            ukx=K*ss*(1-K*z*yy/2)
            uxx=K**4*ss*(3*yy*yy-1)/4
            return -3*AA/K**3*ukx+2*softB/K**6*uxx
        integrated=mp.quad(profile_mean,[-mp.inf,-10/K,0,10/K,mp.inf])
        assert abs(integrated+6*AA/K**3)<mp.mpf('1e-65')
    for lv in [mp.mpf(0),9/(4*mp.pi)]:
        mv=-6*AA/K**3+2*lv/K
        pv=pp+2*lv*V*K
        P_v=mp.mpf(2)/3*K*(1-4*V*V)
        M_v=-2*V/K; Pi_v=2*(2*V*V-1)/K
        velocity=-pv/P_v
        mass=mv-M_v*pv/P_v
        shift=-V*mv-Pi_v*pv/P_v
        assert abs(shift+V*mass+2*K*velocity)<mp.mpf('1e-65')
        rows.append(dict(k=ks,epsilon=eps,lambda_value=str(lv),A=str(AA),
                         m_fixed_velocity=str(mv),p_direct=str(pv),
                         m_fixed_P=str(mass),Pi_fixed_P=str(shift),velocity_fixed_P=str(velocity),
                         A_quadrature_error=str(abs(AA-ac)),momentum_quadrature_error=str(abs(pp-pc))))

# Formal bad continuation is an algebraic diagnostic, not a bad vacuum state.
kk,lam=s.symbols('kk lam',positive=True)
mg=-3*s.sqrt(3)*s.atanh(2*kk/s.sqrt(3))/(2*s.pi*kk**2)+2*lam/kk
mb_cont=-3*s.sqrt(3)*s.atan(2*kk/s.sqrt(3))/(2*s.pi*kk**2)-2*lam/kk
assert s.simplify(s.I*mg.subs({kk:s.I*kk,lam:-lam},simultaneous=True)-mb_cont)==0
small={}
for sig,name in [(1,'good'),(-1,'bad')]:
    vv=s.sqrt(1-sig*kk*kk)
    T=s.atanh(2*kk/s.sqrt(3)) if sig==1 else s.atan(2*kk/s.sqrt(3))
    mv=-3*s.sqrt(3)*T/(2*s.pi*kk**2)+2*sig*lam/kk
    pp=-2*s.sqrt(3)*vv*T/s.pi+2*sig*lam*vv*kk
    P=s.Rational(2,3)*vv*kk**3
    P_v=s.diff(P,kk)/s.diff(vv,kk)
    dv=-pp/P_v
    mb=mv+s.diff(2*kk,kk)/s.diff(vv,kk)*dv
    qb=-vv*mb-2*kk*dv
    mbseries=1/(s.pi*kk)+(sig*16/(9*s.pi)-2*lam/3)*kk
    qbseries=-1/(s.pi*kk)+(sig*49/(18*s.pi)-4*lam/3)*kk
    assert s.simplify(s.series(mb,kk,0,3).removeO()-mbseries)==0
    assert s.simplify(s.series(qb,kk,0,3).removeO()-qbseries)==0
    # For any lambda the simultaneous mass term cancels the O(k^0)
    # rational term that would be produced by shifting Pi alone.
    r,t=s.symbols('r t',positive=True)
    mass=lambda z:mbseries.subs(kk,z)
    charge=lambda z:qbseries.subs(kk,z)
    classical=lambda z:-2*z*s.sqrt(1-sig*z*z)
    chargeonly=-s.Rational(2,3)*(charge(t)*2*r*t-charge(r*t)*2*t)
    massonly=-s.Rational(2,3)*(classical(t)*mass(r*t)-classical(r*t)*mass(t))
    assert s.simplify(s.limit(chargeonly,t,0)-4/(3*s.pi)*(r-1/r))==0
    assert s.simplify(s.limit(massonly,t,0)+4/(3*s.pi)*(r-1/r))==0
    total=s.series(chargeonly+massonly,t,0,3).removeO()
    expected=-2*sig/(3*s.pi)*t*t*(1-r*r)*(r+1/r)
    assert s.simplify(total-expected)==0
    small[name]=dict(m_fixed_P=str(mbseries),Pi_fixed_P=str(qbseries),total_topological_small_k=str(expected))

out=dict(claim_label='c, conditional unprojected-continuum tadpole and stationary mean response',
         precision_dps=80,moments=moments,rows=rows,small_amplitude=small,mode_pencil_residuals=mode_residuals,
         physical_projected_expectation=None,
         qualifications=['The scattering continuum is used with asymptotic normalization 1/sqrt(2 omega).',
          'The local tadpole contains a logarithmic soft translation contribution; its adjoint mass moment is exactly zero.',
          'The same soft term cancels between the direct momentum insertion and mean-profile response before q integration.',
          'A fixed-endpoint state and complete projected functional measure have not been constructed.',
          'Mass response varies endpoint sector; it is not renormalization of the mass operator within a fixed sector.'],
         failures_recorded=['Gaussian contraction of the linear Pi insertion alone is zero and is not the full one-loop expectation.',
          'Discarding the mass correction produces a spurious rational topological contribution.'],
         runtime_seconds=time.monotonic()-start)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Direct continuum tadpole and momentum insertion agree with the conditional dispersion derivative.')
print('ALL CHECKS PASSED')

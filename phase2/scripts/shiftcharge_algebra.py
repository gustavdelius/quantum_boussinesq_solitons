#!/usr/bin/env python3
"""Independent symbolic/frame and fixed-height diagnostics. No legacy imports.

All claims are (c) algebra, conditional where a quantum interpretation is used.
Physical residual coefficients remain null; this script does not fit a phase.
"""
import json
import time
from pathlib import Path
import sympy as s
import mpmath as mp

start = time.monotonic()
mp.mp.dps = 80
h, lam, k = s.symbols('h lam k', real=True, positive=True)

# The advertised one-dimensional example is not a classification theorem.
P1, P2, M1, M2, A, B = s.symbols('P1 P2 M1 M2 A B')
F = A*P1*M2+B*M1*P2
assert s.expand(F+F.xreplace({P1:P2, P2:P1, M1:M2, M2:M1})-(A+B)*(M1*P2+M2*P1)) == 0
p = s.symbols('p:6'); m = s.symbols('m:6')
mat = s.Matrix(3,3,lambda i,j:p[i]*m[j+3]-m[i]*p[j+3])
assert s.expand(mat.det()) == 0
minor=s.Matrix(2,2,lambda i,j:p[i]*m[j+3]-m[i]*p[j+3]).det()
assert s.expand(minor-(p[0]*m[1]-m[0]*p[1])*(p[3]*m[4]-m[3]*p[4]))==0
assert minor != 0
a,b,Q = s.symbols('a b Q')
pv=lambda z:2*s.sqrt(3)*z**4-z**2/s.sqrt(3)-s.sqrt(3)/36
chordP=s.Rational(2,3)*s.sqrt(3)*(a+b)*(a-b)**3
assert s.rem(s.expand(pv(a)-pv(b)-chordP), a*a+a*b+b*b-s.Rational(1,4), a)==0
vertices=s.symbols('Q0:3')
assert s.expand(sum(pv(vertices[i])-pv(vertices[(i+1)%3]) for i in range(3)))==0
assert s.expand(sum(2*(vertices[i]-vertices[(i+1)%3]) for i in range(3)))==0
assert s.rem(s.expand(pv(a)+pv(b)+pv(-a-b)),a*a+a*b+b*b-s.Rational(1,4),a)==0
assert s.expand(2*a+2*b+2*(-a-b))==0
# Explicit transmission lift: W_b-W_a = -F(test, chord a->b).
pt,mt,c=s.symbols('pt mt c')
lift=lambda z:c*(2*pt*z-mt*pv(z))
assert s.expand(lift(b)-lift(a)+c*(pt*2*(a-b)-mt*(pv(a)-pv(b))))==0

# Exact first-order inversion and chamber-reversing subtraction.
pi1,pi2,q1,q2,m1,m2=s.symbols('pi1 pi2 q1 q2 m1 m2')
W=(pi1+h*q1)*(M2+h*m2)-(pi2+h*q2)*(M1+h*m1)
dW=q1*M2+pi1*m2-q2*M1-pi2*m1
assert s.expand(W).coeff(h)==dW
Scb=((pi1+h*q1)*(M2+h*m2)+(pi2+h*q2)*(M1+h*m1))/2
dScb=(q1*M2+pi1*m2+q2*M1+pi2*m1)/2
assert s.expand(Scb).coeff(h)==dScb
Q0,Qd,p0d,pone,qone=s.symbols('Q0 Qd p0d pone qone',nonzero=True)
shift=-pone/p0d
assert s.expand(Q0+h*(Qd*shift+qone)).coeff(h)==qone-Qd*pone/p0d
# gamma_1 in labels and in physical momentum agree by the chain rule.
Qdd,p0dd,pone_d,qone_d=s.symbols('Qdd p0dd pone_d qone_d')
dg_label=qone_d/p0d-Qd*pone_d/p0d**2-pone/p0d*(Qdd/p0d-Qd*p0dd/p0d**2)
dg_fixed=(qone_d-Qdd*pone/p0d-Qd*pone_d/p0d+Qd*pone*p0dd/p0d**2)/p0d
assert s.simplify(dg_label-dg_fixed)==0
# If M is fixed, a common multiple of M is invisible in the wedge.
kap,aa=s.symbols('kap aa')
assert s.expand(dW.subs({m1:0,m2:0,q1:kap*P1+aa*M1,q2:kap*P2+aa*M2})-kap*(P1*M2-P2*M1))==0
# Darboux-frame-only shifts fail integrability in general.
x,y=s.symbols('x y',positive=True)
qx=x**2; qy=y**2; mx=x**s.Rational(1,3); my=y**s.Rational(1,3)
curl=-s.diff(qx,x)*s.diff(my,y)-s.diff(qy,y)*s.diff(mx,x)
assert curl.subs({x:1,y:8}) != 0

# Regulated continuity equation, including every boundary term.
z,t=s.symbols('z t',real=True)
phi=s.Function('phi')(z,t); cutoff=s.Function('f')(z)
assert s.simplify(s.diff(z*cutoff*s.diff(phi,t),z)-s.diff(z*cutoff,z)*s.diff(phi,t)-z*cutoff*s.diff(phi,t,z))==0
eta=s.symbols('eta')
et,ex,exx,u,ux,vv=s.symbols('et ex exx u ux v')
lag=(-vv*u+eta*(et-vv*ex))**2/2-(u+eta*ex)**2/2+(u+eta*ex)**3-(ux+eta*exx)**2/2
assert s.expand(s.expand(lag).coeff(eta,2)-((et-vv*ex)**2/2-(1-6*u)*ex**2/2-exx**2/2))==0
assert s.diff(lag,et).expand().coeff(eta,2)==et-vv*ex
# Pi is linear: no Gaussian quadratic insertion. M is an endpoint difference.
Pi0,linear=s.symbols('Pi0 linear')
assert s.expand(Pi0+eta*linear).coeff(eta,2)==0
# Finite canonical algebra control for endpoint charge and global shift.
fields=s.symbols('field0:4'); momenta=s.symbols('momentum0:4')
PB=lambda f,g:sum(s.diff(f,x)*s.diff(g,p)-s.diff(f,p)*s.diff(g,x) for x,p in zip(fields,momenta))
endpoint=fields[-1]-fields[0]
assert PB(endpoint,sum(momenta))==0
assert PB(endpoint,sum((j+1)*fields[j] for j in range(4)))==0
assert PB(fields[1],sum(p*p/2 for p in momenta))==momenta[1]

# Both signs of the equation, positive-direction branches.
expressions={}
for sig,name in [(1,'good'),(-1,'bad')]:
    v=s.sqrt(1-sig*k*k); P=s.Rational(2,3)*v*k**3
    E=s.Rational(2,3)*k**3-sig*s.Rational(8,15)*k**5
    assert s.simplify(s.diff(E,k)-v*s.diff(P,k))==0
    f0=(s.sqrt(3)/s.pi*(s.Rational(3,4)-k*k)*s.atanh(2*k/s.sqrt(3))-3*k/(2*s.pi)) if sig==1 else (s.sqrt(3)/s.pi*(s.Rational(3,4)+k*k)*s.atan(2*k/s.sqrt(3))-3*k/(2*s.pi))
    f=f0+2*lam*k**3/3
    C=lam-2*sig/s.pi
    D=5*sig*lam/6-23/(9*s.pi)
    speed=s.diff(f,k)/s.diff(P,k)
    assert s.simplify(s.series(speed,k,0,3).removeO()-C-D*k*k)==0
    # Conditional frozen-height q=-M F_P, not a new physical assumption.
    q=-2*k*speed
    assert s.simplify(s.limit(q/(2*k),k,0)+C)==0
    # Exact local elastic deformation independently verifies m at fixed v.
    K=s.sqrt(k*k+2*sig*h*lam)
    assert s.simplify(s.diff(2*K,h).subs(h,0)-2*sig*lam/k)==0
    p_ct=2*sig*lam*v*k
    m_ct=2*sig*lam/k
    dv=-p_ct/(s.diff(P,k)/s.diff(v,k))
    mb=m_ct+s.diff(2*k,k)/s.diff(v,k)*dv
    qb=-v*m_ct+s.diff(-2*k*v,k)/s.diff(v,k)*dv
    assert s.simplify(qb+v*mb+2*k*dv)==0
    assert s.simplify(mb+2*lam*k/(3-4*sig*k*k))==0
    assert s.simplify(qb+4*lam*k*v/(3-4*sig*k*k))==0
    # Counterterm charge shifts from an exact fixed-P deformed profile.
    vc=s.sqrt(1-sig*k*k+2*h*lam)
    Pc=s.Rational(2,3)*k**3*vc
    dk=-s.diff(Pc,h).subs(h,0)/s.diff(P,k)
    assert s.simplify(2*dk-mb)==0
    assert s.simplify((s.diff(-2*k*vc,h).subs(h,0)+s.diff(-2*k*v,k)*dk)-qb)==0
    expressions[name]=dict(F=str(f),C=str(C),D=str(D),frozen_height_small_same_direction_c=str(2*D))
assert s.simplify((5*lam/3-46/(9*s.pi)).subs(lam,9/(4*s.pi))+49/(36*s.pi))==0
assert s.simplify((-5*lam/3-46/(9*s.pi)).subs(lam,-3/(2*s.pi))+47/(18*s.pi))==0

def data(K,sign,branch,lv):
    ss=1 if branch=='good' else -1
    v=sign*mp.sqrt(1-ss*K*K); P=mp.mpf(2)/3*v*K**3
    def f(t):
        base=mp.sqrt(3)/mp.pi*(mp.mpf('.75')-t*t)*mp.atanh(2*t/mp.sqrt(3))-3*t/(2*mp.pi) if ss==1 else mp.sqrt(3)/mp.pi*(mp.mpf('.75')+t*t)*mp.atan(2*t/mp.sqrt(3))-3*t/(2*mp.pi)
        return base+2*lv*t**3/3
    Pk=mp.diff(lambda t:mp.mpf(2)/3*sign*mp.sqrt(1-ss*t*t)*t**3,K)
    V=mp.diff(f,K)/Pk
    return P,2*K,-2*K*V

rows=[]
for branch,lv in [('good',mp.mpf(0)),('good',9/(4*mp.pi)),('bad',-3/(2*mp.pi))]:
    for k1,k2,e2 in [('.2','.4',1),('.3','.6',1),('.2','.4',-1)]:
        P1,M1,q1=data(mp.mpf(k1),1,branch,lv)
        P2,M2,q2=data(mp.mpf(k2),e2,branch,lv)
        Q=P1*M2-P2*M1
        val=-mp.mpf(2)/3*(q1*M2-q2*M1)/Q
        rows.append(dict(branch=branch,lambda_value=str(lv),k1=k1,k2=k2,epsilon2=e2,c_effective_frozen_height=str(val)))
    assert abs(mp.mpf(rows[-1]['c_effective_frozen_height'])-mp.mpf(rows[-2]['c_effective_frozen_height']))>mp.mpf('.01')
def classical(K,e):
    V=e*mp.sqrt(1-K*K)
    return (mp.mpf(2)/3*V*K**3,2*K,-2*K*V,-V*(1-4*K*K),V)
def phase(K,L,e,f):
    p1,m1,pi1,ell1,v1=classical(K,e)
    p2,m2,pi2,ell2,v2=classical(L,f)
    aa,bb=(v1/mp.sqrt(3)+K)/2,(v1/mp.sqrt(3)-K)/2
    cc,dd=(v2/mp.sqrt(3)+L)/2,(v2/mp.sqrt(3)-L)/2
    interaction=(aa-cc)*(bb-dd)/((aa-dd)*(bb-cc))
    return mp.mpf(2)/9*(ell1-ell2)*mp.log(interaction)-mp.mpf(2)/3*(pi1*m2-pi2*m1)
ctrows=[]
for e,f in [(1,1),(1,-1)]:
    K,L=mp.sin(mp.mpf('.3')),mp.sin(mp.mpf('.6'))
    p1=classical(K,e)[0]; p2=classical(L,f)[0]
    derivative1=mp.diff(lambda z:phase(z,L,e,f),K)/mp.diff(lambda z:classical(z,e)[0],K)
    derivative2=mp.diff(lambda z:phase(K,z,e,f),L)/mp.diff(lambda z:classical(z,f)[0],L)
    wanted=3*phase(K,L,e,f)-4*(p1*derivative1+p2*derivative2)
    # Exact stiffness deformation, with the physical momenta inverted afresh.
    def deformed(z):
        stiffness=1+2*z
        x=mp.findroot(lambda x:stiffness**2*classical(x,e)[0]-p1,K)
        y=mp.findroot(lambda y:stiffness**2*classical(y,f)[0]-p2,L)
        return stiffness**mp.mpf('1.5')*phase(x,y,e,f)
    step=mp.mpf('1e-14')
    fd=lambda hh:(deformed(hh)-deformed(-hh))/(2*hh)
    observed=(4*fd(step/2)-fd(step))/3
    assert abs(observed-wanted)<mp.mpf('1e-50')
    ctrows.append(dict(epsilon1=e,epsilon2=f,K_lambda=str(wanted),finite_difference_error=str(abs(observed-wanted))))
out=dict(claim_label='c, conditional algebra where stated',expressions=expressions,pointwise_frozen_height=rows,
         counterterm_phase=ctrows,
         c_bad=None,c_good=None,classification='one-dimensional exhibited wedge, not exhaustive',
         failures_recorded=['Constant c from the frozen-height dispersion ansatz fails across kinematics and directions.','Changing only Darboux gamma and keeping spatial delays fixed fails the curl condition.'],runtime_seconds=time.monotonic()-start)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Rank-zero witness, fixed-P/frame algebra, centroid boundary identity and counterterm checked.')
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""(c) Independent classical gradients, branch-aware continuum integrals,
KP identities and homogeneous one-loop transmission deformation.
No physical determinant or quantum pole is asserted."""
import json,time
from pathlib import Path
import mpmath as m
import sympy as S
start=time.perf_counter();m.mp.dps=80;r=m.sqrt(3);tol=m.mpf('1e-60')
worst={}
def check(name,x,y=0,bound=tol):
    err=abs(x-y);worst[name]=max(worst.get(name,m.mpf(0)),err)
    assert err<bound,(name,m.nstr(err,12),m.nstr(x,20),m.nstr(y,20))
def data(a,ep):
    k=m.sin(a);v=ep*m.cos(a)
    return k,v,(v/r+k)/2,(v/r-k)/2
def pmom(a,ep):return m.mpf(2)/3*ep*m.sin(a)**3*m.cos(a)
def z(a,ep,q,eta):
    k,v,pa,pb=data(a,ep);return ((pb-q)/(pa-q))**eta
def delay(a,ep,q,eta):return -ep*m.log(abs(z(a,ep,q,eta)))/m.sin(a)
def classical(a,ep,q,eta):
    k,v,pa,pb=data(a,ep);ell=-ep*m.cos(3*a);ellD=3*r*(4*q**3-q)
    MD=ep*eta*q;PiD=ep*eta*(6*q*q-1)/r
    return m.mpf(2)/9*(ell-ellD)*ep*m.log(abs(z(a,ep,q,eta)))-m.mpf(2)/3*(-2*k*v*MD-PiD*2*k)
def cross(a1,e1,a2,e2):
    k,v,aa,bb=data(a1,e1);k,v,cc,dd=data(a2,e2)
    return (aa-cc)*(bb-dd)/((aa-dd)*(bb-cc))
def bulkcl(a1,e1,a2,e2):
    k1,v1,_,_=data(a1,e1);k2,v2,_,_=data(a2,e2)
    return m.mpf(2)/9*(-e1*m.cos(3*a1)+e2*m.cos(3*a2))*m.log(abs(cross(a1,e1,a2,e2)))+m.mpf(8)/3*k1*k2*(v1-v2)
def phroots(q):
    c=m.sign(q)*m.sqrt(1+q*q)/(2*r)
    return c+1j*q/2,c-1j*q/2
def Z(q,Q,eta=1):
    a,b=phroots(q);return ((b-Q)/(a-Q))**eta
def dD(q,Q,eta):
    # log derivative directly from the rational phonon transmission
    a,b=phroots(q);c=m.sign(q)*q/(2*r*m.sqrt(1+q*q))
    return m.im(eta*((c-1j/2)/(b-Q)-(c+1j/2)/(a-Q)))
def thetaS(q,a,ep):
    angles=[a+2*m.pi/3,a-2*m.pi/3,a];ww=[1,1,-2]
    return -sum(w*(m.sign(q)*m.atan(m.sin(f)/abs(q))+ep*m.atan(m.sin(f)*m.cos(f)/(abs(q)*m.sqrt(1+q*q)+q*q+m.sin(f)**2))) for f,w in zip(angles,ww))
def dS(q,a,ep):
    return sum(w*m.sin(f)/(q*q+m.sin(f)**2)*(1+ep*m.sign(q)*m.cos(f)/m.sqrt(1+q*q)) for f,w in zip([a+2*m.pi/3,a-2*m.pi/3,a],[1,1,-2]))
def thetaD(q,Q,eta):
    # Continuous branch from zero on EACH half-line, including both orientations.
    s=-2*r*Q;X=m.sign(q)*m.sqrt(1+q*q)+s
    raw=-2*eta*m.atan2(r*q,X)
    X0=m.sign(q)+s
    return raw+(2*m.pi*eta*m.sign(q) if X0<0 else 0)
def integral(a,ep,Q,eta,form):
    def fun(y):
        if y==0 or y==1:return m.mpf(0)
        q=y/(1-y)
        if form=='direct': val=thetaS(q,a,ep)*dD(q,Q,eta)+thetaS(-q,a,ep)*dD(-q,Q,eta)
        else: val=-thetaD(q,Q,eta)*dS(q,a,ep)-thetaD(-q,Q,eta)*dS(-q,a,ep)
        return val/(4*m.pi*(1-y)**2)
    return m.quad(fun,[0,m.mpf('.15'),m.mpf('.4'),m.mpf('.7'),1])
def pmode(a,ep):
    return -ep/(2*m.pi)*sum(w*m.sin(f)*m.log(abs(m.tan(f/2))) for f,w in zip([a+2*m.pi/3,a-2*m.pi/3,a],[1,1,-2]))
for ep in [-1,1]:
 for eta in [-1,1]:
  for av in ['.2','.4','.8']:
    a=m.mpf(av);Q=eta*m.mpf('.63')
    check('classical_negative_delay',m.diff(lambda aa:classical(aa,ep,Q,eta),a)/m.diff(lambda aa:pmom(aa,ep),a),-delay(a,ep,Q,eta))
    for q in [m.mpf('-.7'),m.mpf('.3'),m.mpf('2')]:
      check('defect_modulus',abs(Z(q,Q,eta)),1)
      check('defect_phase_derivative',m.diff(lambda z:thetaD(z,Q,eta),q),dD(q,Q,eta))
      check('soliton_phase_derivative',m.diff(lambda z:thetaS(z,a,ep),q),dS(q,a,ep))
      k,v,pa,pb=data(a,ep);a0,b0=phroots(q)
      cr=(pa-a0)*(pb-b0)/((pa-b0)*(pb-a0))
      check('phonon_edge_factorisation',cr,Z(q,pb)/Z(q,pa))
      check('soliton_phase_exponential',m.exp(1j*thetaS(q,a,ep)),cr)
# Classical edge bootstrap: fix exact directional seam before doing loop algebra.
for th in [m.mpf('.23'),m.mpf('.71')]:
 qs=[-m.cos(th+2*m.pi*j/3)/r for j in [1,2,3]]
 for i in range(3):
  qi,qj=qs[i],qs[(i+1)%3];k=abs(qi-qj);v=r*(qi+qj);ai=m.acos(abs(v));ei=int(m.sign(v))
  for ep in [-1,1]:
   for eta in [-1,1]:
    a=m.mpf('.13');rho=eta*m.sign(qi-qj);Pi=-2*m.sin(a)*ep*m.cos(a)
    check('classical_edge_seam',classical(a,ep,qj,eta)-classical(a,ep,qi,eta),ep*rho*(bulkcl(a,ep,ai,ei)+2*k*Pi))
# Rank-two lift, including exceptional same-triangle two-defect composition.
pot=lambda q:2*r*q**4-q*q/r-r/36
lift=lambda a,ep,Q,eta: ep*eta*(2*m.sin(a)*pot(Q)-pmom(a,ep)*2*Q)
for th in [m.mpf('.23'),m.mpf('.71')]:
 qs=[-m.cos(th+2*m.pi*j/3)/r for j in [1,2,3]]
 for eta in [-1,1]:
  for ep in [-1,1]:
   a=m.mpf('.13')
   check('special_classical_fusion',classical(a,ep,qs[0],eta)+classical(a,ep,qs[1],eta),classical(a,ep,qs[2],-eta))
   check('special_lift_fusion',lift(a,ep,qs[0],eta)+lift(a,ep,qs[1],eta),lift(a,ep,qs[2],-eta))
   for i in range(3):
    qi,qj=qs[i],qs[(i+1)%3];k=abs(qi-qj);v=r*(qi+qj)
    FF=pmom(a,ep)*2*k-2*m.sin(a)*m.mpf(2)/3*v*k**3
    check('lift_edge_equation',lift(a,ep,qj,eta)-lift(a,ep,qi,eta),ep*eta*m.sign(qi-qj)*FF)
  for q in [m.mpf('-.8'),m.mpf('.6')]:
   check('special_phonon_fusion',Z(q,qs[0],eta)*Z(q,qs[1],eta),Z(q,qs[2],-eta))
   check('special_trace_density_fusion',dD(q,qs[0],eta)+dD(q,qs[1],eta),dD(q,qs[2],-eta))
# Stable regular benchmark: two quadrature forms and precision replication.
a=m.mpf('.35');Q=m.mpf('.5');ep=eta=1
v80=integral(a,ep,Q,eta,'direct');parts=integral(a,ep,Q,eta,'parts')
check('integration_by_parts',v80,parts)
with m.workdps(60):v60=integral(a,ep,Q,eta,'direct')
check('precision_60_80',v80,v60,m.mpf('1e-55'))
# Boundary audit's infinity-reference table differs by an explicitly retained endpoint.
# On s<-1: thetaD_ref = thetaD_zero -4pi/3 (q>0), +2pi/3 (q<0).
# thetaS(0+)=2pi, thetaS(0-)=0 for ep=+1, hence I_ref=I_zero-2pi/3.
ref=v80-2*m.pi/3
check('audited_reference_table',ref,m.mpf('-1.3321099117235175816'),m.mpf('6e-20'))
# No implicit identification of the two branches as the same absolute phase.
corrected=v80+delay(a,ep,Q,eta)*pmode(a,ep)
# Independent fixed-P finite difference of the classical phase confirms the sign.
hh=m.mpf('1e-20');p0=pmom(a,ep);p1=pmode(a,ep)
ap=m.findroot(lambda aa:pmom(aa,ep)+hh*pmode(aa,ep)-p0,(a-m.mpf('.001'),a+m.mpf('.001')))
am=m.findroot(lambda aa:pmom(aa,ep)-hh*pmode(aa,ep)-p0,(a-m.mpf('.001'),a+m.mpf('.001')))
check('fixed_P_taylor', (classical(ap,ep,Q,eta)-classical(am,ep,Q,eta))/(2*hh),delay(a,ep,Q,eta)*p1,m.mpf('1e-35'))
# Counterterm K_lambda: numerical derivative of classical phase, all channels.
kvals={}
for e1,e2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
 a1=m.mpf('.3');a2=m.mpf('.6');k1,v1,_,_=data(a1,e1);k2,v2,_,_=data(a2,e2)
 p1=pmom(a1,e1);p2=pmom(a2,e2);loga=m.log(abs(cross(a1,e1,a2,e2)))
 g1=-3*m.cos(2*a1)/(m.sin(a1)*m.sin(3*a1));g2=-3*m.cos(2*a2)/(m.sin(a2)*m.sin(3*a2))
 dx1=-loga/k1+g1*2*k2;dx2=loga/k2-g2*2*k1
 check('bulk_darboux_gradient1',m.diff(lambda aa:bulkcl(aa,e1,a2,e2),a1)/m.diff(lambda aa:pmom(aa,e1),a1),-dx1)
 check('bulk_darboux_gradient2',m.diff(lambda aa:bulkcl(a1,e1,aa,e2),a2)/m.diff(lambda aa:pmom(aa,e2),a2),-dx2)
 kval=3*bulkcl(a1,e1,a2,e2)+4*(p1*dx1+p2*dx2)
 kother=m.mpf(2)/3*(v2-v1)*loga+8*k1*k2*(v1-v2)+4*(p1*g1*2*k2-p2*g2*2*k1)
 check('bulk_lambda_response',kval,kother);kvals[str((e1,e2))]=str(kval)
# Explicit known bulk trace representative; two independent half-line channels.
a1=m.mpf('.3');a2=m.mpf('.6')
def pair_integral(a1,e1,a2,e2):
    def fun(y):
        if y in [0,1]:return m.mpf(0)
        q=y/(1-y)
        return (thetaS(q,a1,e1)*dS(q,a2,e2)+thetaS(-q,a1,e1)*dS(-q,a2,e2))/(4*m.pi*(1-y)**2)
    return m.quad(fun,[0,m.mpf('.15'),m.mpf('.4'),m.mpf('.7'),1])
pairrows=[]
for e1,e2 in [(1,1),(1,-1)]:
    I12=pair_integral(a1,e1,a2,e2);I21=pair_integral(a2,e2,a1,e1)
    check('bulk_trace_endpoint', (I12+I21)/2,-m.pi*(e1+e2)/4)
    J=(I12-I21)/2
    FF=pmom(a1,e1)*2*m.sin(a2)-pmom(a2,e2)*2*m.sin(a1)
    assert abs(FF)>m.mpf('.001')
    pairrows.append({'eps':[e1,e2],'J':str(J),'F_coefficient':str(FF)})
# Explicit high-frequency asymptotes from the independent rational factors.
qp=S.symbols('qp',positive=True);ssym=S.symbols('s',real=True)
for sign in [-1,1]:
    for eta in [-1,1]:
        zz=((sign*S.sqrt(1+qp*qp)+ssym-S.I*S.sqrt(3)*sign*qp)/(sign*S.sqrt(1+qp*qp)+ssym+S.I*S.sqrt(3)*sign*qp))**eta
        target=(-S.Rational(1,2)-S.I*eta*S.sqrt(3)/2)
        assert S.simplify(S.limit(zz,qp,S.oo)-target)==0

out={'label':'(c), continuum representative only','digits':80,'default_tolerance':'1e-60',
 'errors':{k:str(v) for k,v in worst.items()},'I_zero_reference':str(v80),'I_infinity_reference':str(ref),
 'fixed_P_mode_trace_representative':str(corrected),'K_lambda':kvals,'bulk_pairs':pairrows,'seconds':time.perf_counter()-start}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2));print('ALL CHECKS PASSED')

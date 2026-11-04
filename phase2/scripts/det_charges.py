#!/usr/bin/env python3
"""65-digit conditional continuum trace, fixed-P invariant, and phase constants.

Uses the inherited equal-mode-count endpoint as a conditional benchmark, not
as a newly selected regulator or a derived projected determinant.
"""
import json
from pathlib import Path
import time
import mpmath as mp
import sympy as s

started=time.perf_counter(); mp.mp.dps=65
z,U,C=s.symbols('z U C',real=True)
# Gaussian Wick moments <z>=<z^3>=0, <z^2>=C.
poly=s.Poly(s.expand((U+z)**3-3*C*(U+z)),z)
expect=sum(coef*({0:1,1:0,2:C,3:0}[power[0]]) for power,coef in poly.terms())
assert s.expand(expect-U**3)==0
q,Lambda=s.symbols('q Lambda',positive=True)
assert s.diff(s.sqrt(1+q*q),q)==q/s.sqrt(1+q*q)
v,l,p,e,dp,de=s.symbols('v l p e dp de')
# Fixed-P Taylor correction and the invariance under tangent reparametrization.
assert s.expand((e+v*dp)-v*(p+dp)-(e-v*p))==0
assert s.expand((v*dp-l)-v*dp+l)==0
dp1,dp2,g1,g2,d0=s.symbols('dp1 dp2 g1 g2 d0')
label_phase=d0+g1*(-dp1)+g2*(-dp2)
assert s.expand(label_phase-(d0-dp1*g1-dp2*g2))==0
P1,P2,M1,M2=s.symbols('P1 P2 M1 M2')
F=P1*M2-P2*M1
assert s.expand(F.xreplace({P1:P2,P2:P1,M1:M2,M2:M1})+F)==0
assert s.expand(F.subs({P1:-P1,M1:-M1},simultaneous=True)+F)==0
assert F.subs({P1:0,M1:0})==0

def roots(a): return [(a+2*mp.pi/3,1),(a-2*mp.pi/3,1),(a,-2)]
def theta(q,a,eps=1):
    r=mp.sqrt(1+q*q); aq=abs(q)
    return -sum(c*(mp.sign(q)*mp.atan(mp.sin(f)/aq)+eps*mp.atan(mp.sin(f)*mp.cos(f)/(aq*r+q*q+mp.sin(f)**2))) for f,c in roots(a))
def dt(q,a,eps=1):
    return sum(c*mp.sin(f)/(q*q+mp.sin(f)**2)*(1+eps*mp.sign(q)*mp.cos(f)/mp.sqrt(1+q*q)) for f,c in roots(a))
def em(a): return sum(c*(2*mp.sin(f)-mp.sin(2*f)*mp.log(abs(mp.tan(f/2)))) for f,c in roots(a))/(4*mp.pi)
def pm(a,eps=1): return -eps*sum(c*mp.sin(f)*mp.log(abs(mp.tan(f/2))) for f,c in roots(a))/(2*mp.pi)
def ell(a): return mp.cos(a)*pm(a)-em(a)
def ell_analytic(a):
    return (3*mp.sin(a)+sum(mp.sin(f)*(mp.cos(f)-mp.cos(a))*mp.log(abs(mp.tan(f/2)))
                            for f,c in roots(a)[:2]))/(2*mp.pi)
def pc(a,eps=1): return -eps*mp.sqrt(3)*mp.cos(a)/mp.pi*mp.log(abs(mp.tan((a+2*mp.pi/3)/2)/mp.tan((a-2*mp.pi/3)/2)))
def p0(a,eps=1): return 2*eps*mp.sin(a)**3*mp.cos(a)/3
def e0(a): return 2*mp.sin(a)**3/3-8*mp.sin(a)**5/15

rows=[]
for a in map(mp.mpf,['0.15','0.5','0.85']):
    assert abs(ell(a)-ell_analytic(a))<mp.mpf('1e-58')
    for eps in [1,-1]:
        vv=eps*mp.cos(a)
        for sign in [1,-1]:
            assert abs(theta(sign*mp.mpf('1e-40'),a,eps)-mp.pi*(sign+eps))<mp.mpf('1e-37')
        for qq in map(mp.mpf,['0.4','-1.3']):
            assert abs(mp.diff(lambda Q:theta(Q,a,eps),qq)-dt(qq,a,eps))<mp.mpf('1e-58')
        pnum=mp.quad(lambda Q:Q*(dt(Q,a,eps)-dt(-Q,a,eps)),[0,1,mp.inf])/(4*mp.pi)
        assert abs(pnum-pm(a,eps))<mp.mpf('1e-57')
        assert abs(-eps*mp.diff(ell,a)/mp.sin(a)-pc(a,eps))<mp.mpf('1e-57')
        assert abs(mp.diff(e0,a)-vv*mp.diff(lambda A:p0(A,eps),a))<mp.mpf('1e-57')
        assert abs((vv*pc(a,eps)-ell(a))-vv*pc(a,eps)-(em(a)-vv*pm(a,eps)))<mp.mpf('1e-57')
        cut=mp.mpf('1e22'); T=mp.sqrt(1+cut*cut)
        raw=sum(c*mp.sin(f)*(T-1+mp.cos(f)/2*(mp.log((T-mp.cos(f))/(T+mp.cos(f)))-mp.log((1-mp.cos(f))/(1+mp.cos(f))))) for f,c in roots(a))/(2*mp.pi)
        B=cut*T*(theta(cut,a,eps)-theta(-cut,a,eps))/(4*mp.pi)
        ct=3*mp.sin(a)*(T-1)/mp.pi
        renerr=abs(raw-B+ct-em(a))
        assert renerr<mp.mpf('1e-21')
        assert abs(B/cut-3*mp.sin(a)/(2*mp.pi))<mp.mpf('1e-42')
        # Check finite-cutoff antiderivative by independent 65-digit quadrature.
        cutoff=mp.mpf(25); T25=mp.sqrt(1+cutoff*cutoff)
        raw25=sum(c*mp.sin(f)*(T25-1+mp.cos(f)/2*(mp.log((T25-mp.cos(f))/(T25+mp.cos(f)))-mp.log((1-mp.cos(f))/(1+mp.cos(f))))) for f,c in roots(a))/(2*mp.pi)
        rawnum=mp.quad(lambda Q:Q*mp.sqrt(1+Q*Q)*(dt(Q,a,eps)+dt(-Q,a,eps)),[0,1,5,cutoff])/(4*mp.pi)
        assert abs(raw25-rawnum)<mp.mpf('1e-57')
        rows.append({'alpha':str(a),'epsilon':eps,'mode_energy':str(em(a)),
                     'mode_momentum':str(pm(a,eps)),'canonical_momentum':str(pc(a,eps)),
                     'fixed_P_energy_coefficient':str(-ell(a)),'cutoff_error':str(renerr)})

# Analytic small-amplitude expansion via derivatives of the analytic pc function.
assert abs(ell_analytic(mp.mpf(0)))<mp.mpf('1e-58')
for n,expected in [(1,0),(3,4/(3*mp.pi)),(5,-14/(45*mp.pi))]:
    assert abs(mp.diff(ell_analytic,mp.mpf(0),n)/mp.factorial(n)-expected)<mp.mpf('1e-55')
for n,expected in [(1,-4/mp.pi),(3,8/(9*mp.pi)),(5,-8/(45*mp.pi))]:
    assert abs(mp.diff(pc,mp.mpf(0),n)/mp.factorial(n)-expected)<mp.mpf('1e-55')
a=mp.mpf('1e-7')
assert abs((-ell(a)+4*a**3/(3*mp.pi)-14*a**5/(45*mp.pi))/a**7)<1
assert abs(-ell(a)/p0(a)+2/mp.pi)<mp.mpf('1e-13')
assert abs(p0(a)/a**3-mp.mpf(2)/3)<mp.mpf('1e-13')
assert abs(pc(a)/a+4/mp.pi)<mp.mpf('1e-13')

phase=[]
a1,a2=mp.mpf('0.3'),mp.mpf('0.6')
for e1 in [1,-1]:
    for e2 in [1,-1]:
        def integral(A,E,B,F):
            return mp.quad(lambda Q:theta(Q,A,E)*dt(Q,B,F)+theta(-Q,A,E)*dt(-Q,B,F),[0,1,mp.inf])/(4*mp.pi)
        i12=integral(a1,e1,a2,e2); i21=integral(a2,e2,a1,e1)
        assert abs(i12+i21+mp.pi*(e1+e2)/2)<mp.mpf('1e-54')
        assert abs((i12-i21)/2-i12-mp.pi*(e1+e2)/4)<mp.mpf('1e-54')
        assert abs(mp.exp(1j*(i12+i21))*mp.exp(1j*mp.pi*(e1+e2)/2)-1)<mp.mpf('1e-54')
        phase.append({'epsilon1':e1,'epsilon2':e2,'I12':str(i12),'I21':str(i21),'J12':str((i12-i21)/2)})

result={'precision':mp.mp.dps,'charges':rows,'phase_constants':phase,
        'conditional_small_amplitude_energy_over_absP':str(-2/mp.pi),
        'runtime_seconds':time.perf_counter()-started,
        'scope':'Conditional trace and algebra, not an absolute physical E(P) or S matrix.',
        'failures_recorded':['Dropping the equal-mode-count endpoint leaves a nonzero linear UV divergence after the specified tadpole subtraction.']}
Path(__file__).with_name('det_charges_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('ALL CHECKS PASSED')

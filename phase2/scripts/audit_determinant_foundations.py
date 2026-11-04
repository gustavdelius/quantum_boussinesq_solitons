#!/usr/bin/env python3
"""Independent mathematical controls for the fluctuation determinant audit.

(c) formulas tested here are integral/field identities, not a computation of
an absolute two-soliton functional determinant. No regularisation or physical
phase convention is selected. The documented failed *prior claims* are output
as audit findings, not hidden by relaxed tolerances. Working precision: 65 dps.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 65
OUT = {'precision': mp.mp.dps, 'prior_claim_failures': [], 'implementation_failures': ['Initial Hessian illustration imposed an unjustified >1 bound on eigenvalue minus diagonal entry and failed. Replaced by an exact symbolic characteristic-polynomial test, not a weaker numerical tolerance.']}

def n(x):
    return mp.nstr(x, 30)

def theta(q, a, e):
    k, v = mp.sin(a), e*mp.cos(a)
    pb = (v/mp.sqrt(3)-k)/2
    z = pb-mp.sign(q)*mp.sqrt(1+q*q)/(2*mp.sqrt(3))-1j*q/2
    return 2*mp.arg(1+k/z)

def dt(q, a, e):
    return sum(c*mp.sin(p)/(q*q+mp.sin(p)**2)
               *(1+e*mp.sign(q)*mp.cos(p)/mp.sqrt(1+q*q))
               for c,p in zip((1,1,-2),(a+2*mp.pi/3,a-2*mp.pi/3,a)))

def integral(a,e,b,f):
    return mp.quad(lambda q: theta(q,a,e)*dt(q,b,f)
                   +theta(-q,a,e)*dt(-q,b,f),[0,mp.mpf('.1'),1,10,mp.inf])/(4*mp.pi)

def endpoints_and_constants():
    a,b=mp.mpf('.3'),mp.mpf('.7')
    rows=[]
    for e,f in ((1,1),(1,-1),(-1,1),(-1,-1)):
        I,R=integral(a,e,b,f),integral(b,f,a,e)
        boundary=-mp.pi*(e+f)/2
        assert abs(I+R-boundary)<mp.mpf('1e-58')
        rows.append(dict(e1=e,e2=f,I=n(I),I_plus_reverse=n(I+R),J_minus_I=n(-boundary/2)))
    for e in (-1,1):
        assert abs(integral(a,e,a,e)+e*mp.pi/2)<mp.mpf('1e-58')
        for sign in (-1,1):
            endpoint=mp.pi*(sign+e)
            assert abs(theta(sign*mp.mpf('1e-50'),a,e)-endpoint)<mp.mpf('1e-47')
    OUT['phase_endpoints']=rows
    OUT['prior_claim_failures'].append('The Phase 2e same-direction boundary -pi omits orientation; it is +pi for --. Raw I is not exchange-antisymmetric, and has diagonal -epsilon*pi/2.')
    print('1. Both directions: endpoint, exchange, diagonal and I/J constants checked.',flush=True)

def density_identity():
    rows=[]
    for sa,e,sq in (('.4',1,'.6'),('.8',-1,'-.9'),('.35',1,'-1.2'),('.6',-1,'.8')):
        a,q=mp.mpf(sa),mp.mpf(sq)
        k,v=mp.sin(a),e*mp.cos(a)
        omega=abs(q)*mp.sqrt(1+q*q)
        A=mp.exp(1j*theta(q,a,e))
        def density(y):
            h=(1+mp.tanh(y))/2
            hp=k/(4*mp.cosh(y)**2)
            hpp=-k*k*mp.tanh(y)/(4*mp.cosh(y)**2)
            g=1+(A-1)*(h+hp/(1j*q))
            gp=(A-1)*(hp+hpp/(1j*q))
            return (-2j*omega*(abs(g)**2-1)
                    -v*(mp.conj(g)*gp-g*mp.conj(gp)))*2/k/omega
        actual=mp.quad(density,[-90,-10,0,10,90])
        expected=-2j*dt(q,a,e)
        err=abs(actual-expected)
        assert err<mp.mpf('1e-57')
        rows.append(dict(alpha=sa,epsilon=e,q=sq,error=n(err)))
    OUT['potential_mode_density']=rows
    print('2. Direct normalized potential-field Wronskian density identity checked.',flush=True)

def profile_modes(x,a,e,X):
    k,kp=mp.sin(a),mp.cos(a)
    v,vp=e*mp.cos(a),-e*mp.sin(a)
    y=k*(x-X)/2
    sech2=1/mp.cosh(y)**2
    u=k*k*sech2/2
    ux=-k**3*sech2*mp.tanh(y)/2
    ua=k*kp*sech2-k*k*kp*(x-X)*sech2*mp.tanh(y)/2
    fa=kp*(1+mp.tanh(y))+k*kp*(x-X)*sech2/2
    pia=-vp*u-v*ua
    return (fa,pia,-u,v*ux)

def field_symplectic_form():
    a,b=mp.mpf('.4'),mp.mpf('.7')
    k=mp.sin(a)
    local=mp.quad(lambda y:(lambda z:z[1]*z[2]-z[3]*z[0])(profile_modes(2*y/k,a,1,0))*2/k,
                  [-90,-10,0,10,90])
    derivative=mp.diff(lambda t:mp.mpf(2)/3*mp.sin(t)**3*mp.cos(t),a)
    assert abs(local-derivative)<mp.mpf('1e-58')
    # Vary independently the two separated asymptotic profiles. Potential tails
    # are retained. This is not substituted for an exact collision background.
    def mixed(R):
        def integrand(x):
            p=profile_modes(x,a,1,-R/2)
            q=profile_modes(x,b,1,R/2)
            return p[1]*q[0]-q[1]*p[0]
        return mp.quad(integrand,[-mp.inf,-R/2-30,-R/2,-R/2+30,0,R/2-30,R/2,R/2+30,mp.inf])
    expected=4*mp.cos(a)*mp.cos(2*b) # -M_1' Pi_2'
    values=[]
    for R in (200,400):
        value=mixed(mp.mpf(R))
        values.append(dict(separation=R,pairing=n(value),error=n(value-expected)))
    assert abs(mp.mpf(values[-1]['pairing'])-expected)<mp.mpf('1e-28')
    # Numerical tail and derivative ratio; exact trig identity independently.
    assert abs(profile_modes(mp.mpf(500),a,1,0)[0]-2*mp.cos(a))<mp.mpf('1e-60')
    t=sp.symbols('t',positive=True)
    P=sp.Rational(2,3)*sp.sin(t)**3*sp.cos(t)
    assert sp.trigsimp(sp.diff(P,t)-sp.Rational(2,3)*sp.sin(t)*sp.sin(3*t))==0
    assert sp.trigsimp(sp.diff(sp.cos(t),t)/sp.diff(P,t)+sp.Rational(3,2)/sp.sin(3*t))==0
    OUT['field_pairings']=dict(single=n(local),mixed_limit=n(expected),mixed_checks=values,
                              alpha_mode_right_tail=n(2*mp.cos(a)))
    OUT['prior_claim_failures'].append('The potential alpha-variation has a nonzero constant tail. Asymptotically separated alpha modes have a nonzero cross symplectic pairing; block cancellation cannot be justified by localization alone.')
    print('3. Translation pair and nonlocalized alpha-mode cross pairing checked.',flush=True)

def hessian_and_localized_mode_controls():
    # Signature conclusion for a finite real symmetric correction, not a field
    # determinant/Maslov spectral-flow proof. Off-diagonal entries are retained.
    D=mp.diag([-3/(2*mp.sin(3*mp.mpf(a))) for a in ('.3','.7')])
    B=mp.matrix([[2,3],[3,-1]])
    vals=mp.eigsy(100*D+B,eigvals_only=True)
    assert all(v<0 for v in vals)
    h11,h22,h12=sp.symbols('h11 h22 h12',real=True)
    H=sp.Matrix([[h11,h12],[h12,h22]])
    assert sp.expand((H-h11*sp.eye(2)).det())==-h12**2
    assert sp.expand((H-h22*sp.eye(2)).det())==-h12**2
    w,W,T=sp.symbols('w W T',positive=True,real=True)
    alpha=sp.cos(W*T)-sp.I*(w*w+W*W)/(2*w*W)*sp.sin(W*T)
    beta=sp.I*(W*W-w*w)/(2*w*W)*sp.sin(W*T)
    assert sp.simplify(sp.expand_complex(alpha*sp.conjugate(alpha)-beta*sp.conjugate(beta))-1)==0
    # Exact transfer matrix for a frequency pulse omega -> Omega -> omega.
    transfer=sp.Matrix([[sp.cos(W*T),sp.sin(W*T)/W],[-W*sp.sin(W*T),sp.cos(W*T)]])
    initial=sp.Matrix([1,-sp.I*w])
    final=transfer*initial
    assert sp.simplify((final[0]+sp.I*final[1]/w)/2-alpha)==0
    assert sp.simplify((final[0]-sp.I*final[1]/w)/2-beta)==0
    bv=complex(beta.subs({w:1,W:2,T:1}).evalf(40))
    assert abs(bv)>.6
    OUT['limited_hessian_control']=[n(v) for v in vals]
    OUT['equal_frequency_pulse_counterexample']=dict(omega=1,Omega=2,duration=1,beta_imag=bv.imag)
    OUT['prior_claim_failures'].append('Equal asymptotic localized frequencies do not imply determinant cancellation: the exact frequency-pulse control has nonzero Bogoliubov beta. This is a counterexample to the logical implication, not a computation of defect beta.')
    print('4. Limited endpoint Hessian signature and localized-mode logical control checked.',flush=True)


def comoving_sign():
    x,t,v,w=sp.symbols('x t v w',real=True)
    f=sp.Function('f')
    phi=sp.exp(-sp.I*w*t)*f(x-v*t)
    derived=sp.simplify(sp.diff(phi,t,2)/sp.exp(-sp.I*w*t))
    wanted=-w*w*f(x-v*t)+2*sp.I*v*w*sp.diff(f(x-v*t),x)+v*v*sp.diff(f(x-v*t),x,2)
    assert sp.simplify(derived-wanted)==0
    OUT['comoving_derivative_sign']='+2 i v omega d_xi'
    OUT['prior_claim_failures'].append('The displayed co-moving ODE has the wrong sign on its first derivative: +2 i v omega d_xi follows from e^(-i omega t) f(x-vt). The subsequent quartic uses the correct sign.')
    print('5. Co-moving derivative sign checked symbolically.',flush=True)

if __name__=='__main__':
    endpoints_and_constants()
    density_identity()
    field_symplectic_form()
    hessian_and_localized_mode_controls()
    comoving_sign()
    Path(__file__).with_name('audit_determinant_results.json').write_text(json.dumps(OUT,indent=2)+'\n')
    for failure in OUT['prior_claim_failures']:
        print('PRIOR CLAIM FAIL:',failure)
    print('ALL CHECKS PASSED')

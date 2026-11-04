#!/usr/bin/env python3
"""Independent audit; no imports from project checkers and no existing-file writes.

Default: recompute analytic, quadrature, grid and independent half-line checks.
--skip-edge: omit the expensive inverse-Laplace checks (not the final audit run).
Output is audit_twobody_cont_results.json. Refutations assert discrepancies.
All conclusions are project computations (c), conditional where documented.
"""
import argparse
import hashlib
import json
import time
import re
from pathlib import Path
from functools import lru_cache

import mpmath as mp
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
mp.mp.dps = 70
START = time.monotonic()
RESULT = {}
INPUTS = [HERE / n for n in (
    'twobody_cont_phase.py', 'twobody_cont_det_antisym.py', 'twobody_cont_grid.py',
    'twobody_cont_phase.json', 'twobody_cont_det_antisym.json', 'twobody_cont_grid.json',
    'twobody_toda_grid.json', 'twobody_toda_scan.json', 'frame_analysis.json')]
INPUTS += [HERE.parent / n for n in ('twobody_cont.tex', 'twobody_cont.pdf',
                                     'twobody_toda.tex', 'audit_frame_dictionary.tex')]
HASHES = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}


def close(a, b, tol='1e-55'):
    assert abs(a-b) < mp.mpf(tol), (mp.nstr(a, 30), mp.nstr(b, 30), tol)


def emit(name, value):
    RESULT[name] = value
    print(name, json.dumps(value, default=str)[:1200], flush=True)


def symbolic():
    x, k, b, c, lam, c5 = sp.symbols('x k b c lam c5', real=True)
    u = k**2/(2*sp.cosh(k*x/2)**2)
    ode = lambda f: sp.simplify(sp.expand_trig(k*k*f-sp.diff(f,x,2)-3*f*f))
    assert ode(u) == 0
    assert sp.simplify(ode(-u)+6*u*u) == 0
    # Integrating the travelling PDE twice fixes the profile sign, without tau code.
    P = sp.Rational(2,3)*sp.sinh(b)**3*sp.cosh(b)
    E = sp.Rational(2,3)*sp.sinh(b)**3+sp.Rational(8,15)*sp.sinh(b)**5
    L = sp.simplify(sp.cosh(b)*P-E)
    assert sp.simplify(L-sp.Rational(2,15)*sp.sinh(b)**5) == 0
    assert sp.simplify(sp.diff(L,b)/sp.sinh(b)-P) == 0
    assert sp.simplify(5*E-4*sp.cosh(b)*P-sp.Rational(2,3)*sp.sinh(b)**3) == 0
    a,B0,kap=sp.symbols('a B0 kap',positive=True)
    PT=4*B0/a*(kap*sp.cosh(kap)-sp.sinh(kap))
    ET=2*B0**2*(sp.sinh(kap)*sp.cosh(kap)-kap)
    assert sp.simplify(sp.diff(ET,kap)/sp.diff(PT,kap)-a*B0*sp.sinh(kap)/kap)==0
    assert sp.limit(sp.diff(ET,kap)/sp.diff(PT,kap),kap,0)==a*B0
    mapped=PT.subs(kap,sp.sqrt(3)*b)*a/(6*sp.sqrt(3)*B0)
    assert sp.series(mapped-P,b,0,6).removeO()==-sp.Rational(7,15)*b**5
    # Expand first in k, using the elementary Taylor series, then substitute sinh(b).
    atan = sum((-1)**j*(2*k/sp.sqrt(3))**(2*j+1)/sp.Integer(2*j+1) for j in range(5))
    Fk = sp.series(sp.sqrt(3)/sp.pi*(k*k+sp.Rational(3,4))*atan-3*k/(2*sp.pi),k,0,9).removeO()
    F = sp.series(Fk.subs(k,sp.sinh(b)),b,0,8).removeO()
    Fm = sp.series(F-sp.sinh(b)**3/sp.pi,b,0,8).removeO()
    pac = sp.series(-sp.diff(F,b)/sp.sinh(b),b,0,6).removeO()
    pm = sp.series(-sp.diff(Fm,b)/sp.sinh(b),b,0,6).removeO()
    assert F.coeff(b,7) == sp.Rational(97,1890)/sp.pi
    assert Fm.coeff(b,7) == -sp.Rational(431,7560)/sp.pi
    assert pac.coeff(b,5) == -sp.Rational(8,45)/sp.pi
    assert pac.coeff(b,5) != -sp.Rational(4,15)/sp.pi  # printed error
    assert pm.coeff(b,1) == -1/sp.pi
    assert pm.coeff(b,3) == sp.Rational(10,9)/sp.pi
    assert pm.coeff(b,5) == sp.Rational(2,9)/sp.pi
    Fgen = F+2*c*sp.sinh(b)+sp.Rational(2,3)*lam*sp.sinh(b)**3+c5*sp.sinh(b)**5
    ratio = sp.series(Fgen/P,b,0,3).removeO().expand()
    mom = sp.series(-sp.diff(Fgen,b)/sp.sinh(b),b,0,4).removeO().expand()
    assert ratio.coeff(b,0).coeff(c) == -sp.Rational(5,2)
    assert ratio.coeff(b,2).coeff(c) == sp.Rational(53,40)
    assert mom.coeff(b,1).coeff(c) == -sp.Rational(2,3)
    assert mom.coeff(b,3).coeff(c) == sp.Rational(2,45)
    expected_ratio=3*c/b**2+2/sp.pi+lam-sp.Rational(5,2)*c+(sp.Rational(53,40)*c+sp.Rational(3,2)*c5-lam/2-sp.Rational(23,15)/sp.pi)*b*b
    expected_mom=-2*c/b+(-2*c/3-4/sp.pi-2*lam)*b+(2*c/45-5*c5-4*lam/3-sp.Rational(8,9)/sp.pi)*b**3
    assert sp.simplify(ratio-expected_ratio)==0
    assert sp.simplify(mom-expected_mom)==0
    # Independent formal division of the unfactored Hirota polynomial.
    e,r = sp.symbols('e r', positive=True)
    sh = lambda z: sum(z**(2*j+1)/sp.factorial(2*j+1) for j in range(5))
    ch = lambda z: sum(z**(2*j)/sp.factorial(2*j) for j in range(5))
    polynomial = lambda K,W: W*W-K*K-K**4
    n = sp.Poly(sp.expand(-polynomial(sh(e)-sh(r*e), sh(e)*ch(e)-sh(r*e)*ch(r*e))),e)
    d = sp.Poly(sp.expand(polynomial(sh(e)+sh(r*e), sh(e)*ch(e)+sh(r*e)*ch(r*e))),e)
    a0 = sp.factor(n.nth(4)/d.nth(4))
    a2 = sp.factor((n.nth(6)-a0*d.nth(6))/d.nth(4)/a0)
    a4 = sp.factor((n.nth(8)-a0*d.nth(8)-a0*a2*d.nth(6))/d.nth(4)/a0)
    assert sp.simplify(a2-r) == 0
    assert sp.simplify(a4-(r*r/2-sp.Rational(13,60)*r*(1+r*r))) == 0
    cl,cn,logA=sp.symbols('cl cn logA')
    p1=e*(cl*sp.log(e)+cn)
    p2=r*e*(cl*(sp.log(e)+sp.log(r))+cn)
    H=sp.expand(logA*(-p1/e+p2/(r*e)))
    D=sp.expand(-2*r*p1/e+2*p2/(r*r*e))
    DH=sp.expand(-2*r*p1/e-2*p2/(r*r*e))
    assert sp.simplify(H-cl*logA*sp.log(r))==0
    assert sp.simplify(D-2*(1/r-r)*(cl*sp.log(e)+cn)-2*cl*sp.log(r)/r)==0
    assert sp.simplify(DH+2*(1/r+r)*(cl*sp.log(e)+cn)+2*cl*sp.log(r)/r)==0
    # Higher local one-body terms, with arbitrary analytic coefficient.
    high = sp.series(-sp.diff(c5*sp.sinh(b)**5,b)/sp.sinh(b),b,0,4).removeO()
    assert high == -5*c5*b**3
    d1,d2,t = sp.symbols('d1 d2 t')
    W = sp.Matrix([[0,t,d1,0],[-t,0,0,d2],[-d1,0,0,0],[0,-d2,0,0]])
    assert sp.factor(W.det()) == d1*d1*d2*d2
    # Nonzero canonical brackets do not imply a noncommuting constant projector.
    Q=sp.diag(1,1,0,0); evolution=sp.diag(2,2,3,3)
    assert Q*evolution-evolution*Q == sp.zeros(4)
    cs,ft,fx,fxx=sp.symbols('cs ft fx fxx')
    Lbase=ft**2/2-fx**2/2-fx**3+fxx**2/2
    Lscaled=(cs**3*ft)**2/2-cs**2*(cs**2*fx)**2/2-(cs**2*fx)**3+(cs**3*fxx)**2/2
    assert sp.expand(Lscaled-cs**6*Lbase)==0
    emit('symbolic',dict(F=str(F),Fm=str(Fm),pac=str(pac),pm=str(pm),
                          ratio=str(ratio),momentum=str(mom),A4=str(a4),L0=str(L)))


def F_ac(b):
    # An integral of dF/dk replaces the package's closed arctangent expression.
    k=mp.sinh(b)
    return 2*mp.sqrt(3)/mp.pi*mp.quad(lambda z:z*mp.atan(2*z/mp.sqrt(3)),[0,k])


def F_match(b):
    return F_ac(b)-mp.sinh(b)**3/mp.pi


def momentum(b,s=1):
    return -s*mp.diff(F_match,b)/mp.sinh(b)


def charge(b,s=1):
    k=mp.sinh(b);v=s*mp.cosh(b)
    return 2*k, -2*k*v, mp.mpf(2)/3*k**3*v


def A_raw(b,d,s=1,t=1,good=False):
    sine,cosine=(mp.sin,mp.cos) if good else (mp.sinh,mp.cosh)
    k,l=sine(b),sine(d);w,z=s*k*cosine(b),t*l*cosine(d)
    f=lambda K,W: W*W-K*K+(1 if good else -1)*K**4
    return -f(k-l,w-z)/f(k+l,w+z)


def phase(b,d,s=1,t=1,good=False):
    sine,cosine=(mp.sin,mp.cos) if good else (mp.sinh,mp.cosh)
    sign=-1 if good else 1
    return mp.mpf(2)/9*sign*(s*cosine(3*b)-t*cosine(3*d))*mp.log(A_raw(b,d,s,t,good))+mp.mpf(8)/3*sine(b)*sine(d)*(s*cosine(b)-t*cosine(d))


def shifts(b,d,s=1,t=1):
    M,Pi,P=charge(b,s);N,Q,V=charge(d,t)
    g=mp.diff(lambda z:charge(z,s)[1],b)/mp.diff(lambda z:charge(z,s)[2],b)
    j=mp.diff(lambda z:charge(z,t)[1],d)/mp.diff(lambda z:charge(z,t)[2],d)
    log=mp.log(A_raw(b,d,s,t))
    H=log*(-momentum(b,s)/mp.sinh(b)+momentum(d,t)/mp.sinh(d))
    D=g*N*momentum(b,s)-j*M*momentum(d,t)
    dx1=-log/mp.sinh(b)+g*N;dx2=log/mp.sinh(d)-j*M
    close(-mp.diff(lambda z:phase(z,d,s,t),b)/mp.diff(lambda z:charge(z,s)[2],b),dx1)
    close(-mp.diff(lambda z:phase(b,z,s,t),d)/mp.diff(lambda z:charge(z,t)[2],d),dx2)
    S=-3/(2*mp.pi)*(3*phase(b,d,s,t)+4*P*dx1+4*V*dx2)
    return H,D,S


def continuum_checks(data):
    errors=[]
    for s,t in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        b,d=mp.mpf('.53'),mp.mpf('.21')
        shifts(b,d,s,t)
        plus,minus=b+d,b-d
        fact=(mp.sinh(minus/2)**2/mp.sinh(plus/2)**2*(1+2*mp.cosh(plus))/(1+2*mp.cosh(minus)) if s==t else
              mp.cosh(plus/2)**2/mp.cosh(minus/2)**2*(2*mp.cosh(minus)-1)/(2*mp.cosh(plus)-1))
        close(fact,A_raw(b,d,s,t));assert fact>0
    for row in data['finite_amplitude_bad_beta_grid']:
        b,d=mp.mpf(str(row['beta1'])),mp.mpf(str(row['beta2']))
        for t,key in [(1,'overtaking'),(-1,'head_on')]:
            h,v,ss=shifts(b,d,1,t)
            for field,val in [('delta0',phase(b,d,1,t)),('H12_can_matched',h),('D12_can_matched',v),('sound_ct_delta',ss)]:
                err=abs(val-mp.mpf(str(row[key][field])));errors.append(err)
                assert err < mp.mpf('2e-14')
            hh,vv,sss=shifts(d,b,t,1)
            close(h+hh,0);close(v+vv,0);close(ss+sss,0)
    # Independent good grid check of the classical phase only; no good scheme selected.
    for row in data['finite_amplitude_good_alpha_grid']:
        b,d=mp.mpf(str(row['alpha1'])),mp.mpf(str(row['alpha2']))
        for t,key in [(1,'overtaking'),(-1,'head_on')]:
            close(phase(b,d,1,t,True),mp.mpf(str(row[key]['delta0'])),'2e-14')
    small=[]
    for r in map(mp.mpf,['.25','.4','.5','.6','.75']):
        e=mp.mpf('1e-6');l=2*mp.log((1-r)/(1+r))
        for t in [1,-1]:
            # Unfactored Hirota differences lose digits near k1,k2=0.
            # Preserve the original absolute derivative tolerance with guard digits.
            with mp.workdps(110):
                h,d,s=shifts(e,r*e,1,t)
                ph=phase(e,r*e,1,t)
                ph2=(1-r*r)*l if t==1 else 4*r
                ph4=mp.mpf(3)/4*(1-r**4)*l+mp.mpf(7)/3*r*(1-r*r) if t==1 else mp.mpf(2)/3*r*(1+r*r)
                assert abs(ph-ph2*e**2-ph4*e**4)<100*e**6
                if t==-1:
                    AA=1-3*r*e**2+(mp.mpf(9)/2*r*r+mp.mpf(13)/4*r*(1+r*r))*e**4
                    assert abs(A_raw(e,r*e,1,t)-AA)<1000*e**6
            targets=((-mp.mpf(23)/(18*mp.pi)*(1-r*r)*l,
                      -2/mp.pi*(1/r-r),-(1/r-r**3)/(3*mp.pi),
                      -(1-r*r)*l/(2*mp.pi)) if t==1 else
                     (-6*r/mp.pi,2/mp.pi*(1/r+r),((1/r+r**3)/3-mp.mpf(28)*r/9)/mp.pi,-2*r/mp.pi))
            hc,dl,dc,sc=targets
            close(h/e**2,hc,'1e-9');close((d-dl)/e**2,dc,'1e-9');close(s/e**2,sc,'1e-9')
            if t==-1:close((s-sc*e**2)/e**4,-3*r*(1+r*r)/mp.pi,'1e-8')
            small.append(dict(r=str(r),channel=t,H=h,D=d,sound=s))
    # Direct integration of the profile tangent, with tanh coordinate on finite box.
    b=mp.mpf('.63');k=mp.sinh(b);v=mp.cosh(b);R=mp.mpf(70)
    tangent=lambda x:mp.diff(lambda vv:-mp.sqrt(vv*vv-1)*mp.tanh(mp.sqrt(vv*vv-1)*x/2),v)
    integ=mp.quad(lambda x:tangent(x)**2/2,[-R,-20,0,20,R])
    finite=integ-R*v*v/(k*k)
    target=v*v*(mp.pi**2-6)/(9*k**3)
    close(finite,target,'1e-14')
    # Coordinate-tangent ordering for Omega=dpi wedge dphi.
    profile=lambda x,vv:(vv*vv-1)/(2*mp.cosh(mp.sqrt(vv*vv-1)*x/2)**2)
    pairing=mp.quad(lambda x:-profile(x,v)**2-2*v*profile(x,v)*mp.diff(lambda vv:profile(x,vv),v),[-mp.inf,0,mp.inf])
    close(pairing,-mp.diff(lambda vv:mp.mpf(2)/3*vv*(vv*vv-1)**mp.mpf('1.5'),v))
    # Fourier transform of sech^4 determines its second moment independently.
    T=lambda q:mp.mpf(4)/3*mp.gamma(2+1j*q/2)*mp.gamma(2-1j*q/2)
    close(-mp.diff(T,0,2),(mp.pi**2-6)/9)
    # Stable-band phase differentiated directly, no closed trace integrand imported.
    band=[]
    for bs in ['.19','.47']:
        b=mp.mpf(bs);k=mp.sinh(b);v=mp.cosh(b)
        def integrand(t):
            q=mp.sin(t);w=abs(q)*mp.cos(t)
            def trans(tt):
                qq=mp.sin(tt);ww=abs(qq)*mp.cos(tt)
                pp=lambda K,W:W*W-K*K-K**4
                return -pp(k-1j*qq,k*v-1j*ww)/pp(k+1j*qq,k*v+1j*ww)
            derivative=mp.im(mp.diff(trans,t)/trans(t))
            return (w-v*q)*derivative
        value=mp.quad(integrand,[-mp.pi/2,-mp.pi/3,-mp.pi/6,0,mp.pi/6,mp.pi/3,mp.pi/2])/(4*mp.pi)-3*k/mp.pi
        predicted=F_ac(b)-mp.sqrt(3)/2*(k*k+mp.mpf(3)/4)
        close(value,predicted,'1e-50');band.append(dict(beta=bs,value=value))
    q=mp.mpf('1.3');b=mp.mpf('.7');k=mp.sinh(b);w=q*mp.sqrt(1-q*q)
    pp=lambda K,W:W*W-K*K-K**4
    mod=abs(-pp(k-1j*q,k*mp.cosh(b)-1j*w)/pp(k+1j*q,k*mp.cosh(b)+1j*w))
    assert abs(mod-1)>.5
    emit('continuum',dict(max_grid_error=max(errors),small=small,kinetic_error=abs(finite-target),coordinate_symplectic_pairing=pairing,band=band,unstable_transmission_modulus=mod))


def IK(r):
    # Actual scale-free phonon phase integral with k1/k2=r; q=tan(t).
    return -4/mp.pi*mp.quad(lambda t:mp.atan(r/mp.tan(t)),[0,mp.pi/6,mp.pi/3,mp.pi/2])


def phonon_conventions():
    offsets=[2*mp.pi/3,-2*mp.pi/3,mp.mpf(0)];weights=[1,1,-2]
    # Independent good real-contour phase and a stated formal continuation of p.
    def roots(a):return [(mp.sin(a+x),mp.cos(a+x),w) for x,w in zip(offsets,weights)]
    def theta(q,a,eps):
        return -sum(w*(mp.sign(q)*mp.atan(s/abs(q))+eps*mp.atan(s*c/(abs(q)*mp.sqrt(1+q*q)+q*q+s*s))) for s,c,w in roots(a))
    def deriv(q,a,eps):
        return sum(w*s/(q*q+s*s)*(1+eps*mp.sign(q)*c/mp.sqrt(1+q*q)) for s,c,w in roots(a))
    a=mp.mpf('1e-5');b=a/2
    value=mp.quad(lambda q:(theta(q,a,1)*deriv(q,b,1)+theta(-q,a,1)*deriv(-q,b,1))/(4*mp.pi),[0,b,a,mp.mpf('.1'),1,mp.inf])
    close(value,IK(2),'1e-7')
    assert abs(value-IK(mp.mpf('.5')))>mp.mpf('.8')
    beta=mp.mpf('1e-12')
    raw=sum(w*s*mp.atanh(c) for s,c,w in roots(-1j*beta))/(2*mp.pi)
    continued=mp.re(raw/(-1j))
    constant=continued/beta-mp.log(beta)/mp.pi
    target=(-mp.log(2)-1+mp.log(3)/4)/mp.pi
    close(constant,target,'1e-20')
    emit('phonon_conventions',dict(good_I_larger_first=value,IK_inverse_ratio=IK(2),IK_package_ratio=IK(mp.mpf('.5')),mode_linear_coefficient=constant))


def chi(r):
    return mp.quad(lambda t:mp.atanh(t)/t,[0,r])


def ratios(cont,toda):
    rows=[]
    for row in cont['ratio_grid_small_amplitude']:
        r=mp.mpf(row['r_str']);i=IK(r);ch=chi(r)
        close(i,-4*ch/mp.pi-2/mp.pi*mp.log(r)*mp.log((1-r)/(1+r)))
        close(i+IK(1/r),-mp.pi)
        can=mp.pi/2+i+2/mp.pi*(r-1/r)
        target=mp.pi-4*ch/mp.pi+2/mp.pi*(r-1/r)
        difference=can-target
        close(difference,-mp.pi/2-2/mp.pi*mp.log(r)*mp.log((1-r)/(1+r)))
        assert abs(difference)>mp.pi/2
        close(can,mp.mpf(row['overtaking']['delta1_ov_J_plus_D_can_matched_str']),'1e-33')
        # Toda c0 independently integrated in the t-variable of its regular convolution.
        forward=-1/mp.pi*mp.quad(lambda t:mp.sqrt(t)/(1-r*r+t)**mp.mpf('1.5')*mp.log((1-r*r+t)*r*r/t),[0,r*r])
        close(forward,-4/mp.pi*(ch-r))
        rows.append(dict(r=r,I=i,J=mp.pi/2+i,D=2/mp.pi*(r-1/r),canonical=can,Toda_limit=target,difference=difference,head_can=2/mp.pi*(r+1/r),forward=forward))
    # Common-point comparison uses stated definitions, never calls the two objects equal.
    grid=[]
    for row in toda['finite_sea_grid']:
        r=mp.sqrt(mp.mpf(str(row['y2']))/mp.mpf(str(row['y1'])))
        matches=[x for x in cont['fermi_edge_Y_r_grid'] if abs(x['Y']-row['y1'])<1e-12 and abs(x['r']-float(r))<1e-12]
        assert len(matches)==1
        is_ov=row['s2']==1
        val=mp.pi/2+IK(r)+2/mp.pi*(r-1/r) if is_ov else 2/mp.pi*(r+1/r)
        q=row['quantities']['R_antisymmetric']
        tv=mp.mpf(q['value']);err=mp.mpf(q['total_error'])
        assert abs(val-tv)>mp.mpf('.1')+err
        grid.append(dict(Y=row['y1'],r=r,channel='ov' if is_ov else 'ho',continuum=val,Toda=tv,difference=val-tv,h_error=q['h_error'],mesh_error=q['mesh_error']))
    # A concurrently supplied Toda audit confirms the same closed formula but
    # its printed r=0.5 decimal (if still present) is not that formula's value.
    other=HERE.parent/'audit_twobody_toda.tex'
    if other.exists() and '2.1670489477' in other.read_text():
        at_half=next(x for x in rows if x['r']==mp.mpf('.5'))['Toda_limit']
        assert abs(at_half-mp.mpf('2.1670489477'))>mp.mpf('.6')
        RESULT['concurrent_toda_audit_decimal_discrepancy']=at_half-mp.mpf('2.1670489477')
    emit('ratios',rows);emit('common_finite_sea_grid',grid)


def printed_tables(cont):
    text=(HERE.parent/'twobody_cont.tex').read_text()
    table=text.split('\\label{tab:finite_beta}')[1].split('\\end{table}')[0]
    mismatches=[]
    for line in table.splitlines():
        if not re.match(r'^\$\(0\.',line):continue
        cells=line.split('&')
        nums=re.findall(r'[0-9]+\.[0-9]+',cells[0]);b,d=map(mp.mpf,nums)
        saved=next(x for x in cont['finite_amplitude_bad_beta_grid'] if abs(x['beta1']-float(b))<1e-12 and abs(x['beta2']-float(d))<1e-12)
        for j,channel in enumerate(['overtaking','head_on']):
            for i,field in enumerate(['delta0','H12_can_matched','D12_can_matched','delta_renorm_can_matched']):
                raw=re.search(r'[-+]?\d+\.\d+',cells[1+4*j+i]).group()
                err=mp.mpf(raw)-mp.mpf(str(saved[channel][field]))
                if abs(err)>mp.mpf('0.00000051'):
                    mismatches.append(dict(beta=str(b),channel=channel,field=field,printed=raw,json=saved[channel][field],difference=err))
    assert len(mismatches)>20
    row=cont['ratio_grid_small_amplitude'][0]['overtaking']
    serialization_error=abs(IK(mp.mpf('.1'))-mp.mpf(str(row['I_KdV'])))
    assert serialization_error>mp.mpf('1e-50')
    ratio_mismatches=[]
    table=text.split('\\label{tab:ratio_grid}')[1].split('\\end{table}')[0]
    for line in table.splitlines():
        if not re.match(r'^\$0\.',line):continue
        cells=line.split('&');r=mp.mpf(re.search(r'0\.\d+',cells[0]).group())
        i=IK(r);d=2/mp.pi*(r-1/r)
        exact=[2*(1-r*r)*mp.log((1-r)/(1+r)),i,mp.pi/2+i,-4*chi(r)/mp.pi,d,i+d,mp.pi/2+i+d,2/mp.pi*(r+1/r)]
        for j,value in enumerate(exact):
            printed=mp.mpf(re.search(r'[-+]?\d+\.\d+',cells[j+1]).group())
            if abs(printed-value)>mp.mpf('0.00000051'):
                ratio_mismatches.append(dict(r=r,column=j+1,printed=printed,correct=value))
    assert len(ratio_mismatches)>10
    emit('printed_table',dict(mismatches=mismatches,ratio_mismatches=ratio_mismatches,serialized_float_error=serialization_error))


def rectangle():
    # Four independent corner evaluations, including the omitted next Jacobian term.
    z,C,B=sp.symbols('z C B',positive=True)
    factors=[(2,sp.Rational(1,2),1),(1,sp.Rational(1,2),-1),
             (2,sp.Rational(1,4),-1),(1,sp.Rational(1,4),1)]
    fun=lambda a:sp.sqrt(a)/z+C*z/sp.sqrt(a)+B*z**3/a**sp.Rational(3,2)
    rec=sp.expand(sum(s*8*(fun(a)*fun(b)-sp.sqrt(a*b)/z**2) for a,b,s in factors))
    lead=sp.simplify(rec.coeff(z,0));sub=sp.simplify(rec.coeff(z,2))
    assert sp.simplify(lead+2*C*(15*sp.sqrt(2)-20))==0
    assert sp.simplify(sub-8*C*C*(sp.sqrt(2)-1)**2-(102-85*sp.sqrt(2))*B)==0
    # u'=1-C/y already provides a counterexample to the printed 1/Y coefficient:
    # inversion necessarily gives B=C^2. For Toda u'=1-C/y-d0/y^2, B=1/48.
    assert sub.coeff(B)!=0
    # The stated O(y^-3/2) Jacobian remainder permits a Y^-1/2 term.
    T=sp.symbols('T')
    ff=lambda a:sp.sqrt(a)/z+C*z/sp.sqrt(a)+T*z*z/a
    half=sp.expand(sum(s*8*(ff(a)*ff(b)-sp.sqrt(a*b)/z**2) for a,b,s in factors))
    assert sp.simplify(half.coeff(z,1))!=0
    Cn=1/(2*mp.pi);lim=-(15*mp.sqrt(2)-20)/mp.pi
    coef=8*Cn**2*(mp.sqrt(2)-1)**2+(102-85*mp.sqrt(2))/48
    fa=json.loads((HERE/'frame_analysis.json').read_text())
    tab=[dict(Y=x['Y'],saved=x['value'],edge_spread=x['edge_spread'],leading=lim,
              corrected_two_terms=lim+coef/mp.mpf(str(x['Y']))) for x in fa['rectangles']]
    assert abs(mp.mpf(str(tab[-1]['saved']))-lim)<mp.mpf('.0217')
    emit('rectangle',dict(limit=lim,coefficient=sub,correct_Toda_1Y=coef,table=tab))


def free_fits(toda):
    # New unconstrained polynomial windows; all input floats promoted honestly.
    # Fit only saved data; independent fresh edge solver below tests the underlying values.
    results=[]
    for rr in ['0.4','0.5','0.6']:
        points=sorted([x for x in toda['halfline_grid'] if abs(mp.mpf(x['r'])-mp.mpf(rr))<mp.mpf('1e-12')],key=lambda x:mp.mpf(x['y1']))
        variants=[]
        for count,degree in [(5,2),(6,3),(4,1)]:
            pts=points[-count:];Ymax=mp.mpf(pts[-1]['y1'])
            design=mp.matrix([[(Ymax/mp.mpf(x['y1']))**j for j in range(degree+1)] for x in pts])
            vals=mp.matrix([mp.mpf(x['R_antisymmetric']) for x in pts])
            fit,_=mp.qr_solve(design,vals)
            variants.append(dict(window=count,degree=degree,intercept=fit[0]))
        target=mp.pi-4*chi(mp.mpf(rr))/mp.pi+2/mp.pi*(mp.mpf(rr)-1/mp.mpf(rr))
        results.append(dict(r=rr,variants=variants,target=target,spread=max(v['intercept'] for v in variants)-min(v['intercept'] for v in variants)))
    # Refit raw finite-h data independently with quadratic last five and cubic last seven.
    raw=json.loads((HERE/'twobody_toda_scan.json').read_text())['rows']
    raw=sorted([x for x in raw if x['factor']==1],key=lambda x:x['h'])
    fgrid=[]
    for index,p in enumerate(raw[0]['pairs']):
        vals=[]
        for n,degree in [(5,2),(7,3)]:
            pts=raw[:n];hmax=mp.mpf(str(pts[-1]['h']))
            X=mp.matrix([[(mp.mpf(str(x['h']))/hmax)**j for j in range(degree+1)] for x in pts])
            V=mp.matrix([mp.mpf(str(x['pairs'][index]['R_antisymmetric'])) for x in pts])
            coef,_=mp.qr_solve(X,V);vals.append(coef[0])
        fgrid.append(dict(y1=p['y1'],y2=p['y2'],s2=p['s2'],quad5=vals[0],cubic7=vals[1],spread=abs(vals[0]-vals[1])))
    emit('independent_amplitude_fits',results);emit('independent_h_fits',fgrid)


def edge_checks(cont,toda):
    """de Hoog inversion + nested Clenshaw-Curtis, versus package Talbot/Gauss.

    No fitted intercept, Toda output or continuum target enters these solves.
    Degree 32 inverse transforms, 35-digit arithmetic; precision checked separately.
    """
    with mp.workdps(35):
        def g(s):
            return mp.exp(mp.loggamma(1+s/(2*mp.pi))-mp.log(s)/2-s/(2*mp.pi)*mp.log(s/(2*mp.pi*mp.e)))
        def inv(u,power,degree=32):
            if u==0:
                return {0:mp.pi**mp.mpf('1.5')/6,1:mp.sqrt(mp.pi),2:mp.mpf(0)}[power]
            # f has a contact delta; subtracting one avoids it explicitly.
            transform=lambda s:mp.sqrt(mp.pi)*(g(s)-1 if power==0 else g(s)/s**power)
            return mp.invertlaplace(transform,u,method='dehoog',degree=degree)
        @lru_cache(None)
        def solve(ystr):
            y=mp.mpf(ystr);goal=mp.mpf(4)/3*y**mp.mpf('1.5')
            u=max(mp.mpf('.001'),y-mp.log(1+y)/(2*mp.pi)-mp.mpf('.2'))
            for _ in range(12):
                res=inv(u,2)-goal
                if abs(res)<mp.mpf('1e-22'):break
                step=res/inv(u,1)
                u=max(u/2,u-step)
            assert abs(inv(u,2)-goal)<mp.mpf('1e-20')
            return u,inv(u,1)
        def weights(n):
            # Interpolatory Chebyshev-Lobatto moments, not Legendre quadrature.
            nodes=[(1-mp.cos(mp.pi*j/n))/2 for j in range(n+1)]
            A=mp.matrix([[mp.cos(mp.pi*j*k/n) for j in range(n+1)] for k in range(n+1)])
            moments=mp.matrix([0 if k%2 else 1/(mp.mpf(1)-k*k) for k in range(n+1)])
            return nodes,list(mp.lu_solve(A,moments))
        rules={n:weights(n) for n in [24,48]}
        rows=[]
        for saved in toda['halfline_grid']:
            Y=mp.mpf(saved['y1']);r=mp.mpf(saved['r'])
            if not any(abs(mp.mpf(str(x['Y']))-Y)<mp.mpf('1e-12') and abs(mp.mpf(str(x['r']))-r)<mp.mpf('1e-12') for x in cont['fermi_edge_Y_r_grid']):continue
            begin=time.monotonic();u,B1=solve(str(Y));v,B2=solve(str(Y*r*r))
            cache={}
            def integrand(x):
                # w=v*x^2 resolves the origin and retains the B(0+) endpoint.
                key=mp.nstr(x,28)
                if key not in cache:cache[key]=4*v*x*inv(u-v*x*x,0)*inv(v*(1-x*x),1)
                return cache[key]
            vals=[sum(w*integrand(x) for x,w in zip(*rules[n])) for n in [24,48]]
            initial_difference=abs(vals[1]-vals[0]);nodes=48
            if initial_difference>=mp.mpf('2e-7'):
                if 96 not in rules:rules[96]=weights(96)
                vals.append(sum(w*integrand(x) for x,w in zip(*rules[96])))
                nodes=96
            H=vals[-1];quadrature=abs(vals[-1]-vals[-2])
            Hcl=4*r*Y+2*Y*(1-r*r)*mp.log((1-r)/(1+r))
            forward=H-Hcl
            symmetric=B1*B2-mp.pi-4*r*Y
            anti=forward-symmetric
            discrepancy=abs(anti-mp.mpf(saved['R_antisymmetric']))
            assert quadrature<mp.mpf('2e-7'),(Y,r,quadrature)
            assert discrepancy<mp.mpf('2e-8'),(Y,r,discrepancy)
            row=dict(Y=Y,r=r,u=u,v=v,R12=forward,anti=anti,quad_difference=quadrature,initial_24_48_difference=initial_difference,nodes=nodes,saved_difference=discrepancy,seconds=time.monotonic()-begin)
            rows.append(row);print('edge',json.dumps(row,default=str),flush=True)
        precision=[]
        for u in [mp.mpf('.2'),mp.mpf('5'),mp.mpf('50')]:
            err=max(abs(inv(u,k,32)-inv(u,k,48)) for k in [0,1,2])
            assert err<mp.mpf('1e-22');precision.append(err)
        rect=[]
        for Y in [4,8,16,32]:
            B={j:solve(str(mp.mpf(j)))[1] for j in [2*Y,Y,Y/2,Y/4]}
            value=2*(B[2*Y]-B[Y])*(B[Y/2]-B[Y/4])-8*(mp.sqrt(2*Y)-mp.sqrt(Y))*(mp.sqrt(mp.mpf(Y)/2)-mp.sqrt(mp.mpf(Y)/4))
            rect.append(dict(Y=Y,value=value))
        emit('independent_edge',dict(rows=rows,precision_differences=precision,rectangles=rect))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-edge',action='store_true');parser.add_argument('--reuse-edge',action='store_true');args=parser.parse_args()
    cont=json.loads((HERE/'twobody_cont_grid.json').read_text())
    toda=json.loads((HERE/'twobody_toda_grid.json').read_text())
    symbolic();continuum_checks(cont);phonon_conventions();ratios(cont,toda);printed_tables(cont);rectangle();free_fits(toda)
    if args.reuse_edge:
        previous=json.loads((HERE/'audit_twobody_cont_results.json').read_text())
        assert previous['complete_edge_run'] and previous['input_hashes']==HASHES
        RESULT['independent_edge']=previous['independent_edge']
        RESULT['edge_run_reused_from_runtime_seconds']=previous['runtime_seconds']
    elif not args.skip_edge:edge_checks(cont,toda)
    assert HASHES=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}
    RESULT['input_hashes']=HASHES
    RESULT['runtime_seconds']=time.monotonic()-START
    RESULT['complete_edge_run']=not args.skip_edge
    (HERE/'audit_twobody_cont_results.json').write_text(json.dumps(RESULT,indent=2,default=str)+'\n')
    print('runtime_seconds',RESULT['runtime_seconds'])
    print('ALL CHECKS PASSED')


if __name__=='__main__':
    main()

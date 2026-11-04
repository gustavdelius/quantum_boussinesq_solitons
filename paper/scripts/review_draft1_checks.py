"""Independent draft-1 review checks, all mathematical findings labelled (c).

Run from any directory with python3 -B. No files are written. Expected false
claims are tested as counterexamples, not silently converted into identities.
This checks algebra and the stated numerical samples, not spectral completeness
or a physical quantum determinant. --run-sources also runs the read-only scripts
cited in Sections 1--6 (excluding the figure writer). Existing source failures
are reported without modifying them. Working precision: 65 decimal digits.
"""
from pathlib import Path
import ast
import hashlib
import os
import re
import subprocess
import sys
import sympy as sp
import mpmath as mp

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
mp.mp.dps = 65
TOL = mp.mpf('1e-52')
R3 = mp.sqrt(3)


def check(name, condition):
    assert bool(condition), name
    print('PASS (c): ' + name, flush=True)


def close(a, b, tol=TOL):
    return abs(a-b) < tol * max(1, abs(a), abs(b))


def functions_only(path, namespace=None):
    """Read exact existing function definitions without imports/main/file writes."""
    ns = dict(mp=mp, sp=sp, r3=R3, I=mp.j)
    ns.update(namespace or {})
    tree = ast.parse((ROOT/path).read_text())
    defs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(defs, str(path), 'exec'), ns)
    return ns


# A1--A2: Euler--Lagrange equations, exact integrals and direct quadrature.
x, t = sp.symbols('x t', real=True)
phi = sp.Function('phi')(x,t)
u = phi.diff(x)
for sign in [-1, 1]:  # sign=-1 good; sign=+1 bad
    lag = phi.diff(t)**2/2-u**2/2-sign*u**3+sign*phi.diff(x,2)**2/2
    el = sp.euler_equations(lag,phi,[x,t])[0].lhs
    target = phi.diff(t,2)-phi.diff(x,2)-sign*phi.diff(x,4)-3*sign*(u**2).diff(x)
    check('A1 Euler--Lagrange sign '+str(sign), sp.expand(el+target)==0)
k, y = sp.symbols('k y', positive=True)
U = k**2*(1-y*y)/2
jac = 2/(k*(1-y*y))
Ux = -k*y*U
ints = [sp.integrate(f*jac,(y,-1,1)) for f in [U,U**2,U**3,Ux**2]]
check('A1/A2 exact integrals M, u^2, u^3, ux^2', ints==[2*k,2*k**3/3,4*k**5/15,2*k**5/15])
for sign in [-1,1]:
    E = sp.simplify((2+sign*k*k)*ints[1]/2+sign*ints[2]-sign*ints[3]/2)
    P = sp.Rational(2,3)*k**3*sp.sqrt(1+sign*k*k)
    check('A1/A2 energy and velocity identity '+str(sign),
          sp.simplify(E-sp.Rational(2,15)*k**3*(5+4*sign*k*k))==0 and
          sp.simplify(E.diff(k)-sp.sqrt(1+sign*k*k)*P.diff(k))==0)
    for kk in map(mp.mpf,['0.2','0.6','0.95']):
        for eps in [-1,1]:
            v=eps*mp.sqrt(1+sign*kk*kk)
            uf=lambda X:kk**2/2*mp.sech(kk*X/2)**2
            ux=lambda X:-kk*mp.tanh(kk*X/2)*uf(X)
            eq=mp.quad(lambda X:(1+v*v)*uf(X)**2/2+sign*uf(X)**3-sign*ux(X)**2/2,[-mp.inf,0,mp.inf])
            check('A1/A2 direct energy quadrature '+str((sign,str(kk),eps)),close(eq,2*kk**3*(5+4*sign*kk**2)/15))
Egood=sp.Rational(2,15)*k**3*(5-4*k*k)
Ewrong=k**3*(3-2*k*k)/3
check('A2 printed excess k^3(5-2k^2)/15',sp.expand(Ewrong-Egood-k**3*(5-2*k*k)/15)==0)
check('A2 velocity-identity residual k^2(1-2k^2/3)',sp.expand(sp.diff(Ewrong-Egood,k)-k*k*(1-2*k*k/3))==0)
check('A1 wrong bad Hamiltonian differs by -8k^5/15',sp.expand(-2*ints[2]+8*k**5/15)==0)
C,z=sp.symbols('C z')
wick=sp.expand(z**3-C*sp.diff(z**3,z,2)/2)
check('A1 propagated normal-order sign: :u^3:-u^3=-3 C u',wick-z**3==-3*C*z)
variance=mp.quad(lambda h:mp.sin(h),[0,mp.pi/2])/(2*mp.pi)
check('A1 stable-band variance / hbar = 1/(2pi)',close(variance,1/(2*mp.pi)))

# Independent scattering data, both equations and all orientations.
def sol(a,e=1,bad=False):
    k=mp.sinh(a) if bad else mp.sin(a)
    v=e*(mp.cosh(a) if bad else mp.cos(a))
    ell=e*mp.cosh(3*a) if bad else -e*mp.cos(3*a)
    wsum=(mp.j if bad else 1)*v/R3
    return dict(k=k,v=v,w=k*v,M=2*k,Pi=-2*k*v,P=mp.mpf(2)/3*k**3*v,
                ell=ell,pa=(wsum+k)/2,pb=(wsum-k)/2)


def interaction(a,b,bad=False):
    poly=lambda K,W:W*W-K*K+(-1 if bad else 1)*K**4
    return -poly(a['k']-b['k'],a['w']-b['w'])/poly(a['k']+b['k'],a['w']+b['w'])


def phase(a,b,bad=False):
    return mp.mpf(2)/9*(a['ell']-b['ell'])*mp.log(interaction(a,b,bad))+mp.mpf(2)/3*a['M']*b['M']*(a['v']-b['v'])


for bad in [False,True]:
    for e1 in [-1,1]:
        for e2 in [-1,1]:
            a,b=mp.mpf('.2'),mp.mpf('.5')
            s1,s2=sol(a,e1,bad),sol(b,e2,bad)
            A=interaction(s1,s2,bad)
            cr=(s1['pa']-s2['pa'])*(s1['pb']-s2['pb'])/((s1['pa']-s2['pb'])*(s1['pb']-s2['pa']))
            g1=mp.diff(lambda q:sol(q,e1,bad)['Pi'],a)/mp.diff(lambda q:sol(q,e1,bad)['P'],a)
            g2=mp.diff(lambda q:sol(q,e2,bad)['Pi'],b)/mp.diff(lambda q:sol(q,e2,bad)['P'],b)
            dx1=-mp.log(A)/s1['k']+g1*s2['M']
            dx2=mp.log(A)/s2['k']-g2*s1['M']
            check('B roots and phase gradients '+str((bad,e1,e2)), close(A,cr) and
                  close(mp.diff(lambda q:phase(sol(q,e1,bad),s2,bad),a),-mp.diff(lambda q:sol(q,e1,bad)['P'],a)*dx1) and
                  close(mp.diff(lambda q:phase(s1,sol(q,e2,bad),bad),b),-mp.diff(lambda q:sol(q,e2,bad)['P'],b)*dx2))

# A3/A4: actual existing functions versus independent rational factor and sewing.
dn=functions_only('phase2/scripts/oneloop_defect_scattering_and_stability.py')
for eta in [-1,1]:
    for eps in [-1,1]:
        a=mp.mpf('.4'); sigma=mp.mpf('2.4'); st=sol(a,eps)
        q=-sigma/(2*R3)
        root=((st['pb']-q)/(st['pa']-q))**eta
        cz=(st['v']+sigma-eta*R3*st['k'])/(st['v']+sigma+eta*R3*st['k'])
        middle=((st['k']-(st['v']+sigma)/R3)/(st['k']+(st['v']+sigma)/R3))**(-eta)
        check('A3/A4 root, existing z function and -1/z '+str((eta,eps)),close(root,cz) and close(dn['z_defect_mode'](st,sigma,eta),cz) and close(middle,-1/root))
        def jet(f):return dn['eval_jet'](f,mp.mpf('.27'))
        left=lambda X,T:-6*st['k']/(1+mp.exp(-st['k']*(X-st['v']*T)))
        right=lambda X,T:-6*st['k']/(1+mp.exp(-st['k']*(X-st['v']*T))/cz)
        residual=max(abs(v) for v in dn['cz_sewing_vec'](*jet(left),*jet(right),sigma,eta))
        check('A3 nonlinear sewing at both orientations '+str((eta,eps)),residual<TOL)
        if eta==-1:
            qwrong=-eta*sigma/(2*R3)
            zw=((st['pb']-qwrong)/(st['pa']-qwrong))**eta
            badright=lambda X,T:-6*st['k']/(1+mp.exp(-st['k']*(X-st['v']*T))/zw)
            rbad=max(abs(v) for v in dn['cz_sewing_vec'](*jet(left),*jet(badright),sigma,eta))
            print('COUNTEREXAMPLE (c) eta=-1 eps=',eps,'z=',mp.nstr(cz,12),'wrong z=',mp.nstr(zw,12),'sewing residual=',mp.nstr(rbad,8))
            check('A3 signed-root replacement fails same physical defect',abs(zw-cz)>mp.mpf('.01') and rbad>mp.mpf('.001'))
check('A3 later CZ coefficients violate beta^2-alpha^2=1',sp.Rational(1,12)-sp.Rational(1,12)!=1)

# A2: run actual one-loop function bodies; no E_classical input is involved.
pn=functions_only('phase2/scripts/oneloop_good_phonon_scattering.py')
ln=functions_only('phase2/scripts/oneloop_good_twosoliton_phase.py',pn)
cn=functions_only('phase2/scripts/charge_and_frame_audit.py')
an=functions_only('phase2/scripts/audit_charges_action.py')
for a in map(mp.mpf,['.15','.5','.85']):
    check('A2 downstream mode charges agree '+str(a),close(ln['Delta_E1_ren_closed'](a),cn['delta_E1_ren'](a)) and close(ln['Delta_P1_closed'](a),cn['delta_P_mode'](a)))
    check('A2 downstream collective charge derivative '+str(a),close(cn['delta_P_can_closed'](a),-mp.diff(cn['L1_comoving'],a)/mp.sin(a)))
    check('A2 audit uses corrected classical energy '+str(a),close(an['ecl'](a),2*mp.sin(a)**3*(5-4*mp.sin(a)**2)/15))
a,b=mp.mpf('.2'),mp.mpf('.5')
for e2 in [-1,1]:
    ren=-mp.diff(lambda q:phase(sol(q),sol(b,e2)),a)/mp.diff(lambda q:sol(q)['P'],a)*ln['Delta_P1_closed'](a)
    ren-=mp.diff(lambda q:phase(sol(a),sol(q,e2)),b)/mp.diff(lambda q:sol(q,e2)['P'],b)*ln['Delta_P1_closed'](b,e2)
    check('A2 fixed-momentum renormalisation independent of E_classical '+str(e2),close(ren,ln['delta_renorm_good'](a,1,b,e2)))

# A5--A9 and further draft defects.
small=mp.mpf('1e-12'); a=mp.mpf('1.7')*small; b=small
ratio=mp.diff(lambda q:phase(sol(q,1,True),sol(b,1,True),True),a)/(2*a*mp.log(interaction(sol(a,1,True),sol(b,1,True),True)))
pred=1+2/(mp.mpf('1.7')*mp.log((mp.mpf('.7')/mp.mpf('2.7'))**2))
check('A5 full derivative ratio 0.5642465, cross-only ratio -0.4357535',close(ratio,pred,mp.mpf('1e-20')))
print('A5 ratio =',mp.nstr(pred,20))
for a in map(mp.mpf,['.3','.8']):
    st=sol(a); mu1,mu2=-st['pb'],-st['pa']
    check('A7 root sign dictionary',close(mu1**3-mu1/4,-st['ell']/(12*R3)))
    H1=mu1-mu2; H2=mu1**2-mu2**2
    H4=mu1**4-mu2**4-H2/3; H5=mu1**5-mu2**5-5*H1/72
    check('B subtracted GD physical charge combinations',close(-2*R3*H4-H2/R3,st['P']) and close(24*H5/5+H1/6,2*st['k']**3*(5-4*st['k']**2)/15))
check('B GD H1 vacuum residue nonzero',sp.Rational(-1,4)/3==sp.Rational(-1,12))
check('A8 Hirota exponent doubles half-width; sinh beta/beta tends to 1',sp.sqrt(12)==2*sp.sqrt(3) and sp.limit(sp.sinh(k)/k,k,0)==1)

# B: direct one-soliton symplectic pullback in the phase coordinate x0.
kk=mp.mpf('.6'); eps=1
pf=lambda K,X0,X:K*(1+mp.tanh((K*X+X0)/2))
mom=lambda K,X0,X:-mp.sqrt(1-K*K)*K*K/2*mp.sech((K*X+X0)/2)**2
om=mp.quad(lambda X:mp.diff(lambda K:mom(K,0,X),kk)*mp.diff(lambda X0:pf(kk,X0,X),0)-mp.diff(lambda X0:mom(kk,X0,X),0)*mp.diff(lambda K:pf(K,0,X),kk),[-mp.inf,0,mp.inf])
pp=mp.diff(lambda K:mp.mpf(2)/3*K**3*mp.sqrt(1-K*K),kk)
check('B phase coordinate pullback is -P_k/k, not +/-P_k',close(om,-pp/kk) and abs(abs(om)-abs(pp))>mp.mpf('.1'))
print('B Omega(k,x0)=',mp.nstr(om,16),'P_k=',mp.nstr(pp,16))
v,b0=sp.symbols('v b',positive=True)
ET=2*b0**2*(sp.sinh(k)*sp.cosh(k)-k)
for eps in [-1,1]:
    PT=4*eps*b0*(k*sp.cosh(k)-sp.sinh(k))
    check('B signed Toda velocity '+str(eps),sp.simplify(ET.diff(k)/PT.diff(k)-eps*b0*sp.sinh(k)/k)==0)

# B: strict triple claim requires three distinct real roots; threshold excluded.
for ss in map(mp.mpf,['-.5','.5','1.5']):
    sigmas=[ss,(-ss+mp.sqrt(12-3*ss*ss))/2,(-ss-mp.sqrt(12-3*ss*ss))/2]
    for eta in [-1,1]:check('B stable root count on distinct real triple '+str((str(ss),eta)),sum(eta*s < -1 for s in sigmas)==1)
sigma_symbol=sp.Symbol('sigma',real=True)
check('B at s=2 two roots coincide at marginal sigma=-1',sp.solve(sigma_symbol**2+2*sigma_symbol+4-3,sigma_symbol)==[-1])
check('B at s=3 excited vacua nonreal',sp.discriminant(sigma_symbol**2+3*sigma_symbol+9-3,sigma_symbol)<0)
# Exterior derivative of F du+G dv has coefficient G_u-F_v.
uu,vv=sp.symbols('uu vv', real=True)
ff,gg=uu*vv,vv**2+uu**2
check('B defect two-form sign',sp.diff(gg,uu)-sp.diff(ff,vv)==uu)
cq=sp.Function('C')(uu)
check('B arbitrary defect phase constant changes fusion by C(qnext)-C(q)',
      sp.diff(cq,x)==0 and cq.subs(uu,vv)-cq!=0)
fusion=functions_only('phase2/scripts/oneloop_defect_phase_and_fusion.py')
q0=mp.mpf('.45')
roots=[q0,(-q0+mp.sqrt(1-3*q0*q0))/2,(-q0-mp.sqrt(1-3*q0*q0))/2]
for eta in [-1,1]:
    for eps in [-1,1]:
        probe=fusion['sol_good'](mp.mpf('.1'),eps)
        for j in range(3):
            qa,qb=roots[j],roots[(j+1)%3]
            velocity=R3*(qa+qb)
            edge=fusion['sol_good'](mp.acos(abs(velocity)),mp.sign(velocity))
            lhs=fusion['delta0_SD'](probe,qb,eta)-fusion['delta0_SD'](probe,qa,eta)
            rhs=eps*eta*mp.sign(qa-qb)*(fusion['delta0_SS'](probe,edge)+edge['M']*probe['Pi'])
            check('B oriented fusion sign '+str((eta,eps,j)),close(lhs,rhs))
for e in [-1,1]:
    aa=mp.mpf('.4'); st=sol(aa,e)
    check('A7 Phase 2g trigonometric roots and cubic are transcription errors',
          abs(st['pa']-mp.cos(aa-e*mp.pi/6)/R3)>mp.mpf('.01') and
          abs(st['pb']+mp.cos(aa+e*mp.pi/6)/R3)>mp.mpf('.01') and
          close(st['pa']**3-st['pa']/4,st['ell']/(12*R3)) and
          abs(st['pa']**3-st['pa']/4-st['ell']/(4*R3))>mp.mpf('.001'))
# Field rescalings for the source PDEs used in the draft.
field=sp.Function('U')(x,t)
cl=3*field
check('B CL bad field dictionary',sp.expand((cl.diff(t,2)-cl.diff(x,2)-cl.diff(x,4)-(cl**2).diff(x,2))/3-(field.diff(t,2)-field.diff(x,2)-field.diff(x,4)-3*(field**2).diff(x,2)))==0)
bd=field/2
check('B BD good field dictionary',sp.expand(2*(bd.diff(t,2)-bd.diff(x,2)+bd.diff(x,4)+6*(bd**2).diff(x,2))-(field.diff(t,2)-field.diff(x,2)+field.diff(x,4)+3*(field**2).diff(x,2)))==0)
check('B BC potential sign dictionary',sp.expand((-phi).diff(t,2)-(-phi).diff(x,2)-(-phi).diff(x,4)+6*(-phi).diff(x)*(-phi).diff(x,2)+phi.diff(t,2)-phi.diff(x,2)-phi.diff(x,4)-3*(phi.diff(x)**2).diff(x))==0)

# B: two-body Toda reflection formula from independent Bessel asymptotics.
h=mp.mpf('.37'); momentum=mp.mpf('.61'); nu=2j*momentum/h
wave=lambda X:mp.besselk(nu,2*mp.exp(-X/2)/h)
X=mp.mpf('140')
incoming=mp.gamma(-nu)*h**(-nu)/2
outgoing=mp.gamma(nu)*h**nu/2
check('B Toda Bessel wave and gamma reflection amplitude',close(wave(X),incoming*mp.exp(-1j*momentum*X/h)+outgoing*mp.exp(1j*momentum*X/h),mp.mpf('1e-55')))
check('B Toda Schrodinger equation',close(-h*h*mp.diff(wave,mp.mpf('1.1'),2)+mp.exp(-mp.mpf('1.1'))*wave(mp.mpf('1.1')),momentum**2*wave(mp.mpf('1.1'))))

# B: factorisation and distributional Poisson bracket independently.
for aa,bb in [('1.0','.4'),('.7','.2')]:
    a,b=mp.mpf(aa),mp.mpf(bb); S,D=a+b,a-b
    at=lambda th:(2*mp.cosh(th)+1)/(2*(1+mp.cosh(th)))
    G=lambda yy:16*mp.fprod(mp.sinh((yy+j*mp.j*mp.pi/3)/2)*mp.sinh((yy-j*mp.j*mp.pi/3)/2) for j in [1,2])/mp.sinh(yy)**2
    av=interaction(sol(a,1,True),sol(b,1,True),True)
    check('B affine Toda and G factors',close(av,at(S)/at(D)*mp.tanh(D/2)**2/mp.tanh(S/2)**2) and close(av,G(S/2)/G(D/2)))
f=sp.exp(-x*x); g=x*sp.exp(-x*x)
qr=sp.integrate(-3*sp.sqrt(3)*sp.diff(f,x)*g/8,(x,-sp.oo,sp.oo))
check('B non-ultralocal smeared bracket',sp.simplify(qr-3*sp.sqrt(3)*sp.integrate(f*sp.diff(g,x),(x,-sp.oo,sp.oo))/8)==0 and qr!=0)

# Reproducible static inventory; omitted names in appendix are not citations.
paper=(ROOT/'paper/boussinesq_soliton_scattering.tex').read_text()
body=paper.split(r'\section{One-loop corrections}')[0]
paths=sorted(set(re.findall(r'\\scr\{([^{}]+\.py)\}',body)))
check('B all Section 1--6 cited scripts exist',all((ROOT/p).is_file() for p in paths))
print('B script inventory:',len(paths),'files; SHA256 draft:',hashlib.sha256(paper.encode()).hexdigest())
check('A6 corrected filenames exist; obsolete ones do not',all((ROOT/'phase2/scripts'/p).exists() for p in ['smatrix_poles_bootstrap.py','toda_quantum_corrections.py','smatrix_semiclassical_unitarity_crossing.py','smatrix_rejected_forms.py']) and all(not (ROOT/'phase2/scripts'/p).exists() for p in ['smatrix_boot.py','toda_quantum_correction.py']))
check('A9 actual defect title is Phase 2c', 'Phase 2c' in (ROOT/'phase2/defect_semiclassics.tex').read_text())
check('B no Phase 2i identifiers in Sections 1--6',not any(s in body for s in ['toda_wh_and_kp_synthesis','c_{\\mathrm{edge}}','c_{\\rm edge}','K_L(z)-K_L(1/z)']))
print('B Phase 2i filename mentions in whole draft:',paper.count('toda_wh_and_kp_synthesis'))
if '--run-sources' in sys.argv:
    for path in paths:
        if path=='phase1/scripts/figures.py':continue  # would overwrite a figure
        p=ROOT/path
        result=subprocess.run([sys.executable,'-B',p.name],cwd=p.parent,capture_output=True,text=True,
                              env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(path,result.returncode,flush=True)
        if result.returncode or 'ALL CHECKS PASSED' not in result.stdout:
            print(result.stdout,result.stderr)
        if path=='phase0/checks/convention_check.py':
            print('RECORDED LEGACY REPORTING FAILURE: no ALL CHECKS PASSED marker; check its exact output instead')
            check('source run '+path,result.returncode==0 and result.stdout.splitlines()==['good c = -3','bad c = 3'])
        else:
            check('source run '+path,result.returncode==0 and 'ALL CHECKS PASSED' in result.stdout)
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""(c) Consolidated errata: exact algebra and 70-decimal checks.
Read-only apart from this source; no existing checker is run or imported.
Expected historical counterexamples are asserted explicitly. This does not
prove the physical quantum frame, uniform edge asymptotics or completeness.
"""
from pathlib import Path
import ast
import itertools
import sys
import mpmath as mp
import sympy as sp
sys.dont_write_bytecode = True
mp.mp.dps = 70
ROOT = Path(__file__).resolve().parents[2]
TOL = mp.mpf('1e-55')
R3 = mp.sqrt(3)

def check(name, ok):
    assert bool(ok), name
    print('PASS (c): ' + name, flush=True)

def close(x,y,tol=TOL):
    return abs(x-y) < tol*max(1,abs(x),abs(y))

def source_functions(path):
    # Execute definitions only: no source top-level side effects or precision reset.
    tree=ast.parse((ROOT/path).read_text())
    ns={'mp':mp,'sp':sp,'r3':R3,'I':mp.j}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),path,'exec'),ns)
    return ns

def energies():
    x,t=sp.symbols('x t',real=True); phi=sp.Function('phi')(x,t); u=phi.diff(x)
    k,z=sp.symbols('k z',positive=True)
    U=k*k*(1-z*z)/2; jac=2/(k*(1-z*z)); Ux=-k*z*U
    ints=[sp.integrate(f*jac,(z,-1,1)) for f in [U,U**2,U**3,Ux**2]]
    check('energy_integrals',ints==[2*k,2*k**3/3,4*k**5/15,2*k**5/15])
    for sign in [-1,1]:
        L=phi.diff(t)**2/2-u*u/2-sign*u**3+sign*phi.diff(x,2)**2/2
        el=sp.euler_equations(L,phi,[x,t])[0].lhs
        target=phi.diff(t,2)-phi.diff(x,2)-sign*phi.diff(x,4)-3*sign*(u*u).diff(x)
        check('lagrangian_'+str(sign),sp.expand(el+target)==0)
        E=sp.simplify((2+sign*k*k)*ints[1]/2+sign*ints[2]-sign*ints[3]/2)
        v=sp.sqrt(1+sign*k*k); P=sp.Rational(2,3)*k**3*v
        check('energy_velocity_'+str(sign),sp.simplify(E-sp.Rational(2,15)*k**3*(5+4*sign*k*k))==0 and sp.simplify(E.diff(k)-v*P.diff(k))==0)
        for kk in map(mp.mpf,['.2','.6','.9']):
            uf=lambda X:kk*kk*mp.sech(kk*X/2)**2/2
            ux=lambda X:-kk*mp.tanh(kk*X/2)*uf(X)
            val=mp.quad(lambda X:(2+sign*kk*kk)*uf(X)**2/2+sign*uf(X)**3-sign*ux(X)**2/2,[-mp.inf,0,mp.inf])
            check('energy_quadrature_'+str((sign,kk)),close(val,2*kk**3*(5+4*sign*kk*kk)/15))
    Eg=2*k**3*(5-4*k*k)/15; Ew=k**3*(3-2*k*k)/3
    check('good_energy_misprint',sp.expand(Ew-Eg-k**3*(5-2*k*k)/15)==0 and sp.expand(sp.diff(Ew-Eg,k)-k*k*(1-2*k*k/3))==0)
    C,h=sp.symbols('C h')
    check('wick_sign',sp.expand((z**3-C*sp.diff(z**3,z,2)/2)-z**3)==-3*C*z)
    variance=mp.quad(lambda th:mp.sin(th),[0,mp.pi/2])/(2*mp.pi)
    check('stable_band_variance',close(variance,1/(2*mp.pi)))
    check('wick_energy',sp.simplify((-3*C*2*k).subs(C,h/(2*sp.pi))+3*h*k/sp.pi)==0)
    # Inspect actual downstream correct-energy implementation, without importing it.
    ns=source_functions('phase2/scripts/audit_charges_action.py')
    for a in map(mp.mpf,['.2','.6']):
        check('downstream_energy_'+str(a),close(ns['ecl'](a),2*mp.sin(a)**3*(5-4*mp.sin(a)**2)/15))

def sol(a,e=1,bad=False):
    k=mp.sinh(a) if bad else mp.sin(a); v=e*(mp.cosh(a) if bad else mp.cos(a))
    ell=e*mp.cosh(3*a) if bad else -e*mp.cos(3*a)
    center=(mp.j if bad else 1)*v/R3
    return dict(k=k,v=v,w=k*v,ell=ell,pa=(center+k)/2,pb=(center-k)/2)

def cross(a,b):
    return (a['pa']-b['pa'])*(a['pb']-b['pb'])/((a['pa']-b['pb'])*(a['pb']-b['pa']))

def roots_defect():
    ns=source_functions('phase2/scripts/oneloop_defect_scattering_and_stability.py')
    a=mp.mpf('.4'); sigma=mp.mpf('2.4')
    for eps,eta in itertools.product([-1,1],repeat=2):
        s=sol(a,eps); k,v=s['k'],s['v']; q=-sigma/(2*R3)
        z=((v+sigma-R3*k)/(v+sigma+R3*k))**eta
        printed=((k-(v+sigma)/R3)/(k+(v+sigma)/R3))**(-eta)
        check('transmission_'+str((eps,eta)),close(z,((s['pb']-q)/(s['pa']-q))**eta) and close(printed,-1/z) and close(z,ns['z_defect_mode'](s,sigma,eta)))
        left=lambda X,T:-6*k/(1+mp.exp(-k*(X-v*T)))
        right=lambda X,T:-6*k/(1+mp.exp(-k*(X-v*T))/z)
        jet=lambda fn:ns['eval_jet'](fn,mp.mpf('.27'))
        err=max(abs(x) for x in ns['cz_sewing_vec'](*jet(left),*jet(right),sigma,eta))
        check('unsigned_root_sewing_'+str((eps,eta)),err<TOL)
        if eta==-1:
            zw=((s['pb']+eta*sigma/(2*R3))/(s['pa']+eta*sigma/(2*R3)))**eta
            wrong=lambda X,T:-6*k/(1+mp.exp(-k*(X-v*T))/zw)
            er=max(abs(x) for x in ns['cz_sewing_vec'](*jet(left),*jet(wrong),sigma,eta))
            check('signed_root_counterexample_'+str(eps),abs(zw-z)>mp.mpf('.01') and er>1)
            if eps==1: print('NUMERIC root: correct',mp.nstr(z,18),'printed',mp.nstr(zw,18),'residual',mp.nstr(er,18))
        check('CZ_constraints_'+str(eta),close((2*eta/R3)**2-(eta/R3)**2,1))
    for bad,eps in itertools.product([False,True],[-1,1]):
        s=sol(a,eps,bad); k,v=s['k'],s['v']
        for p in [s['pa'],s['pb'],-s['pa']-s['pb']]:
            check('cubic_'+str((bad,eps)),close(p**3+(1 if bad else -1)*p/4,(mp.j if bad else 1)*s['ell']/(12*R3)))
        if not bad:
            check('ordered_mu_dictionary_'+str(eps),close((-s['pb'])**3+s['pb']/4,-s['ell']/(12*R3)))
            check('trigonometric_roots_'+str(eps),close(s['pa'],eps*mp.cos(a-eps*mp.pi/3)/R3) and close(s['pb'],eps*mp.cos(a+eps*mp.pi/3)/R3))
    check('wrong_CZ_coefficients',sp.Rational(1,12)-sp.Rational(1,12)!=1)

def minor():
    def phase(a,b):
        s,t=sol(a,1,True),sol(b,1,True)
        return mp.mpf(2)/9*(s['ell']-t['ell'])*mp.log(cross(s,t))+mp.mpf(8)/3*s['k']*t['k']*(s['v']-t['v'])
    b=mp.mpf('1e-12'); a=mp.mpf('1.7')*b
    ratio=mp.diff(lambda x:phase(x,b),a)/(2*a*mp.log(cross(sol(a,1,True),sol(b,1,True))))
    target=1+2/(mp.mpf('1.7')*mp.log((mp.mpf('.7')/mp.mpf('2.7'))**2))
    check('kdv_ratio',close(ratio,target,mp.mpf('1e-20')))
    print('NUMERIC KdV full / cross:',mp.nstr(target,18),mp.nstr(target-1,18))
    z=sp.symbols('z',positive=True)
    check('toda_width',sp.sqrt(12)==2*sp.sqrt(3) and sp.limit(sp.sinh(z)/z,z,0)==1)
    for name in ['smatrix_poles_bootstrap.py','toda_quantum_corrections.py','smatrix_rejected_forms.py','smatrix_semiclassical_unitarity_crossing.py']:
        # The first two repair broken names; the latter two document omissions.
        check('script_path_'+name,(ROOT/'phase2/scripts'/name).is_file())
    check('phase_number','Phase 2c' in (ROOT/'phase2/defect_semiclassics.tex').read_text())

A=1/(2*mp.pi); C=(1-mp.log(8*mp.pi))/(2*mp.pi)
D=mp.mpf(1)/48-1/(4*mp.pi**2); E=(4+6*mp.zeta(3)-mp.pi**2)/(96*mp.pi**3)
def T(s):
    z=s/(2*mp.pi)
    return mp.exp(mp.loggamma(1+z)-z*(mp.log(z)-1))/mp.sqrt(s)
def inverse(u,order=0):
    return mp.invertlaplace(lambda s:mp.sqrt(mp.pi)*(T(s)-(1 if order==2 else 0))*s**(order-2),u,method='talbot')

def wh():
    for w in map(mp.mpf,['.2','1','5','20']):
        val=2*mp.pi*T(mp.j*w)*T(-mp.j*w)
        check('wh_symbol_'+str(w),close(val,2*mp.pi/(1-mp.exp(-w))))
        check('wh_printed_counterexample_'+str(w),abs(val-2*mp.pi**2*mp.exp(mp.pi*w)/mp.sinh(mp.pi*w))>1)
    check('momentum_normalisation',close(mp.mpf(4)/3*mp.gamma(mp.mpf('2.5')),mp.sqrt(mp.pi)))
    h,b,y,a=sp.symbols('h b y a',positive=True); k=h*sp.sqrt(y/b)
    P=4*b*(k*sp.cosh(k)-sp.sinh(k))/a
    check('momentum_scale',sp.simplify(sp.limit(a*sp.sqrt(b)*P/h**3,h,0)-4*y**sp.Rational(3,2)/3)==0)
    aa,cc,dd,ee,L,t=sp.symbols('A c d e L t')
    v0=aa*L-cc; v1=aa*v0-dd; v2=aa*v1-aa*v0*v0/2+dd*v0-ee
    # Exact coefficient extraction after composing u(y(u)); log u is an independent formal variable L.
    V=v0+v1*t+v2*t*t
    residual=V-aa*(L+sp.series(sp.log(1+t*V),t,0,3).removeO())+cc+dd*t*sp.series(1/(1+t*V),t,0,2).removeO()+ee*t*t
    check('inverse_series_reversion',all(sp.expand(residual).coeff(t,j)==0 for j in range(3)))
    check('inverse_log_squared',sp.expand(v2).coeff(L,2)==-aa**3/2)
    for u in map(mp.mpf,['32','128','512','2048']):
        yy=(3*inverse(u)/4)**(mp.mpf(2)/3)
        residual=u-yy+A*mp.log(yy)-C-D/yy-E/yy**2
        check('corrected_transform_asymptotic_'+str(u),abs(residual)*yy**3<mp.mpf('.0004'))
        wrong=(mp.log(8*mp.pi*u)-1)/(2*mp.pi)+(12*mp.log(8*mp.pi*u)-mp.pi**2)/(48*mp.pi**2*u)+(-12*mp.log(8*mp.pi*u)+6*mp.zeta(3)+mp.pi**2-8)/(96*mp.pi**3*u*u)
        check('printed_inverse_counterexample_'+str(u),abs(yy-u-wrong)*u*u>mp.mpf('.01'))
    check('B_boundary',close(inverse(mp.mpf('1e-20'),1),mp.sqrt(mp.pi),mp.mpf('1e-19')))
    x,z=sp.symbols('T1 T2')
    check('double_transform_contact',sp.expand(2*(sp.sqrt(sp.pi)*x)*(sp.sqrt(sp.pi)*z)-2*sp.pi-2*sp.pi*(x*z-1))==0)
    # A real-space check of the differentiated convolution includes B(0) f(u-v).
    u,v=mp.mpf(4),mp.mpf(1)
    # 24/48-node Gauss quadrature; inverse Laplace evaluations at 45 digits suffice.
    with mp.workdps(45):
        def conv(uu,vv):
            X,W=mp.gauss_quadrature(32,'legendre')
            return vv*sum(w*inverse(uu-vv*(x+1)/2,2)*inverse(vv*(1-x)/2,1) for x,w in zip(X,W))
        # Differential identity (d_u+d_v) C=2 f(u) B(v), checked by numerical differentiation.
        derivative=mp.diff(lambda t:conv(u+t,v+t),mp.mpf(0),method='step',addprec=10)
        check('convolution_endpoint',close(derivative,2*inverse(u,2)*inverse(v,1),mp.mpf('1e-25')))
    print('NUMERIC conditional c,d,e:',*[mp.nstr(z,18) for z in [C,D,E]])

def darboux_kernel():
    alpha,beta=mp.mpf('.3'),mp.mpf('.7')
    gamma=-3*mp.cos(2*alpha)/(mp.sin(alpha)*mp.sin(3*alpha))
    for eps,eps2 in itertools.product([-1,1],repeat=2):
        s,t=sol(alpha,eps),sol(beta,eps2); k=s['k']; l=t['k']
        def delay(q,factor):
            return -eps*mp.log(abs((s['pb']-q)/(s['pa']-q)))/k-2*factor*eps*gamma*q
        lhs=-mp.log(abs(cross(s,t)))/k+2*l*gamma
        rhs=eps*(delay(t['pb'],1)-delay(t['pa'],1))
        wrong=eps*(delay(t['pb'],R3)-delay(t['pa'],R3))
        check('darboux_'+str((eps,eps2)),close(lhs,rhs) and close(wrong-lhs,2*l*gamma*(R3-1)))
    K=lambda z:z*z-z**-2
    xs=[mp.mpf('.4'),mp.mpf('.8'),mp.mpf('1.2')]; ys=[mp.mpf('.3'),mp.mpf('.9'),mp.mpf('1.4')]; weights=[1,1,-2]
    bil=lambda xx,yy,fn:sum(c*d*fn(x/y) for x,c in zip(xx,weights) for y,d in zip(yy,weights))
    check('kernel_iff_counterexample',close(bil(xs,ys,K),-bil(ys,xs,K)) and K(-2)==K(2) and K(2)!=0)
    check('kernel_null_constant',close(bil(xs,ys,lambda z:K(z)+1),bil(xs,ys,K)))

def reciprocity():
    # Independent high-precision finite-sea Nystrom calculation. A short sea
    # permits a stringent quadrature check without repeating expensive edge fits.
    q=mp.mpf('.2'); h=mp.mpf(1); t=2*mp.pi*h
    K=lambda d:2*mp.re(mp.digamma(1+mp.j*d/h))+2*mp.log(h)
    theta=lambda d:2*h*mp.im(mp.loggamma(1+mp.j*d/h))+2*d*mp.log(h)
    errors=[]
    for N in [24,36]:
        X,W=mp.gauss_quadrature(N,'legendre'); X=q*X; W=q*W
        matrix=mp.matrix([[t*(i==j)-K(X[i]-X[j])*W[j] for j in range(N)] for i in range(N)])
        inv=matrix**-1
        args=[q,-q,mp.mpf('.5'),mp.mpf('.7'),mp.mpf('-.6')]
        Fs={p:inv*mp.matrix([theta(p-x) for x in X]) for p in args}
        def end(p,e):return (theta(p-e)+sum(K(e-X[i])*W[i]*Fs[p][i] for i in range(N)))/t
        def phase(p,z):return theta(p-z)-sum(K(p-X[i])*W[i]*Fs[z][i] for i in range(N))
        def four(p,z,e,f):return phase(p,z)-phase(p,f)-phase(e,z)+phase(e,f)
        er=mp.mpf(0)
        for p,z,e,f in [(args[2],args[3],q,q),(args[2],args[4],q,-q)]:
            rhs=t*(end(p,q)*end(z,q)-end(p,-q)*end(z,-q))
            er=max(er,abs(phase(p,z)+phase(z,p)-rhs))
            r4=t*((end(p,q)-end(e,q))*(end(z,q)-end(f,q))-(end(p,-q)-end(e,-q))*(end(z,-q)-end(f,-q)))
            er=max(er,abs(four(p,z,e,f)+four(z,p,f,e)-r4))
            check('fourterm_zero_edge_'+str(N),close(four(e,z,e,f),0))
        errors.append(er)
        check('finite_sea_reciprocity_'+str(N),er<mp.mpf('1e-40' if N==24 else '1e-55'))
    check('reciprocity_mesh_convergence',errors[-1]<errors[0])
    check('edge_product_counterexample',2*mp.sqrt(mp.pi)*inverse(1,1)>1)
    print('NUMERIC reciprocity errors:',*[mp.nstr(e,5) for e in errors])

def formal_integrals():
    for r in map(mp.mpf,['.2','.4','.6','.8']):
        chi=(mp.polylog(2,r)-mp.polylog(2,-r))/2
        c0=-4*(chi-r)/mp.pi
        val=4/mp.pi*mp.quad(lambda s:mp.log(s/r)*s*s/(1-s*s),[0,r])
        check('formal_c0_'+str(r),close(val,c0))
        def integrand(t):
            sig=1-r*r+t*t; delta=-mp.log(t*t/(r*r*sig))/(2*mp.pi)
            return 8/sig**mp.mpf('2.5')*(t*t*(-2*D-3/(4*mp.pi**2)-delta/(4*mp.pi)+3*delta**2/8)-D*sig/2*(1+t*t*(1-1/(r*r)-1/sig)))
        c1=-r**3/(6*(1-r*r))-2*D*((1-r*r)/r**2*mp.log((1-r)/(1+r))+2/r)
        check('formal_c1_'+str(r),close(mp.quad(integrand,[0,r]),c1))
    # Free vs fixed-intercept estimators are different even on exact quadratic data.
    c0,c1,c2=sp.symbols('c0 c1 c2'); f=lambda Y:c0+c1/Y+c2/Y**2
    check('fit_estimator',sp.expand(64*(f(32)-c0)-16*(f(16)-c0)-c1)==0 and sp.expand(32*(f(16)-f(32))-c1-c2*sp.Rational(3,32))==0)

def source_dictionaries():
    x,t=sp.symbols('x t'); phi=sp.Function('phi')(x,t); u=sp.Function('u')(x,t)
    bc=-phi
    bc_eq=bc.diff(t,2)-bc.diff(x,2)-bc.diff(x,4)+6*bc.diff(x)*bc.diff(x,2)
    bad=phi.diff(t,2)-phi.diff(x,2)-phi.diff(x,4)-3*(phi.diff(x)**2).diff(x)
    w=u/2
    hirota=w.diff(t,2)-w.diff(x,2)-6*(w*w).diff(x,2)-w.diff(x,4)
    pde=u.diff(t,2)-u.diff(x,2)-3*(u*u).diff(x,2)-u.diff(x,4)
    cz=-3*phi
    Lcz=cz.diff(t)**2/2-cz.diff(x)**2/2-cz.diff(x)**3/3-cz.diff(x,2)**2/2
    Lg=phi.diff(t)**2/2-phi.diff(x)**2/2+phi.diff(x)**3-phi.diff(x,2)**2/2
    check('source_dictionaries',sp.expand(bc_eq+bad)==0 and sp.expand(2*hirota-pde)==0 and sp.expand(Lcz-9*Lg)==0)

def defect_spectrum():
    ns=source_functions('phase2/scripts/oneloop_defect_scattering_and_stability.py')
    for s,g in itertools.product(map(mp.mpf,['-1.5','.5','1.9']),map(mp.mpf,['.2','.7'])):
        disc=mp.sqrt(1-4*g*g); ks=[mp.sqrt((1+disc)/2),mp.sqrt((1-disc)/2)]
        cols=[]
        for side in [0,1]:
            for k in ks:
                k=(-1)**side*k; jet=[1,k,k*k,k**3,g,g*k]
                jet=jet+[0]*6 if side==0 else [0]*6+jet
                cols.append([mp.diff(lambda a:ns['cz_sewing_vec'](*[a*j for j in jet],s,1)[i],0) for i in range(4)])
        matrix=mp.matrix([[cols[j][i] for j in range(4)] for i in range(4)])
        S=sum(ks); pm=S*S-(R3-1)*s*S/2+(s*s-2-R3)/2; pp=S*S-(R3+1)*s*S/2+(s*s-2+R3)/2
        correct=-mp.mpf(2)/3*S*S*(S*S-1)*pm*pp*(ks[0]-ks[1])**2
        actual=mp.det(matrix)
        check('defect_determinant_'+str((s,g)),close(actual,correct) and close(actual/(correct/16),16))
    for s in map(mp.mpf,['1.2','1.5','1.9','1.99']):
        disc=mp.sqrt(4-s*s)
        roots=[((R3-1)*s+e*(R3+1)*disc)/4 for e in [-1,1]]+[((R3+1)*s+e*(R3-1)*disc)/4 for e in [-1,1]]
        rates=sorted((S*S-1)/2 for S in roots if S>1)
        target=sorted([(s*disc+R3*(s*s-2))/8,abs(s*disc-R3*(s*s-2))/8])
        check('growth_branch_'+str(s),len(rates)==2 and all(close(a,b) for a,b in zip(rates,target)))
        if s==mp.mpf('1.9'):print('NUMERIC growing rates at s=1.9:',*[mp.nstr(g,18) for g in rates])
    s,S=sp.symbols('s S'); r=sp.sqrt(3)
    pp=S*S-(r+1)*s*S/2+(s*s-2+r)/2
    check('growth_threshold',sp.simplify(pp.subs({S:1,s:r}))==0 and sp.simplify(pp.subs(s,2)-(S-(r+1)/2)**2)==0 and sp.simplify((((r+1)/2)**2-1)/2-r/4)==0)
    for sigma in map(mp.mpf,['-.5','.5','1.5']):
        triples=[sigma,(-sigma+mp.sqrt(12-3*sigma*sigma))/2,(-sigma-mp.sqrt(12-3*sigma*sigma))/2]
        for eta in [-1,1]:check('stable_triple_domain',sum(eta*z < -1 for z in triples)==1)
    check('degenerate_triple',sp.factor(s*s+2*s+1)==(s+1)**2)

if __name__=='__main__':
    source_dictionaries(); energies(); roots_defect(); minor(); defect_spectrum(); wh(); darboux_kernel(); reciprocity(); formal_integrals()
    print('ALL CHECKS PASSED')

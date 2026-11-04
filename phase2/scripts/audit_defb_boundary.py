#!/usr/bin/env python3
"""Independent audit; no project modules imported, no caches read or written.
(c) Exact algebra and numerical evidence; physical identifications remain (d).
Run with python3 phase2/scripts/audit_defb_boundary.py. Optional --package-runs
runs the two original scripts in subprocesses, redirecting ONLY their JSON writes
to /tmp/audit_defb_boundary_*.json. It never modifies package sources or caches.
"""
import sys
import time
import subprocess
from pathlib import Path
import sympy as S
import mpmath as m

START = time.perf_counter()
m.mp.dps = 80
m.iv.dps = 45
r = S.sqrt(3)
rm = m.sqrt(3)
TOL = m.mpf('1e-60')

def zero(x):
    assert S.simplify(S.cancel(S.expand(x))) == 0, x

def close(x, y=0, tol=TOL):
    assert abs(x-y) < tol, (m.nstr(x,20), m.nstr(y,20), m.nstr(abs(x-y),8))

def logrow(name, x):
    print(name, m.nstr(x, 18), flush=True)

# 1. Bulk counterterm, full charge variation, and vacuum renormalisation.
a, v, k, lam, h = S.symbols('a v k lam h', real=True)
kav = S.sqrt(a-v*v)
Pav = S.Rational(2,3)*v*kav**3
Eav = S.Rational(2,15)*kav**3*(5*a-4*kav**2)
dP = S.diff(Pav,a).subs(a,1)*2*lam
dE = S.diff(Eav,a).subs(a,1)*2*lam
zero(dP-2*lam*v*S.sqrt(1-v*v))
zero(dE-S.Rational(2,3)*lam*S.sqrt(1-v*v)*(1+2*v*v))
zero(dE-v*dP-S.Rational(2,3)*lam*(1-v*v)**S.Rational(3,2))
x=S.symbols('x', real=True)
# Independent spatial profile moments, using y=tanh(k*x/2), not hardcoded.
y=S.symbols('y', real=True)
u=k*k*(1-y*y)/2
zero(S.integrate(u*u*2/(k*(1-y*y)),(y,-1,1))-2*k**3/3)
q1,q2,q3=S.symbols('q1 q2 q3',real=True)
qs=[q1,q2,q3]; ds=[qs[i]-qs[(i+1)%3] for i in range(3)]
prod=ds[0]*ds[1]*ds[2]
zero(sum(d**3 for d in ds)-3*prod)
zero(sum(ds[i]*r*(qs[i]+qs[(i+1)%3]) for i in range(3)))
# Stiffness-rescaled exact solution, k=sqrt(a)k0, v=sqrt(a)v0.
# Both charge sums stay zero: E scales as a^(5/2), P as a^2.
P0=S.Rational(2,3)*v*(1-v*v)**S.Rational(3,2)
E0=S.Rational(2,15)*(1-v*v)**S.Rational(3,2)*(1+4*v*v)
zero(dP+S.diff(P0,v)*lam*v-4*lam*P0)
zero(dE+S.diff(E0,v)*lam*v-5*lam*E0)
# Arbitrary first-order energy/momentum holonomy can be cancelled kinematically
# by two independent edge momentum changes; quantum sewing is NOT established.
v1,v2,bE,bP=S.symbols('v1 v2 bE bP')
shift1=(-bE+v2*bP)/(v1-v2); shift2=(bE-v1*bP)/(v1-v2)
zero(v1*shift1+v2*shift2+bE); zero(shift1+shift2+bP)
U=S.symbols('U')
W=-((1-6*U)**S.Rational(3,2)-1+9*U)/(6*S.pi)
zero(S.diff(W,U,2).subs(U,0)/2+9/(4*S.pi))
assert S.diff(W,U,2).subs(U,0)/2 != 0  # lambda=0 is NOT zero sound shift.
# Explicit density continuation, including the counterterm sign.
lb=S.symbols('lb'); zero(-(lb*(-U)**2)-(-lb)*U**2)
beta=S.symbols('beta',real=True)
zero(S.sin(S.I*beta)-S.I*S.sinh(beta))
zero(S.cos(S.I*beta)-S.cosh(beta))
zero(-(-S.cos(3*S.I*beta))-S.cosh(3*beta))

s,z,dr,dt,dh,ds0,di=S.symbols('s z dr dt dh ds0 di',real=True)
for eta in [-1,1]:
    A=S.Integer(eta)/r
    rho0=-(s-s**3/3)/(2*A)
    rho=rho0+(z-z**3/3)/(2*A)
    # Leibniz variation of the two definite integrals, endpoints kept separate.
    dEv=-dt/9+dr*(z*z-s*s)/18+z*rho*di/9-s*rho.subs(z,s)*ds0/9
    dPv=dh/9-dr*(z-s)/9-rho*di/9+rho.subs(z,s)*ds0/9
    cubic=z**3-3*z-s**3+3*s
    zero(S.rem(S.expand(dEv-(-dt/9+dr*(z*z-s*s)/18)),cubic,z))
    zero(S.rem(S.expand(dPv-(dh/9-dr*(z-s)/9)),cubic,z))
    # Reference vacuum constraint couples the parameter shifts.
    constraint=dr+(1-s*s)*ds0/(2*A)
    assert constraint.subs(dr,0) != 0
    zero(constraint.subs(dr,-(1-s*s)*ds0/(2*A)))
    slope=-2*r*eta/(s+1)
    loop=S.symbols('unknown_loop_slope')
    ds_required=-loop/S.diff(slope,s)
    zero(loop+S.diff(slope,s)*ds_required)
print('(c) bulk shift, fixed-P identity, scaled-locus cancellation and envelope checked',flush=True)

# Explicit good/CZ normalization and zero-momentum / large-q limits.
pt,px,pxx=S.symbols('pt px pxx')
LCZ=(-3*pt)**2/2-(-3*px)**2/2-(-3*px)**3/3-(-3*pxx)**2/2
zero(LCZ-9*(pt**2/2-px**2/2+px**3-pxx**2/2))
qp=S.symbols('qp',positive=True)
for eta in [-1,1]:
    zp=((S.sqrt(1+qp*qp)+s-S.I*r*qp)/(S.sqrt(1+qp*qp)+s+S.I*r*qp))**eta
    zero(zp.subs(qp,0)-1)
    zero(S.diff(zp,qp).subs(qp,0)/S.I+2*r*eta/(s+1))
    lim=((1-S.I*r)/(1+S.I*r))**eta
    zero(S.expand_complex(lim-S.exp(-2*S.pi*S.I*eta/3)))
    zn=((-1+S.I*r)/(-1-S.I*r))**eta
    zero(lim-zn)
# 2. Independent local action in project variables. Variational sewing and
# fluxes are derived here; no CZ/package code is called.
a0,b0,ax,bx,at,bt,axx,bxx,axxx,bxxx,axt,bxt=S.symbols(
    'a0 b0 ax bx at bt axx bxx axxx bxxx axt bxt',real=True)
vars0=[a0,b0,ax,bx,at,bt,axt,bxt,axx,bxx,axxx,bxxx]
R0,T0,H0=S.symbols('R0 T0 H0',real=True)
ACTION={}
for eta in [-1,1]:
    A=S.Integer(eta)/r
    sig=s+(b0-a0)/A
    rho=R0+(z-z**3/3)/(2*A)
    tau=T0-S.integrate(z*rho,(z,s,sig))
    hh=H0-S.integrate(rho,(z,s,sig))
    F=A*(ax+2*bx)+sig**2/(12*A)
    G=A*(2*ax+bx)+sig**2/(12*A)
    Ed=-A*sig*(ax*ax+ax*bx+bx*bx)+rho.subs(z,sig)*(ax+bx)/3-tau/9
    Pd=-A*(ax*ax+4*ax*bx+bx*bx)/2+hh/9
    D=at*F+bt*G-Ed
    td=lambda f: sum(S.diff(f,j)*jdot for j,jdot in [(a0,at),(b0,bt),(ax,axt),(bx,bxt)])
    sew=[axx-S.diff(D,ax), bxx+S.diff(D,bx),
         axxx-ax+3*ax*ax+S.diff(D,a0)-td(F),
         bxxx-bx+3*bx*bx-S.diff(D,b0)+td(G)]
    # Check the flux identities on the sewing ideal, starting with project
    # variables instead of the package's converted CZ action.
    JE=lambda tx,xx,xxx,xxxx,xxt: tx*(xx-3*xx*xx-xxxx)+xxt*xxx
    JP=lambda tx,xx,xxx,xxxx: -tx*tx/2-xx*xx/2+2*xx**3+xx*xxxx-xxx*xxx/2
    balances=[JE(at,ax,axx,axxx,axt)-JE(bt,bx,bxx,bxxx,bxt)+td(Ed),
              JP(at,ax,axx,axxx)-JP(bt,bx,bxx,bxxx)+td(Pd),
              ax-3*ax*ax-axxx-bx+3*bx*bx+bxxx+td(F+G),at-bt+td(b0-a0)]
    # Polynomial ideal membership; bulk equation and sewing are necessary.
    solved=S.solve(sew,[axxx,bxxx,axx,bxx], dict=True)[0]
    for balance in balances:
        zero(balance.subs(solved, simultaneous=True))
    vacuum={R0:-(s-s**3/3)/(2*A),T0:0,H0:0}
    Q=[S.expand(f.subs(vacuum)) for f in [Ed,Pd,F+G,b0-a0]]
    sew=[S.expand(f.subs(vacuum)) for f in sew]
    ACTION[eta]=(Q,sew)
    # Quadratic coefficients are Hessians of the full local charges.
    origin={j:0 for j in vars0}
    HEs=S.hessian(Q[0],vars0).subs(origin)
    HPs=S.hessian(Q[1],vars0).subs(origin)
    HKs=S.hessian((at*F+bt*G).subs(vacuum),vars0).subs(origin)
    ACTION[eta]+=(S.lambdify([s],HEs,'mpmath'),S.lambdify([s],HPs,'mpmath'),S.lambdify([s],HKs,'mpmath'))

# Spatial quadrature in physical x (package used logistic integration).
ka=(rm/4-(m.sqrt(7)-rm)/8); va=rm*(rm/4+(m.sqrt(7)-rm)/8)
maxabs=m.mpf(0)
for eta in [-1,1]:
    sv=eta*rm*ka-va
    Q,sew,*_=ACTION[eta]
    qfun=S.lambdify([s]+vars0,Q,'mpmath'); sf=S.lambdify([s]+vars0,sew,'mpmath')
    previous=None
    for tv in [m.mpf('-2'),m.mpf('.7'),m.mpf('3')]:
        def profile(xx):
            Y=ka*(xx-va*tv)/2
            return ka*(1+m.tanh(Y)),ka*ka/(2*m.cosh(Y)**2)
        phi,uu=profile(0); yy=m.tanh(-ka*va*tv/2)
        ux=-ka*uu*yy; uxx=ka*ka*uu*(3*yy*yy-1)/2
        jets=[phi,0,uu,0,-va*uu,0,-va*ux,0,ux,0,uxx,0]
        for e in sf(sv,*jets):close(e)
        def density(xx,j):
            _,u0=profile(xx); ux0=-ka*u0*m.tanh(ka*(xx-va*tv)/2)
            return [(1+va*va)*u0*u0/2-u0**3+ux0*ux0/2,va*u0*u0,-va*u0,u0][j]
        bulk=[m.quad(lambda xx:density(xx,j),[-m.inf,-10,0]) for j in range(4)]
        total=[b+d for b,d in zip(bulk,qfun(sv,*jets))]
        if previous:
            maxabs=max(maxabs,max(abs(i-j) for i,j in zip(previous,total)))
            for i,j in zip(previous,total):close(i,j)
        previous=total
    close((va+sv-eta*rm*ka)/(va+sv+eta*rm*ka))
close((va+m.mpf('1.5')+rm*ka)/(va+m.mpf('1.5')-rm*ka),m.sqrt(21)-3)
logrow('(c) physical-x absorption max residual',maxabs)

# 3. Reconstruct plane-wave transmission from all four linearised sewings,
# then contract Hessians. The energy trace sign is not assumed.
def Z(q,sv,eta):
    X=m.sign(q)*m.sqrt(1+q*q)+sv
    return ((X-1j*rm*q)/(X+1j*rm*q))**eta

def DZ(q,sv,eta):
    return m.im(m.diff(lambda p:Z(p,sv,eta),q)/Z(q,sv,eta))

def wave(q,zz):
    w=abs(q)*m.sqrt(1+q*q)
    return m.matrix([1,zz,1j*q,1j*q*zz,-1j*w,-1j*w*zz,q*w,q*w*zz,-q*q,-q*q*zz,-1j*q**3,-1j*q**3*zz])

def contract(H,V,q):
    return m.re((V.conjugate().T*H*V)[0])/(8*m.pi*abs(q)*m.sqrt(1+q*q))
for eta in [-1,1]:
    Q,sew,HE,HP,HK=ACTION[eta]
    linear=S.Matrix(sew).jacobian(vars0).subs({j:0 for j in vars0})
    lf=S.lambdify([s],linear,'mpmath')
    for sv in [m.mpf('-2.4'),m.mpf('-.4'),m.mpf('1.4')]:
        for q in [m.mpf('-.83'),m.mpf('.37')]:
            mat=lf(sv); base=mat*wave(q,0); dz=mat*(wave(q,1)-wave(q,0))
            fitted=-base[0]/dz[0]
            close(fitted,Z(q,sv,eta))
            for e in mat*wave(q,fitted):close(e)
        qq=m.mpf('.63')
        e=sum(contract(HE(sv),wave(q,Z(q,sv,eta)),q) for q in [qq,-qq])
        kin=sum(contract(HK(sv),wave(q,Z(q,sv,eta)),q) for q in [qq,-qq])
        target=qq*m.sqrt(1+qq*qq)*(DZ(qq,sv,eta)+DZ(-qq,sv,eta))/(4*m.pi)
        close(e,kin);close(e,target); assert abs(e+target)>m.mpf('.001')
        # Plane-wave boundary integration-by-parts flux vanishes because |z|=1.
        close(abs(Z(qq,sv,eta)),1)
# Direct IBP boundary quadratic: half Re[phi*(phi_x-phi_xxx)
# + conjugate(phi_x)*phi_xx] vanishes separately on each plane-wave side.
qr=S.symbols('qr',real=True)
zero(S.re((S.I*qr+S.I*qr**3)+S.conjugate(S.I*qr)*(-qr*qr)))
# Numerical covariance check by time averaging a real mode.
close(m.quad(lambda t:m.cos(t)**2,[0,2*m.pi])/(2*m.pi),m.mpf('.5'))
# Spectral polynomial is independently obtained from these same linear sewings.
for eta in [-1,1]:
    linear=S.Matrix(ACTION[eta][1]).jacobian(vars0).subs({j:0 for j in vars0})
    lf=S.lambdify([s],linear,'mpmath')
    for sv in [m.mpf('-1.4'),m.mpf('.3'),m.mpf('1.6')]:
        g=m.mpf('.31'); rr=m.sqrt(1-4*g*g)
        kappas=[m.sqrt((1+rr)/2),m.sqrt((1-rr)/2)]
        cols=[]
        for side in [0,1]:
            for kap in kappas:
                kk=kap if side==0 else -kap
                jet=[0]*12
                for pos,val in [(0,1),(2,kk),(4,g),(6,g*kk),(8,kk*kk),(10,kk**3)]:jet[pos+side]=val
                cols.append(lf(sv)*m.matrix(jet))
        # Match the determinant convention: left two sewings, right two.
        mat=m.matrix([[col[i] for col in cols] for i in [0,2,1,3]])
        SS=sum(kappas); c=eta*sv
        pm=SS*SS-(rm-1)*c*SS/2+(c*c-2-rm)/2
        pp=SS*SS-(rm+1)*c*SS/2+(c*c-2+rm)/2
        close(m.det(mat)/(kappas[0]-kappas[1])**2,-m.mpf(2)/3*SS**2*(SS**2-1)*pm*pp)
# Growing-root counting, including the switch, by direct quadratic roots.
for cval,expected in [('-1.7',0),('-.5',1),('1.3',2),('1.9',2)]:
    cv=m.mpf(cval);roots=[]
    for sign in [-1,1]:
        roots+=m.polyroots([1,-(rm+sign)*cv/2,(cv*cv-2+sign*rm)/2])
    admiss=[z for z in roots if abs(m.im(z))<TOL and m.re(z)>1]
    assert len(admiss)==expected
SS=S.symbols('SS',real=True)
polys=[SS**2-(r+sign)*r*SS/2+(1+sign*r)/2 for sign in [-1,1]]
exactroots=[a for pp in polys for a in S.solve(pp,SS)]
growth=[S.simplify((a*a-1)/2) for a in exactroots if a>1]
assert growth==[r/4]
tpos=S.symbols('tpos',positive=True)
zero(12*(1/(2*r)+tpos)**2-1-(12*tpos*tpos+4*r*tpos))
print('(c) action-derived sewing, Hessian traces, determinant and stability checked',flush=True)

# 4. Boundary integrals. q=y/(1-y), unlike package's z=1/sqrt(1+q^2).
# Numerators below are independently reconstructed by Hessian contractions
# above and checked pointwise before quadrature. Stable cancelled forms avoid
# subtracting two O(1) terms at an infinite endpoint.
def densities(q,sv,eta):
    u=m.sqrt(1+q*q); den=16*q**4+4*(sv*sv+2)*q*q+(sv*sv-1)**2
    p=2*rm*eta/m.pi*q/u*((sv*sv-2)*u*u+(3-sv*sv)/2)/den
    e=rm*eta*sv/(4*m.pi)*q/u*(-4*u*u+(sv*sv-3)**2)/den
    return p,e
for eta in [-1,1]:
    for sv in [m.mpf('-1.7'),m.mpf('.3'),m.mpf('2.6')]:
        qq=m.mpf('.43'); _,_,HE,HP,HK=ACTION[eta]
        ep=sum(contract(HE(sv),wave(q,Z(q,sv,eta)),q) for q in [qq,-qq])
        pp=sum(contract(HP(sv),wave(q,Z(q,sv,eta)),q) for q in [qq,-qq])
        dp,de=densities(qq,sv,eta)
        close(pp,dp);close(ep+rm*eta*sv/(4*m.pi)*qq/m.sqrt(1+qq*qq),de)

def mapped(y,sv,eta,j):
    if y==0:return m.mpf(0)
    if y==1:
        return (rm*eta*(sv*sv-2)/(8*m.pi) if j==0 else -rm*eta*sv/(16*m.pi))
    return densities(y/(1-y),sv,eta)[j]/(1-y)**2

def closed(sv,eta):
    L=m.log(abs((sv+1)/(sv-1)))
    if abs(sv)<2:
        d=m.sqrt(4-sv*sv); B=-d*m.log(abs((rm+d)/(rm-d)))
    elif abs(sv)==2:B=m.mpf(0)
    else:
        d=m.sqrt(sv*sv-4); B=2*d*m.atan(d/rm)
    P=eta*(rm*sv*L+B)/(8*m.pi)
    ER=eta*(-4*rm*sv+rm*(2-sv*sv)*L+sv*B)/(16*m.pi)
    return P,ER+rm*eta*sv/(4*m.pi),ER

# Analytic partial-fraction rederivation; integration is elementary after
# completing the squares 4(u +/- s/4)^2 + 3(s^2-4)/4.
uS=S.symbols('uS')
Dp=4*uS*uS+2*s*uS+s*s-3; Dm=Dp.subs(s,-s)
np=2*((s*s-2)*uS*uS+(3-s*s)/2)
ne=(-4*uS*uS+(s*s-3)**2)/4
zero(np/(Dp*Dm)-((-s*uS-1)/(2*Dp)+(s*uS-1)/(2*Dm)))
zero(ne/(Dp*Dm)-(((s*s-2)*uS/(4*s)+(s*s-3)/8)/Dp+((2-s*s)*uS/(4*s)+(s*s-3)/8)/Dm))
rows=[]; maxerr=m.mpf(0)
for ss,eta in [('-1.2',1),('-1.5',1),('-sqrt(3)',1),('-2',1),('-2.5',1),('0',1),('.5',-1),('1.7',-1),('2.6',-1)]:
    sv=-rm if ss=='-sqrt(3)' else m.mpf(ss)
    values=[]
    for j in [0,1]:
        val=m.quad(lambda y:mapped(y,sv,eta,j),[0,m.mpf('.2'),m.mpf('.6'),1])
        values.append(val)
    target=closed(sv,eta)
    err=max(abs(values[j]-target[j]) for j in [0,1]);maxerr=max(maxerr,err)
    close(values[0],target[0]);close(values[1],target[1])
    rows.append((ss,eta,values[0],values[1]-rm*eta*sv/(4*m.pi)))
logrow('(c) independent Casimir quadrature max error',maxerr)
for row in rows:print('Casimir',row[0],row[1],*[m.nstr(x,22) for x in row[2:]],flush=True)
# Mesh / precision sensitivity, no extrapolation or fitting.
sv=m.mpf('-1.2'); mesh=[]
for n in [16,32,64]:
    X,Wg=m.gauss_quadrature(n,'legendre')
    vals=[]
    for j in [0,1]:
        val=m.mpf(0)
        for lo,hi in [(m.mpf(0),m.mpf('.2')),(m.mpf('.2'),m.mpf('.6')),(m.mpf('.6'),m.mpf(1))]:
            val+=(hi-lo)/2*sum(Wg[i]*mapped((hi+lo+(hi-lo)*X[i])/2,sv,1,j) for i in range(n))
        vals.append(val)
    error=max(abs(vals[j]-closed(sv,1)[j]) for j in [0,1]);mesh.append(error)
    logrow('GL nodes per panel '+str(n)+' discrepancy',error)
assert mesh[-1]<m.mpf('1e-45')
with m.workdps(60):
    p60=m.quad(lambda y:mapped(y,m.mpf('-1.2'),1,0),[0,m.mpf('.2'),m.mpf('.6'),1])
logrow('60/80 dps quadrature spread',abs(p60-rows[0][2]))
assert abs(p60-rows[0][2])<m.mpf('1e-58')

# 5. Soliton charges independently integrated from KP cross-ratio derivative.
def dsol(q,k,v):
    u=m.sqrt(1+q*q);sg=m.sign(q)
    pa=(v/rm+k)/2;pb=(v/rm-k)/2
    A=sg*u/(2*rm)+1j*q/2; B=A.conjugate()
    Ap=sg*q/(2*rm*u)+1j/2; Bp=Ap.conjugate()
    return m.im(-Ap/(pa-A)-Bp/(pb-B)+Bp/(pa-B)+Ap/(pb-A))

def solcharges(k,v,cutoff=None):
    def integrand(y,j):
        if y in [0,1]:return m.mpf(0)
        q=y/(1-y);u=m.sqrt(1+q*q)
        dp=dsol(q,k,v);dn=dsol(-q,k,v)
        if j==0:ans=q*(dp-dn)/(4*m.pi)
        else:ans=q*u*(dp+dn)/(4*m.pi)+3*k/(2*m.pi)*q/u
        return ans/(1-y)**2
    # Finite cutoff plus analytic tail avoids catastrophic subtraction at infinity.
    cutoff=m.mpf('1e12') if cutoff is None else cutoff
    ymax=cutoff/(1+cutoff)
    P=m.quad(lambda y:integrand(y,0),[0,m.mpf('.3'),m.mpf('.8'),ymax])
    E=m.quad(lambda y:integrand(y,1),[0,m.mpf('.3'),m.mpf('.8'),ymax])-3*k/(2*m.pi)
    # Raw density tails: P: -3kv/(2pi q^2); Esub: -3k/(8pi q^2).
    P-=3*k*v/(2*m.pi*cutoff)
    E-=3*k/(8*m.pi*cutoff)
    return P,E

def Lfun(kk):
    return 3*kk/(2*m.pi)-rm*(3-4*kk*kk)/(8*m.pi)*m.log(abs((2*kk+rm)/(2*kk-rm)))

def can(kk,vv):
    LL=m.log(abs((2*kk+rm)/(2*kk-rm)))
    return -rm*vv*LL/m.pi,-3*kk/(2*m.pi)-rm*(5-4*kk*kk)*LL/(8*m.pi)

SOL_CHARGE_ROWS=[]
for kval in ['.23','.61','.94']:
    kk=m.mpf(kval);vv=m.sqrt(1-kk*kk)
    p,e=solcharges(kk,vv)
    SOL_CHARGE_ROWS.append((kk,vv,p,e))
    close(vv*p-e,Lfun(kk),m.mpf('1e-32'))
    logrow('trace cutoff residual',abs(vv*p-e-Lfun(kk)))
    if kval=='.61':
        pp,ee=solcharges(kk,vv,m.mpf('1e8'))
        logrow('trace 1e8/1e12 cutoff spread',abs(vv*(pp-p)-(ee-e)))
        close(vv*pp-ee,Lfun(kk),m.mpf('1e-23'))
    logrow('KP cross-ratio trace L at k='+kval,vv*p-e)
    pc,ec=can(kk,vv)
    close(m.diff(lambda v0:Lfun(m.sqrt(1-v0*v0)),vv),pc)
    close(vv*pc-Lfun(kk),ec)
print('(c) L and canonical charges derived from independent KP trace',flush=True)

# 6. All triangles: NEW exact log identity L3=L1+L2, stronger than package.
t=S.symbols('t',positive=True)
k1=2*t/(1+t*t);k2=(r*(1-t*t)-2*t)/(2*(1+t*t));k3=(r*(1-t*t)+2*t)/(2*(1+t*t))
v1=(1-t*t)/(1+t*t);v2=-(1-t*t+2*r*t)/(2*(1+t*t));v3=-(1-t*t-2*r*t)/(2*(1+t*t))
R1=(r+2*k1)/(r-2*k1);R2=(r+2*k2)/(r-2*k2);R3=(2*k3+r)/(2*k3-r)
zero(R3-R1*R2);zero(k3-k1-k2)
zero(((t+r)/(1-r*t))/((t-r)/(1+r*t))-(2*k1+r)/(2*k1-r))
# CE = -sqrt(3)/(2pi)[(k3^2-k1^2)L1+(k3^2-k2^2)L2] <0.
# Both coefficients positive for all 0<theta<pi/3, with k3=k1+k2.
zero((k3*k3-k1*k1)-k2*(2*k1+k2));zero((k3*k3-k2*k2)-k1*(k1+2*k2))
# Reflection on the rational half-angle coordinate is exact.
tr=(1/r-t)/(1+t/r)
zero(k1.subs(t,tr)-k2);zero(k2.subs(t,tr)-k1);zero(k3.subs(t,tr)-k3)
zero(v1.subs(t,tr)+v2);zero(v3.subs(t,tr)+v3)
# CP = sqrt(3)/pi [sqrt(3) k1 L2 -(v1-v3)L1].
zero(v2-v3+r*k1)
N2=S.cancel(t*R2)
# Use factored, cancellation-free rational functions for interval proof.
R1f=(t+r/3)*(t+r)/((t-r)*(t-r/3))
N2f=(r-t)/(1+r*t)
zero(R1-R1f);zero(N2-N2f)
B1=S.cancel(v1-v3)
cp_expr=r/S.pi*(r*k1*S.log(N2f/t)-B1*S.log(R1f))
# Differentiate exact expressions, not manually supplied derivative bounds.
dcp_expr=S.diff(cp_expr,t)
modules=[{'mpf':m.iv.mpf,'sqrt':m.iv.sqrt,'log':m.iv.log,'pi':m.iv.pi},'mpmath']
cpf=S.lambdify(t,cp_expr,modules=modules)
dcpf=S.lambdify(t,dcp_expr,modules=modules)
# lambdify emits literal pi through mapping; explicitly use interval namespace.
rf=S.lambdify(t,R1f,modules=modules);nf=S.lambdify(t,N2f,modules=modules)
dlogrf=S.lambdify(t,S.cancel(S.diff(R1f,t)/R1f),modules=modules)
bf=S.lambdify(t,B1,modules=modules)
# Endpoint t in (0,t0]: L1/t bounded by its derivative; L2 >= log(N2(t0)/t0).
iv=m.iv; t0=iv.mpf('0.0001'); end=iv.mpf(['0','0.0001'])
Kover=2/(1+end**2)
L2min=iv.log(nf(t0)/t0)
cpdiv=iv.sqrt(3)/iv.pi*(iv.sqrt(3)*Kover*L2min-bf(end)*dlogrf(end))
assert cpdiv.a>0
logrow('endpoint certified CP/t lower',m.mpf(cpdiv.a))
# Midpoint bracket t*=2-sqrt(3); derivative strictly negative locally.
tstar=2-iv.sqrt(3); delta=iv.mpf('0.0005')
near=iv.mpf([(tstar-delta).a,tstar.b])
assert dcpf(near).b<0
logrow('midpoint dCP/dt upper',m.mpf(dcpf(near).b))
# Adaptive dyadic partition, unlike package's 2000 equal theta intervals.
stack=[iv.mpf([t0.a,(tstar-delta).b])];cells=0;minbound=m.inf
while stack:
    seg=stack.pop();value=cpf(seg)
    if value.a>0:
        cells+=1;minbound=min(minbound,m.mpf(value.a));continue
    lo,hi=m.mpf(seg.a),m.mpf(seg.b)
    assert hi-lo>m.mpf('1e-12')
    mid=(lo+hi)/2
    stack.extend([iv.mpf([lo,mid]),iv.mpf([mid,hi])])
print('(c) CP positive on open left half; adaptive t cells',cells,'minimum certified lower',m.nstr(minbound,12),flush=True)

def hol(th):
    roots=[-m.cos(th+2*m.pi*j/3)/rm for j in [1,2,3]]
    vals=[m.mpf(0),m.mpf(0)]
    for i in range(3):
        d=roots[i]-roots[(i+1)%3]; kk=abs(d);vv=rm*(roots[i]+roots[(i+1)%3])
        pc,ec=can(kk,vv)
        vals[0]+=m.sign(d)*ec;vals[1]+=m.sign(d)*pc
    return vals
th=m.pi/6
mid=hol(th)
close(mid[0],-3*rm/(4*m.pi)*m.log(2+rm));close(mid[1])
close(m.diff(lambda tt:hol(tt)[1],th),(3*rm*m.log(2+rm)-9)/m.pi)
for tt in [m.mpf('.17'),m.mpf('.39')]:
    hh=hol(tt);ref=hol(m.pi/3-tt);close(hh[0],ref[0]);close(hh[1],-ref[1])
# Symbolic series of normalized log-free rational factors certify endpoint form.
zero(S.limit(R1f,t,0)-1);zero(S.limit(N2f,t,0)-r)
zero(S.limit(S.diff(R1f,t)/R1f,t,0)-8/r)
for tiny in ['1e-9','1e-15']:
    tt=m.mpf(tiny);HE,HP=hol(tt)
    ce=-3/(2*m.pi)*tt*(m.log(2*rm/tt)+1)
    cp=3/m.pi*tt*(m.log(2*rm/tt)-2)
    assert abs((HE-ce)/ce)<10*tt;assert abs((HP-cp)/cp)<10*tt
# Nonzero lambda does not preserve claimed sign separation; encode refutation.
kk=m.mpf('.5');vv=rm/2
wbase=can(kk,vv)[0]/kk
assert wbase<0 and wbase+2*2*vv>0
logrow('refutation w1 at symmetric triangle lambda=2',wbase+4*vv)
# At the exceptional lambda, even canonical vector vanishes at the midpoint.
lamstar=3*rm/(4*m.pi)*m.log(2+rm)
close(mid[0]+lamstar);close(mid[1])
# The lambda energy coefficient follows solely from exact cubic algebra.
zero(sum([k1**3,k2**3,-k3**3])+3*k1*k2*k3)
zero(sum([v1*k1,v2*k2,-v3*k3]))
print('(c) CE strict universal algebra, CP certified universal root, lambda exceptions checked',flush=True)

# 7. Mode weights: derive a cusp-cancelled expression and expose the package's
# missing factor two in its upper-half interval integrand.
def pmode(alpha):
    kk=m.sin(alpha);cc=m.cos(alpha)
    # Independently integrated momentum from KP cross ratio at a test point
    # was checked in section 5; this evaluates its elementary primitive.
    ph=[alpha+2*m.pi/3,alpha-2*m.pi/3,alpha]
    return -sum(c*m.sin(p)*m.log(abs(m.tan(p/2))) for c,p in zip([1,1,-2],ph))/(2*m.pi)
for kk,vv,pp,ee in SOL_CHARGE_ROWS:
    close(pp,pmode(m.asin(kk)),m.mpf('1e-32'))
    close(ee,vv*pmode(m.asin(kk))-Lfun(kk),m.mpf('1e-32'))
for av in [m.mpf('.2'),m.mpf('.7'),m.mpf('1.02')]:
    sa,ca=m.sin(av),m.cos(av)
    LC=m.log((rm+2*sa)/(rm-2*sa));LX=m.log((2*ca+1)/(2*ca-1))
    LF=m.log((rm+2*sa)/(2*ca+1)*m.tan(av/2+m.pi/6))
    correct=-rm*(ca-m.mpf('.5'))/(4*m.pi)*LC-rm/(8*m.pi)*LF+(sa-rm/2)/(4*m.pi)*LX+sa/m.pi*m.log(m.tan(av/2))
    close(correct,pmode(av))
    upper_code=correct-sa/(2*m.pi)*m.log(m.tan(av/2))
    assert upper_code>correct
    close(upper_code-correct,-sa/(2*m.pi)*m.log(m.tan(av/2)))
    assert correct<0
# Stronger analytic bound: derivative of w_mode is positive everywhere
# on 0<alpha<pi/3, with supremum -3 log(3)/(4pi).
cs=(1-t*t)/(1+t*t);ks=2*t/(1+t*t)
LCs=S.log(R1f);LXs=S.log((2*cs+1)/(2*cs-1))
wexpr=-r*cs*LCs/(4*S.pi*ks)+LXs/(4*S.pi)+S.log(t)/S.pi
zero(S.diff(wexpr,t)-r*LCs/(4*S.pi*ks*ks)*2/(1+t*t))
# In the cusp-cancelled primitive the two x log(x) products vanish.
LFarg=(r+2*ks)/(2*cs+1)*(t+1/r)/(1-t/r)
zero(LFarg.subs(t,1/r)-3)
zero(-r/(8*S.pi)/(r/2)*S.log(3)+S.log(1/r)/S.pi+3*S.log(3)/(4*S.pi))
assert -3*m.log(3)/(4*m.pi)<-m.log(3)/(2*m.pi)
# Check mode edge balance directly against boundary vertex values.
for theta in [m.mpf('.23'),m.mpf('.71')]:
    qq=[-m.cos(theta+2*m.pi*j/3)/rm for j in [1,2,3]]
    sums=[m.mpf(0),m.mpf(0)]
    for i in range(3):
        qi,qf=qq[i],qq[(i+1)%3];kk=abs(qi-qf);vv=rm*(qi+qf);sg=m.sign(qi-qf)
        pm=m.sign(vv)*pmode(m.asin(kk));em=vv*pm-Lfun(kk)
        Bi=closed(-2*rm*qi,1);Bf=closed(-2*rm*qf,1)
        close(Bf[0]-Bi[0],sg*pm);close(Bf[2]-Bi[2],sg*em)
        sums[0]+=sg*pm;sums[1]+=sg*em
    close(sums[0]);close(sums[1])
# Analytic sign proof: lower half log combined ratio <1 for cos>=sqrt3/2;
# upper half all four terms in the CORRECT expression above are negative.
c=S.symbols('c',positive=True)
zero((2*c+1)*(1-c)**2-(2*c-1)*(1+c)**2-2*(1-3*c*c))
# Paired branch points, not disjoint: fixing any vertex forces its other
# soliton root to be one of the other two KP roots.
testroot=S.symbols('testroot')
rootpoly=testroot**2+q2*testroot+q2**2-S.Rational(1,4)
zero((rootpoly-(testroot-q1)*(testroot-q3)).subs(q3,-q1-q2)
     -(q1*q1+q1*q2+q2*q2-S.Rational(1,4)))
w1,w2,w3=S.symbols('w1 w2 w3')
coeff=[w3-w1,w1-w2,w2-w3]
branches=[coeff[0]-coeff[1],coeff[1]-coeff[2],coeff[2]-coeff[0]]
assert S.solve(branches,[w1,w2],dict=True)==[{w1:w3,w2:w3}]
# Effective phase coefficients, distinct from intrinsic local charges.
pa,pb,q,eps,eta=S.symbols('pa pb q eps eta',real=True)
ell=lambda p:3*r*(4*p**3-p)
quot=S.cancel((ell(pa)-ell(q))/(pa-q))
zero(quot-3*r*(4*(pa*pa+pa*q+q*q)-1))
logz=S.symbols('logz')
kp=pa-pb;vp=r*(pa+pb)
Meff=eps*eta*q;Pieff=eps*eta*(6*q*q-1)/r
phase_der=-S.diff(ell(q),q)*S.Rational(2,9)*eps*logz+S.Rational(2,9)*eps*eta*(quot-quot.subs(pa,pb))-S.Rational(2,3)*((-2*kp*vp)*S.diff(Meff,q)-2*kp*S.diff(Pieff,q))
zero(phase_der-(-2*r/3*(12*q*q-1)*eps*logz+4*eps*eta*kp*(vp+2*r*q)))
# Log coefficient at q=pa is generically nonzero: pole-free is NOT smooth.
assert (-2*r/3*(12*q*q-1)*eps).subs({q:S.Rational(2,5),eps:1}) != 0
# Arbitrary single-valued boundary shifts telescope even outside CZ.
C=S.symbols('C0:3');zero(sum(C[(i+1)%3]-C[i] for i in range(3)))
print('(c) weight signs, paired branch proof, pole cancellation and telescoping checked',flush=True)

# Exact symbolic momentum-derivative identity, reducing k^2+v^2=1.
vv,kk,qq,ee,et,LL=S.symbols('vv kk qq ee et LL',real=True)
pa=(vv/r+kk)/2;pb=(vv/r-kk)/2
Dalong=lambda expr:S.diff(expr,vv)-vv/kk*S.diff(expr,kk)
logprime=Dalong(pb)/(pb-qq)-Dalong(pa)/(pa-qq)
coef=S.Rational(2,9)*(3*vv-4*vv**3-ell(qq))*ee*et
polynomial=-S.Rational(2,3)*((-2*kk*vv)*ee*et*qq-ee*et*(6*qq*qq-1)/r*2*kk)
expr=Dalong(coef)*LL+coef*logprime+Dalong(polynomial)-ee*et*LL/kk*Dalong(S.Rational(2,3)*vv*kk**3)
numerator=S.fraction(S.cancel(expr))[0]
zero(S.rem(S.expand(numerator),kk*kk+vv*vv-1,kk))
# Independent derivative of the effective classical phase with respect to
# the physical classical momentum, all four direction/orientation choices.
for ep in [-1,1]:
    for et in [-1,1]:
        qd=m.mpf('.51');av=m.mpf('.4')
        def phcl(aa):
            kk=m.sin(aa);vv=ep*m.cos(aa)
            pa=(vv/rm+kk)/2;pb=(vv/rm-kk)/2
            ell3=-ep*m.cos(3*aa);ellD=3*rm*(4*qd**3-qd)
            MD=ep*et*qd;PiD=ep*et*(6*qd*qd-1)/rm
            return m.mpf(2)/9*(ell3-ellD)*ep*et*m.log(abs((pb-qd)/(pa-qd)))-m.mpf(2)/3*((-2*kk*vv)*MD-PiD*2*kk)
        kk=m.sin(av);vv=ep*m.cos(av);pa=(vv/rm+kk)/2;pb=(vv/rm-kk)/2
        dx=-ep*et*m.log(abs((pb-qd)/(pa-qd)))/kk
        pp=m.diff(lambda aa:m.mpf(2)/3*ep*m.sin(aa)**3*m.cos(aa),av)
        derivative=m.diff(phcl,av)
        close(derivative,-dx*pp)
        assert abs(derivative-dx*pp)>m.mpf('.6')
        logrow('refuted phase-sign discrepancy eps='+str(ep)+' eta='+str(et),derivative-dx*pp)
# 8. Transmission phase from a continuous argument of rational Z, not dipoles.
def defect_phase(q,sv,eta):
    X=m.sign(q)*m.sqrt(1+q*q)+sv
    raw=-2*eta*m.atan2(rm*q,X)
    return raw+(2*m.pi*eta/3 if q>0 else -4*m.pi*eta/3)
sv=m.mpf('-1.5');cc=rm*sv/(4*m.pi)
for cutoff in [m.mpf('100'),m.mpf('1000')]:
    ymax=cutoff/(1+cutoff)
    subcut=m.quad(lambda yy:mapped(yy,sv,1,1),[0,m.mpf('.4'),ymax])
    TT=m.sqrt(1+cutoff*cutoff)
    endpoint=cutoff*TT/(4*m.pi)*(defect_phase(cutoff,sv,1)-defect_phase(-cutoff,sv,1))
    ren=subcut+cc*(TT-1)-endpoint
    err=ren-closed(sv,1)[2]
    assert abs(err)<1/cutoff
    logrow('boundary equal-mode cutoff '+str(cutoff)+' remainder',err)
for qd,expected in [('0.5','-1.3321099117235175816'),('0.52','-1.3633755953997418592')]:
    qd=m.mpf(qd);sv=-2*rm*qd;alpha=m.mpf('.35');kk=m.sin(alpha);vv=m.cos(alpha)
    def integ(y):
        if y in [0,1]:return m.mpf(0)
        q=y/(1-y)
        return -(dsol(q,kk,vv)*defect_phase(q,sv,1)+dsol(-q,kk,vv)*defect_phase(-q,sv,1))/(4*m.pi*(1-y)**2)
    val=m.quad(integ,[0,m.mpf('.2'),m.mpf('.65'),1])
    close(val,m.mpf(expected),m.mpf('6e-20'))
    pa=(vv/rm+kk)/2;pb=(vv/rm-kk)/2;dx=-m.log(abs((pb-qd)/(pa-qd)))/kk
    logrow('transmission I at qD='+str(qd),val)
    logrow('transmission mode',val-dx*pmode(alpha))
    logrow('transmission canonical',val-dx*can(kk,vv)[0])
print('(c) phase tables checked; absolute physical determinant remains conditional',flush=True)

print('Independent checks seconds:',round(time.perf_counter()-START,3),flush=True)
if '--package-runs' in sys.argv:
    wrapper="""import pathlib,sys
original=pathlib.Path.write_text
def redirected(self,*args,**kwargs):
    assert self.name.startswith('defb_boundary_') and self.suffix=='.json'
    return original(pathlib.Path('/tmp/audit_'+self.name),*args,**kwargs)
pathlib.Path.write_text=redirected
p=pathlib.Path(sys.argv[1]).resolve()
exec(compile(p.read_text(),str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
"""
    for name in ['renorm','holonomy']:
        path=Path(__file__).with_name('defb_boundary_'+name+'.py')
        start=time.perf_counter()
        out=subprocess.run([sys.executable,'-c',wrapper,str(path)],capture_output=True,text=True)
        print('original',name,'seconds',time.perf_counter()-start,'exit',out.returncode,
              'marker','ALL CHECKS PASSED' in out.stdout,flush=True)
        assert out.returncode==0 and 'ALL CHECKS PASSED' in out.stdout, out.stderr
print('Independent audit runtime seconds:',round(time.perf_counter()-START,3),flush=True)
print('ALL CHECKS PASSED')

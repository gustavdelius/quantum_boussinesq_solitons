#!/usr/bin/env python3
"""Independent defq audit. No imports from project verification modules.
All mathematical results are (c); physical interpretations remain (d).
Uses action variation, jet algebra, independent linear solves and 70-digit
quadrature. Refutations assert the discrepancy, never the false equality.
"""
import sympy as sy
import mpmath as mp
mp.mp.dps = 70
R = sy.sqrt(3)
r = mp.sqrt(3)
TOL = mp.mpf('1e-55')

def zero(x):
    assert sy.cancel(sy.expand(x)) == 0, sy.factor(x)

def close(x, y=0, tol=TOL):
    assert abs(x-y) < tol, (mp.nstr(x,25), mp.nstr(y,25))

# Independent CZ action, then field dictionary; free rho0 retained until vacuum.
a,b,x,y,t,u,xt,yt,s,rr,z = sy.symbols('a b x y t u xt yt s rr z', real=True)
xx,yy,xxx,yyy = sy.symbols('xx yy xxx yyy', real=True)
variables = [a,b,x,y,t,u,xt,yt,xx,yy,xxx,yyy]
# Derive the bulk flux primitives by total jet differentiation after the PDE.
x4,xxt=sy.symbols('x4 xxt')
pde=xx-x4-6*x*xx
Dx=lambda f: sum(sy.diff(f,v)*w for v,w in [(x,xx),(xx,xxx),(xxx,x4),(t,xt),(xt,xxt)])
BE=t*(x-3*x*x-xxx)+xt*xx
BP=-t*t/2-x*x/2+2*x**3+x*xxx-xx*xx/2
zero(t*pde+(x-3*x*x)*xt+xx*xxt-Dx(BE))
zero(-pde*x-t*xt-Dx(BP))
zero(pde-Dx(x-3*x*x-xxx))
zero(((-3*t)**2/2-(-3*x)**2/2-(-3*x)**3/3-(-3*xx)**2/2)/9
     -(t*t/2-x*x/2+x**3-xx*xx/2))
models = {}
nonlinear = {}
for eta in (1,-1):
    alpha = sy.Integer(eta)/R
    UU,VV=sy.symbols('UU VV')
    sigCZ=s+alpha*(UU-VV)
    gauge=-sigCZ**2/(4*alpha)
    zero(sy.diff(gauge,VV)-sy.diff(gauge,UU)-sigCZ)
    # sigma = sigma0 + (U-V)/(3 alpha), U=-3a, V=-3b.
    sigma = s-(a-b)/alpha
    rho_z = rr + (z-z**3/3)/(2*alpha)
    tau_z = sy.integrate(-z*rho_z,z)
    h_z = sy.integrate(-rho_z,z)
    rho = rho_z.subs(z,sigma)
    tau = (tau_z.subs(z,sigma)-tau_z.subs(z,s))
    h = (h_z.subs(z,sigma)-h_z.subs(z,s))
    U,V,UX,VX,UT,VT = -3*a,-3*b,-3*x,-3*y,-3*t,-3*u
    FC = alpha*UX+2*alpha*VX-sigma**2/(4*alpha)
    GC = 2*alpha*UX+alpha*VX-sigma**2/(4*alpha)
    HC = alpha*sigma*(UX**2+UX*VX+VX**2)+rho*(UX+VX)+tau
    D = sy.expand((UT*FC+VT*GC+HC)/9)
    dt = lambda f: sum(sy.diff(f,v)*w for v,w in [(a,t),(b,u),(x,xt),(y,yt)])
    F,G = sy.diff(D,t),sy.diff(D,u)
    E = sy.expand(t*F+u*G-D)
    sewing = {xx:sy.diff(D,x), yy:-sy.diff(D,y),
              xxx:x-3*x*x-sy.diff(D,a)+dt(F),
              yyy:y-3*y*y+sy.diff(D,b)-dt(G)}
    # Derive localized momentum by integrating the exact boundary-flux one-form.
    flux = lambda v,w,j,k: -v*v/2-w*w/2+2*w**3+w*k-j*j/2
    needed = sy.expand(-(flux(t,x,xx,xxx)-flux(u,y,yy,yyy)).subs(sewing))
    coeff = [sy.diff(needed,v) for v in [t,u,xt,yt]]
    zero(needed-sum(c*v for c,v in zip(coeff,[t,u,xt,yt])))
    PD=sy.Integer(0)
    for v,c in zip([a,b,x,y],coeff):
        PD += sy.integrate(sy.expand(c-sy.diff(PD,v)),v)
    zero(dt(PD)-needed)
    delta=sy.symbols('delta')
    perturbed={xx:sewing[xx], yy:sewing[yy], xxx:sewing[xxx]-delta, yyy:sewing[yyy]-delta}
    difference=sy.expand((flux(t,x,xx,xxx)-flux(u,y,yy,yyy)).subs(perturbed)
                        -(flux(t,x,xx,xxx)-flux(u,y,yy,yyy)).subs(sewing))
    zero(difference+(x-y)*delta)
    # Fix vacuum reference; compare to the claimed compact primitive.
    expectedP=-alpha*(x*x+4*x*y+y*y)/2+h/9
    zero(dt(PD-expectedP))
    PD=expectedP
    Pi=F+G
    MD=b-a
    energyflux=lambda v,w,j,k,wt: v*(w-3*w*w-k)+wt*j
    zero((energyflux(t,x,xx,xxx,xt)-energyflux(u,y,yy,yyy,yt)+dt(E)).subs(sewing))
    zero((x-3*x*x-xxx-y+3*y*y+yyy+dt(Pi)).subs(sewing))
    zero(t-u+dt(MD))
    nonlinear[eta] = (
        sy.lambdify([s]+variables,[v.subs({rr:-(s-s**3/3)/(2*alpha)}) for v in [E,PD,Pi,MD]],'mpmath'),
        sy.lambdify([s]+variables,[v.subs({rr:-(s-s**3/3)/(2*alpha)}) for v in
            [xxx-sewing[xxx],xx-sewing[xx],yyy-sewing[yyy],yy-sewing[yy]]],'mpmath'))
    # Independently generated quadratic Taylor coefficients at rho(s)=0.
    vacuum={rr:-(s-s**3/3)/(2*alpha)}
    eps=sy.symbols('eps')
    scaling={v:eps*v for v in [a,b,x,y,t,u,xt,yt]}
    quad=lambda expr: sy.expand(expr.subs(vacuum).subs(scaling, simultaneous=True)).coeff(eps,2)
    D2,E2,P2=map(quad,[D,E,PD])
    F1,G1=sy.diff(D2,t),sy.diff(D2,u)
    zero(F1-(alpha*x+2*alpha*y-s*(a-b)/2))
    zero(P2-(-alpha*(x*x+4*x*y+y*y)/2-alpha*(1-s*s)*(a-b)**2/4))
    linear=[xxx-x+sy.diff(D2,a)-dt(F1), sy.diff(D2,x)-xx,
            y-yyy+sy.diff(D2,b)-dt(G1),sy.diff(D2,y)+yy]
    linearfun=sy.lambdify([s]+variables,linear,'mpmath')
    Efun=sy.lambdify([s,a,b,x,y,t,u],E2,'mpmath')
    Pfun=sy.lambdify([s,a,b,x,y,t,u],P2,'mpmath')
    Kfun=sy.lambdify([s,a,b,x,y,t,u],t*F1+u*G1,'mpmath')
    models[eta]=(linearfun,Efun,Pfun,Kfun)
    # KP quotient proof, signed k=a_root-b_root avoids absolute-value branches.
    p,q=sy.symbols('p q',real=True)
    si,sf=-2*R*p,-2*R*q
    vacfinal={x:0,y:0,a:-(sf-si)*alpha,b:0}
    k=p-q; v=R*(p+q)
    sol=[sy.Rational(2,15)*k**3*(5-4*k*k),sy.Rational(2,3)*k**3*v,-2*k*v,2*k]
    for charge,target in zip([E,PD,Pi,MD],sol):
        final=charge.subs(vacfinal).subs(s,si).subs(rr,-(si-si**3/3)/(2*alpha))
        initial=charge.subs({x:0,y:0,a:0,b:0}).subs(s,si).subs(rr,-(si-si**3/3)/(2*alpha))
        zero(sy.rem(sy.expand(final-initial-eta*target),p*p+p*q+q*q-sy.Rational(1,4),p))
print('(c) action-derived charges, arbitrary-jet flux balance and KP quotient: PASS (both eta)')

# Build mode columns by evaluating freshly differentiated sewing equations.
def column(sig,eta,lam,rate,side):
    j=[mp.mpc(0)]*12
    for idx,val in ([(0,1),(2,lam),(4,rate),(6,rate*lam),(8,lam**2),(10,lam**3)] if side==0 else
                    [(1,1),(3,lam),(5,rate),(7,rate*lam),(9,lam**2),(11,lam**3)]):
        j[idx]=val
    return mp.matrix(models[eta][0](sig,*j))

def matcols(cols):
    return mp.matrix([[col[j] for col in cols] for j in range(4)])

def transmission(q,sig,eta):
    w=abs(q)*mp.sqrt(1+q*q)
    L=column(sig,eta,1j*q,-1j*w,0)
    B=column(sig,eta,1j*q,-1j*w,1)
    idx=max(range(4),key=lambda j:abs(B[j]))
    zz=-L[idx]/B[idx]
    assert max(abs(L[j]+zz*B[j]) for j in range(4)) < TOL*max(1,abs(q)**3)
    return zz

def rational_z(q,sig,eta):
    X=mp.sign(q)*mp.sqrt(1+q*q)+sig
    return ((X-1j*r*q)/(X+1j*r*q))**eta

def phaseprime(q,sig,eta):
    return mp.im(mp.diff(lambda v:rational_z(v,sig,eta),q)/rational_z(q,sig,eta))

def average(fun,q,sig,eta):
    zz=transmission(q,sig,eta); w=abs(q)*mp.sqrt(1+q*q)
    amps=[1,zz,1j*q,1j*q*zz,-1j*w,-1j*w*zz]
    # Quarter-cycle polarization identity; standard vacuum covariance 1/(4 pi omega).
    return (fun(sig,*[mp.re(v) for v in amps])+fun(sig,*[mp.im(v) for v in amps]))/(4*mp.pi*w)

maxtrace=mp.mpf(0)
for eta in (1,-1):
    for sig in map(mp.mpf,['-2.5','-1.5','0.3','1.5','2.4']):
        for q in map(mp.mpf,['0.17','0.8','2.3']):
            close(transmission(q,sig,eta),rational_z(q,sig,eta))
            pp=sum(average(models[eta][2],sign*q,sig,eta) for sign in (1,-1))
            rhs=q*(phaseprime(q,sig,eta)-phaseprime(-q,sig,eta))/(4*mp.pi)
            maxtrace=max(maxtrace,abs(pp-rhs)); close(pp,rhs)
            K=sum(average(models[eta][3],sign*q,sig,eta) for sign in (1,-1))
            Ed=sum(average(models[eta][1],sign*q,sig,eta) for sign in (1,-1))
            Erhs=abs(q)*mp.sqrt(1+q*q)*(phaseprime(q,sig,eta)+phaseprime(-q,sig,eta))/(4*mp.pi)
            close(Ed,K); close(K,Erhs)  # J_bulk=0 for these plane waves.
            assert abs(-K-Erhs)>mp.mpf('1e-8') # refutes the printed minus sign.
print('(c) momentum trace max residual',mp.nstr(maxtrace,5),'; printed energy-trace minus sign REFUTED')
# Printed cosine normalization is inconsistent unless a further half weight is supplied.
close(mp.quad(lambda h:mp.cos(h)**2,[0,2*mp.pi])/(2*mp.pi),mp.mpf('0.5'))
assert mp.mpf('0.5') != mp.mpf('0.25')

# Determinant away from zeros, both orientations. Spectrum parameter is eta*sigma.
def polynomial(S,c,branch):
    return S*S-(r+branch)*c*S/2+(c*c-2+branch*r)/2

def rates(c):
    values=[]
    for branch in (-1,1):
        roots=mp.polyroots([1,-(r+branch)*c/2,(c*c-2+branch*r)/2],maxsteps=200)
        for S in roots:
            if abs(mp.im(S))<TOL and mp.re(S)>1+TOL:
                values.append((mp.re(S)**2-1)/2)
    return sorted(values)

for eta in (1,-1):
    for sig in map(mp.mpf,['-1.9','-0.5','1.5','1.9']):
        for gg in map(mp.mpf,['0.17','0.7']):
            ks=mp.polyroots([1,0,-1,0,gg*gg],maxsteps=300)
            ks=[k for k in ks if mp.re(k)>0]
            kk,ll=ks
            S=kk+ll
            M=matcols([column(sig,eta,kk,gg,0),column(sig,eta,ll,gg,0),column(sig,eta,-kk,gg,1),column(sig,eta,-ll,gg,1)])
            rhs=-mp.mpf(2)/3*S*S*(S*S-1)*polynomial(S,eta*sig,-1)*polynomial(S,eta*sig,1)
            close(mp.det(M)/(kk-ll)**2,rhs)
        for gg in rates(eta*sig):
            ks=mp.polyroots([1,0,-1,0,gg*gg],maxsteps=300)
            ks=[k for k in ks if mp.re(k)>0]
            M=matcols([column(sig,eta,ks[0],gg,0),column(sig,eta,ks[1],gg,0),column(sig,eta,-ks[0],gg,1),column(sig,eta,-ks[1],gg,1)])
            close(mp.det(M))
close(rates(r)[0],r/4)
assert abs(rates(r)[0]-(2*r-1)/8)>mp.mpf('0.1')
for n in range(1,102):
    if n in (34,68): continue
    qs=[-mp.cos(mp.pi*n/102+2*j*mp.pi/3)/r for j in range(3)]
    for eta in (1,-1):
        counts=sorted(len(rates(-2*r*eta*q)) for q in qs)
        assert counts==([0,1,1] if n in (17,51,85) else [0,1,2]),(n,counts)
print('(c) full sewing determinants and KP counts PASS; threshold rate =',mp.nstr(r/4,16))
print('(c) eta=-1 sigma=-1.9 rates', [mp.nstr(v,16) for v in rates(mp.mpf('1.9'))])
# Wrong scattering-root replacement genuinely changes waves; not a spectrum fix.
q=mp.mpf('.8'); sig=mp.mpf('2.4')
assert abs(transmission(q,sig,-1)-rational_z(q,-sig,-1))>mp.mpf('.1')

# Time-dependent check uses the logistic field coordinate as integration variable,
# rather than the package's spatial tanh quadrature.
def soljet(k,v,h):
    x=2*k*k*h*(1-h)
    xx=2*k**3*h*(1-h)*(1-2*h)
    xxx=2*k**4*h*(1-h)*(1-6*h+6*h*h)
    return [2*k*h,x,-v*x,-v*xx,xx,xxx]
def interlace(L,B):
    return [z for pair in zip(L,B) for z in pair]
def bulkcharges(k,v,lo,hi):
    def integrand(h,idx):
        x=2*k*k*h*(1-h);xx=2*k**3*h*(1-h)*(1-2*h)
        # Polynomial densities divided by k h(1-h); handle endpoints by continuity.
        if h==0 or h==1:return [0,0,-2*k*v,2*k][idx]
        return [(1+v*v)*x*x/2-x**3+xx*xx/2,v*x*x,-v*x,x][idx]/(k*h*(1-h))
    return [mp.quad(lambda h:integrand(h,j),[lo,hi]) for j in range(4)]
for eta in (1,-1):
    for absorb in (False,True):
        aa=mp.mpf('.4');k=mp.sin(aa);v=mp.cos(aa)
        sig=eta*r*k-v if absorb else mp.mpf('2.4')
        zz=0 if absorb else (v+sig-eta*r*k)/(v+sig+eta*r*k)
        previous=None
        for time in map(mp.mpf,['-3','.27','4']):
            hL=1/(1+mp.exp(k*v*time))
            hR=mp.mpf(0) if absorb else zz/(mp.exp(k*v*time)+zz)
            jets=interlace(soljet(k,v,hL),soljet(k,v,hR))
            residual=nonlinear[eta][1](sig,*jets)
            assert max(abs(z) for z in residual)<TOL
            bd=nonlinear[eta][0](sig,*jets)
            BL=bulkcharges(k,v,0,hL)
            BR=[0]*4 if absorb else bulkcharges(k,v,hR,1)
            total=[BL[i]+BR[i]+bd[i] for i in range(4)]
            if previous is not None:
                for one,two in zip(total,previous):close(one,two)
            previous=total
        if eta==-1 and not absorb:
            wrong=(v-sig-eta*r*k)/(v-sig+eta*r*k)
            wrongh=wrong/(mp.exp(k*v*mp.mpf('.27'))+wrong)
            bad=nonlinear[eta][1](sig,*interlace(soljet(k,v,1/(1+mp.exp(k*v*mp.mpf('.27')))),soljet(k,v,wrongh)))
            assert max(abs(vv) for vv in bad)>1
            print('(c) soliton convention z',mp.nstr(zz,14),'wrong z',mp.nstr(wrong,14),'wrong residual',mp.nstr(max(abs(vv) for vv in bad),14))
print('(c) independent transmission/absorption quadratures PASS, both eta')
# Reconstruct the package's claimed absorption input without invoking its code.
q1=r/4; q2=(-q1+mp.sqrt(1-3*q1*q1))/2
kv=q1-q2; vv=r*(q1+q2); eta_abs=-1
sig_wrong=-2*r*q1*eta_abs
z_wrong=(vv+sig_wrong-eta_abs*r*kv)/(vv+sig_wrong+eta_abs*r*kv)
sig_correct=-2*r*q1
z_correct=(vv+sig_correct-eta_abs*r*kv)/(vv+sig_correct+eta_abs*r*kv)
close(z_correct); assert z_wrong>1
print('(c) package absorption test actually has z =',mp.nstr(z_wrong,18),'(not zero)')

# UV from independent rational phase: numerical derivative and quadrature,
# not the package's dipole antiderivative. Theta normalized at each infinity.
def theta(q,sig,eta):
    zratio=rational_z(q,sig,eta)/mp.e**(-1j*eta*2*mp.pi/3)
    return mp.arg(zratio)
# Both signed q limits have the SAME transmission constant, contrary to the text.
close(rational_z(-mp.mpf('1e60'),mp.mpf('1.5'),1),mp.e**(-2j*mp.pi/3))
assert abs(rational_z(-mp.mpf('1e60'),mp.mpf('1.5'),1)-mp.e**(2j*mp.pi/3))>1
def defect_modes(sig,eta):
    gamma=mp.acos(sig/2); E=P=mp.mpf(0)
    for psi,weight in [(2*mp.pi/3-gamma,-eta),(-2*mp.pi/3-gamma,eta)]:
        sn,cs=mp.sin(psi),mp.cos(psi)
        I=mp.quad(lambda z:1/(1-cs*cs*z*z),[0,mp.mpf('.9'),1])
        E+=weight*sn*(1+cs*cs*I)/(2*mp.pi)
        P+=weight*sn*cs*I/(2*mp.pi)
    return E,P
for eta,sig in [(1,mp.mpf('-1.5')),(-1,mp.mpf('1.5'))]:
    SD=-eta*r*sig/2
    for cutoff in [mp.mpf('1e3'),mp.mpf('1e5')]:
        integrand=lambda q: q*mp.sqrt(1+q*q)*(phaseprime(q,sig,eta)+phaseprime(-q,sig,eta))/(4*mp.pi)
        dens=mp.quad(integrand,[0,1,10,100,cutoff])
        end=cutoff*mp.sqrt(1+cutoff**2)*(theta(cutoff,sig,eta)-theta(-cutoff,sig,eta))/(4*mp.pi)
        mode=dens-end
        assert abs(dens/cutoff-SD/(2*mp.pi))<2/cutoff
        assert abs(mode/cutoff-SD/mp.pi)<2/cutoff
    finiteE,finiteP=defect_modes(sig,eta)
    assert abs(mode-SD/mp.pi*(mp.sqrt(1+cutoff*cutoff)-1)-finiteE)<10/cutoff
    pdensity=lambda q:r*eta*q*(2*(sig*sig-2)*q*q+sig*sig-1)/(mp.pi*mp.sqrt(1+q*q)*(16*q**4+4*(sig*sig+2)*q*q+(sig*sig-1)**2))
    close(mp.quad(pdensity,[0,1,10,mp.inf]),finiteP)
    Q=mp.mpf('1e20')
    close(Q*mp.sqrt(1+Q*Q)*(phaseprime(Q,sig,eta)+phaseprime(-Q,sig,eta))/(4*mp.pi),SD/(2*mp.pi),mp.mpf('1e-38'))
    print('(c) UV slopes eta',eta,'density',mp.nstr(dens/cutoff,14),'mode',mp.nstr(mode/cutoff,14))
# Counterterm slope and restricted no-potential-deformation result.
md,base,etaS=sy.symbols('md base etaS')
zero(sy.expand(R*etaS*(base+R*etaS*md)/2-R*etaS*base/2-3*md/2).subs(etaS**2,1))
hp,tp=sy.symbols('hp tp')
res=-(x-y)*tp-(t-u)*hp
assert sy.solve(sy.Poly(res,x,y,t,u).coeffs(),[tp,hp])=={tp:0,hp:0}

# Independent quadratures for the bulk mode charges. Integrate the rational
# phase kernel after T=sqrt(1+q^2), then z=1/T. No closed log formula reused.
def modes(v):
    aa=mp.acos(abs(v)); eps=mp.sign(v) if v else 1
    E=P=mp.mpf(0)
    for delta,weight in [(2*mp.pi/3,1),(-2*mp.pi/3,1),(0,-2)]:
        phi=aa+delta; sn,cs=mp.sin(phi),mp.cos(phi)
        I=mp.quad(lambda z:1/(1-cs*cs*z*z),[0,mp.mpf('.9'),1])
        E+=weight*sn*(1+cs*cs*I)/(2*mp.pi)
        P+=eps*weight*sn*cs*I/(2*mp.pi)
    return E,P

def canonical(v):
    E,P=modes(v)
    L=v*P-E
    Pc=mp.diff(lambda vv:vv*modes(vv)[1]-modes(vv)[0],v)
    return v*Pc-L,Pc

def triangle(q):
    d=mp.sqrt(1-3*q*q)
    return [q,(-q+d)/2,(-q-d)/2]

def edges(qs):
    out=[]
    for i in range(3):
        aa,bb=qs[i],qs[(i+1)%3]
        v=r*(aa+bb); k=abs(aa-bb); orient=mp.sign(aa-bb)
        out.append((v,k,orient))
    return out

for seed in ['-.2','.1','.45','-.03','.23']:
    qs=triangle(mp.mpf(seed)); es=edges(qs)
    hm=[mp.mpf(0),mp.mpf(0)]; hc=hm.copy()
    for i,(v,k,o) in enumerate(es):
        em,pm=modes(v); ec,pc=canonical(v)
        before=defect_modes(-2*r*qs[i],1);after=defect_modes(-2*r*qs[(i+1)%3],1)
        close(after[0]-before[0],o*em);close(after[1]-before[1],o*pm)
        for j,val in enumerate([em,pm]): hm[j]+=o*val
        for j,val in enumerate([ec,pc]): hc[j]+=o*val
        close(ec-em,v*(pc-pm))
        # Absorption-locus shift checked as velocity shift, independent of alpha formula.
        pcl=lambda vv:mp.mpf(2)/3*vv*(1-vv*vv)**mp.mpf('1.5')
        ecl=lambda vv:mp.mpf(2)/15*(1-vv*vv)**mp.mpf('1.5')*(1+4*vv*vv)
        close(mp.diff(ecl,v),v*mp.diff(pcl,v))
        dv=(pm-pc)/mp.diff(pcl,v)
        close(pc+mp.diff(pcl,v)*dv,pm);close(ec+mp.diff(ecl,v)*dv,em)
    for val in hm:close(val)
    assert max(abs(val) for val in hc)>mp.mpf('.01')
    print('(c) quadrature canonical holonomy',seed,*[mp.nstr(val,16) for val in hc])
# Symmetric nondegenerate triangle: the momentum component can vanish exactly.
special=[mp.mpf(0),mp.mpf('.5'),mp.mpf('-.5')]
special_h=[sum(o*canonical(v)[j] for v,k,o in edges(special)) for j in range(2)]
close(special_h[1]);assert abs(special_h[0])>mp.mpf('.1')
print('(c) symmetric triangle holonomy',*[mp.nstr(val,16) for val in special_h])
# Exact telescoping for completely arbitrary vertex counterterms.
Q0,Q1,Q2=sy.symbols('Q0 Q1 Q2');zero((Q1-Q0)+(Q2-Q1)+(Q0-Q2))

# Independent Hirota polynomial and root ratios, both orientations and directions.
qs=triangle(mp.mpf('-.2')); es=edges(qs)
for eps in (1,-1):
    k3=mp.sin(mp.mpf('.35'));v3=eps*mp.cos(mp.mpf('.35'))
    pa=(v3/r+k3)/2;pb=(v3/r-k3)/2
    H=lambda k,w:w*w-k*k+k**4
    for eta in (1,-1):
        logs=[eps*eta*mp.log(abs((pb-q)/(pa-q))) for q in qs]
        for kind in (0,1):
            wts=[(modes(v) if kind==0 else canonical(v))[1]/k for v,k,o in es]
            direct=mp.mpf(0);test=mp.mpf(0)
            for i,(v,k,o) in enumerate(es):
                logA=mp.log(abs(-H(k3-k,k3*v3-k*v)/H(k3+k,k3*v3+k*v)))
                close(logA,eps*eta*o*(logs[(i+1)%3]-logs[i]))
                direct+=eps*eta*o*logA*wts[i]
                test+=eps*eta*o*logA
            sbp=sum((wts[(i-1)%3]-wts[i])*logs[i] for i in range(3))
            close(direct,sbp);close(test)
            assert abs(direct)>mp.mpf('.001')
            if eps==eta==1: print('(c) absorbed momentum remainder',kind,mp.nstr(direct,16))
# Alleged poles of d_q delta_D are removable on the scattering-root locus.
pA,pB,Q=sy.symbols('pA pB Q')
ell=lambda p:3*R*(4*p**3-p)
quot=sy.cancel((ell(pA)-ell(Q))/(pA-Q))
zero(quot-3*R*(4*(pA*pA+pA*Q+Q*Q)-1))
assert sy.limit((pA-Q)*quot,Q,pA)==0
print('(c) pole-based exclusion of root shifts REFUTED: numerator cancels the pole')
print('ALL CHECKS PASSED')

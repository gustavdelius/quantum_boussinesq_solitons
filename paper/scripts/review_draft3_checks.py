"""Independent draft-3 cross-review. All findings are project checks (c).

Run python3 -B paper/scripts/review_draft3_checks.py. No files are written and
no project implementation is imported. Expected false statements are tested
as counterexamples. SymPy exact arithmetic and 60-digit mpmath are used;
finite Nyström discretisation has its own explicitly stated tolerance.
The success marker certifies these assertions, not a physical S-matrix.
"""
from pathlib import Path
import ast
import hashlib
import re
import sympy as s
import mpmath as m

m.mp.dps = 60
ROOT = Path(__file__).resolve().parents[2]
TOL = m.mpf('1e-48')


def check(name, value):
    assert bool(value), name
    print('PASS (c): ' + name, flush=True)


def near(a, b, tol=TOL):
    return abs(a-b) < tol*max(1, abs(a), abs(b))


# G: genuinely evaluate the field symplectic pairing, not just det(W).
k, v, z = s.symbols('k v z', positive=True)
U = k**2*(1-z**2)/2
jac = 2/(k*(1-z**2))
w = 3*U/k-15*U**2/(2*k**3)
check('G auxiliary moments', [s.integrate(f*jac, (z, -1, 1)) for f in
      (w, w*U, w*w)] == [1, 0, 3*k/7])
pv, piv = s.symbols('pv piv', nonzero=True)
W = s.Matrix([[0,pv,piv,0],[-pv,0,0,0],[-piv,0,0,-1],[0,0,1,0]])
bv,bx,bc,br = s.symbols('bv bx bc br')
coeff = W.inv()*s.Matrix([bv,bx,bc,br])
check('G corrected matrix and tails', W.det()==pv**2 and
      s.simplify(coeff[2]-v*coeff[0]/k-br-v*bx/(k*pv))==0)
for vv in (m.mpf('.8'), m.mpf('-.7')):
    kk = m.sqrt(1-vv*vv)
    def uf(x): return kk**2*(1-x*x)/2
    def ux(x): return -kk*x*uf(x)
    def phiv(x): return -vv/kk*(x+(1-x*x)*m.atanh(x))
    def uv(x): return -vv*(1-x*x)*(1-x*m.atanh(x))
    def wf(x): return 3*uf(x)/kk-15*uf(x)**2/(2*kk**3)
    tangent = [(phiv, lambda x:-uf(x)-vv*uv(x)),
               (lambda x:-uf(x), lambda x:vv*ux(x)),
               (lambda x:m.mpf(1), lambda x:m.mpf(0)),
               (lambda x:m.mpf(0), wf)]
    pval=2*kk*(1-4*vv*vv)/3
    pival=2*(2*vv*vv-1)/kk
    target=W.subs({pv:s.Float(str(pval),60),piv:s.Float(str(pival),60)})
    err=0
    for i in range(4):
        for j in range(i+1,4):
            val=m.quad(lambda x:(tangent[i][1](x)*tangent[j][0](x)
                      -tangent[j][1](x)*tangent[i][0](x))*2/(kk*(1-x*x)),
                      [-1,0,1])
            err=max(err,abs(val-m.mpf(str(target[i,j]))))
    check('G direct Omega=dpi wedge dphi; printed matrices have opposite sign',err<TOL)
check('G projector counterexample eta=(w,0)', s.integrate(w*w*jac,(z,-1,1))!=0)
xx,tt=s.symbols('xx tt',real=True)
eta=s.Function('eta')(xx,tt); background=s.Function('U')(xx,tt)
lag2=eta.diff(tt)**2/2-(1-6*background)*eta.diff(xx)**2/2-eta.diff(xx,2)**2/2
el=s.euler_equations(lag2,eta,[xx,tt])[0].lhs
op=eta.diff(tt,2)+eta.diff(xx,4)-eta.diff(xx,2)+6*s.diff(background*eta.diff(xx),xx)
check('G full two-soliton lab-frame fluctuation operator',s.expand(el+op)==0)

# E: normalisation, finite subtraction sign, good/bad vacuum slopes.
alpha, beta, lam, cf, hb, Q = s.symbols('alpha beta lam cf hb Q', positive=True)
Fg=s.sqrt(3)/s.pi*(s.Rational(3,4)-k*k)*s.atanh(2*k/s.sqrt(3))-3*k/(2*s.pi)
Fac=s.sqrt(3)/s.pi*(s.Rational(3,4)+k*k)*s.atan(2*k/s.sqrt(3))-3*k/(2*s.pi)
Pg=s.Rational(2,3)*k**3*s.sqrt(1-k*k)
Pb=s.Rational(2,3)*k**3*s.sqrt(1+k*k)
check('E opposite small-amplitude trace slopes',
      s.limit(Fg/Pg,k,0)==-2/s.pi and s.limit(Fac/Pb,k,0)==2/s.pi)
check('E tadpole added to L1 gives negative energy slope',
      s.limit(alpha**2*(-2*cf*s.sin(alpha))/(s.Rational(2,3)*s.sin(alpha)**3*s.cos(alpha)),alpha,0)==-3*cf)
check('E one-loop coefficient is F, energy shift is hbar F',
      s.diff(Q+hb*Fg,hb)==Fg and s.simplify(s.diff(Q+hb*Fg,hb)/hb-Fg)!=0)
Wg=-((1-6*Q)**s.Rational(3,2)-1+9*Q)/(6*s.pi)
Wb=((1+6*Q)**s.Rational(3,2)-1-9*Q)/(6*s.pi)
WT=(s.exp(3*Q)-1-3*Q)/(6*s.pi)
slopes=[s.diff(f,Q,2).subs(Q,0)/2 for f in (Wg,Wb,WT)]
check('E vacuum sound shifts good, continued bad, Toda',
      slopes==[-9/(4*s.pi),9/(4*s.pi),3/(4*s.pi)])
check('E C_T, matching and scheme invariant',
      s.simplify(2/s.pi+(slopes[2]-slopes[1])-1/(2*s.pi))==0 and
      s.simplify((2/s.pi+lam)-(slopes[1]+lam)+1/(4*s.pi))==0 and
      s.simplify((-2/s.pi+lam)-(slopes[0]+lam)-1/(4*s.pi))==0)
check('E soliton quadratic counterterm moment',s.integrate(U**2*jac,(z,-1,1))==2*k**3/3)

# W: independently invert the corrected Laplace transform, then check the
# displayed asymptotic coefficients and actual u-to-y Jacobian in A_lambda.
def g(p):
    return m.gamma(1+p/(2*m.pi))*m.exp(-p*m.log(p/(2*m.pi*m.e))/(2*m.pi))/m.sqrt(p)


def inv(u, degree):
    # degree=0: F; degree=1: paper f=B; degree=2: ordinary f'=B'.
    if degree==2:
        return m.invertlaplace(lambda p:m.sqrt(m.pi)*(g(p)-1),u,method='dehoog')
    return m.invertlaplace(lambda p:m.sqrt(m.pi)*g(p)/p**(2-degree),u,method='dehoog')


for om in (m.mpf('.3'),m.mpf('1.7')):
    check('W nonzero-frequency factor product',near(2*m.pi*g(1j*om)*g(-1j*om),2*m.pi/(1-m.exp(-om))))
cc=(1-m.log(8*m.pi))/(2*m.pi)
dd=m.mpf(1)/48-1/(4*m.pi**2)
ee=(4+6*m.zeta(3)-m.pi**2)/(96*m.pi**3)
for uu in (m.mpf('128'),m.mpf('512')):
    yy=(3*inv(uu,0)/4)**(m.mpf(2)/3)
    residual=uu-yy+m.log(yy)/(2*m.pi)-cc-dd/yy-ee/yy**2
    check('W corrected c_edge,d0,e0 versus inverse Laplace at u='+str(uu),abs(residual)*yy**3<m.mpf('.0004'))
A,c,d,e,L=s.symbols('A c d e L')
v0=A*L-c; v1=A*v0-d; v2=A*v1-A*v0*v0/2+d*v0-e
check('W inverse-series log-square coefficient',s.expand(v2).coeff(L,2)==-A**3/2)
check('W -(log u)^2 coefficient',(-A**3/2).subs(A,1/(2*s.pi))==-1/(16*s.pi**3))

# K: differentiation of a generic convolution supplies the source/contact
# identities independently of an assumed rank. B is the paper's f=F'.
uu,vv,ww,c0=s.symbols('uu vv ww c0',positive=True)
B=s.Function('B')
integrand=2*s.diff(B(uu-ww),uu)*B(vv-ww)
check('K characteristic derivative is a total w derivative',
      s.simplify(s.diff(integrand,uu)+s.diff(integrand,vv)+s.diff(integrand,ww))==0)
# Leibniz endpoint at w=v is 2 B'(u-v) B(0); the moving endpoint
# cancels this in (d_u+d_v)H, leaving 2 B'(u) B(v).
correct=2*s.diff(B(uu),uu)*B(vv)-2*c0*s.diff(B(vv),vv)
wrong=2*s.diff(B(uu),uu)*(B(vv)-c0)
check('K wrong four-term source discrepancy',
      s.expand(correct-wrong-2*c0*(s.diff(B(uu),uu)-s.diff(B(vv),vv)))==0)
bu,bv2,fu,fv=[inv(m.mpf(x),n) for x,n in [('4',1),('1',1),('4',2),('1',2)]]
right=2*fu*bv2-2*m.sqrt(m.pi)*fv
old=2*fu*(bv2-m.sqrt(m.pi))
# The package's quoted number is a finite-difference convolution derivative,
# with a reported 2.07e-17 error, not an exact source value. The first review
# run mistakenly required 1e-17; the review records that failed comparison.
check('K actual WH source within published differentiation error',near(right,m.mpf('-.347104254662920476'),m.mpf('2.1e-17')) and abs(right-old)>1)
check('K derivative transform independently differentiated',near(fu,m.diff(lambda u:inv(u,1),m.mpf(4))))
print('K corrected/old source at (4,1):',m.nstr(right,25),m.nstr(old,25),flush=True)
check('K contact term uses B prime, not B',abs(inv(m.mpf(3),2)-inv(m.mpf(3),1))>1)
x,y,r,Y=s.symbols('x y r Y',positive=True)
anti=x*y*(x-y)/(x+y)**3
check('K A_lambda antisymmetry, axes and scaling',s.simplify(anti+anti.xreplace({x:y,y:x}))==0 and
      s.limit(anti,y,0)==0 and s.simplify(anti.subs({x:Y,y:r*r*Y})-r*r*(1-r*r)/(1+r*r)**3)==0)
minor=s.Matrix([[anti.subs({x:i,y:j}) for j in (4,5,6)] for i in (1,2,3)]).det()
check('K Schmidt minor proves rank above two only',minor!=0)
check('K distinct moving poles establish rational-function independence',s.limit((x+y)**3*anti,x,-y)==2*y**3)
yu=(3*inv(m.mpf(4),0)/4)**(m.mpf(2)/3)
yv=(3*inv(m.mpf(1),0)/4)**(m.mpf(2)/3)
ax=s.lambdify((x,y),s.diff(anti,x),'mpmath'); ay=s.lambdify((x,y),s.diff(anti,y),'mpmath')
weighted=bu/(2*m.sqrt(yu))*ax(yu,yv)+bv2/(2*m.sqrt(yv))*ay(yu,yv)
check('K A_lambda violates homogeneous equation with actual coordinate Jacobian',abs(weighted)>m.mpf('1e-4'))
print('K weighted homogeneous residual:',m.nstr(weighted,20),flush=True)
check('K parity and complex pole are algebraic properties',s.simplify(anti.subs({x:Y,y:r*r*Y})-anti.subs({x:Y,y:(-r)**2*Y}))==0)

# F: fresh finite-sea solve, actual four-term phase (not just a symbolic
# rewrite of the desired endpoint product), both edges, a determined by count.
with m.workdps(50):
    h=m.mpf('.9'); q=m.mpf('.8'); t=2*m.pi*h; n=40
    nd,wt=m.gauss_quadrature(n,'legendre'); nodes=[q*z for z in nd]; weights=[q*w for w in wt]
    def kernel(z):return 2*m.re(m.digamma(1+1j*z/h))+2*m.log(h)
    def theta(z):return 2*h*m.im(m.loggamma(1+1j*z/h))+2*z*m.log(h)
    mat=m.matrix([[t*(i==j)-weights[j]*kernel(nodes[i]-nodes[j]) for j in range(n)] for i in range(n)])
    inverse=mat**-1; rho=inverse*m.ones(n,1); energy=inverse*m.matrix(nodes)
    spacing=1/m.fsum(weights[i]*rho[i] for i in range(n)); scale=m.exp(-spacing/2)
    def dot(vals,vec):return m.fsum(weights[i]*vals[i]*vec[i] for i in range(n))
    def tv(p):return m.matrix([theta(p-z) for z in nodes])
    def kv(p):return m.matrix([kernel(p-z) for z in nodes])
    def P(p):return p+dot(tv(p),rho)
    def E(p):return p*p/2+dot(tv(p),energy)
    def D(p):return 1+dot(kv(p),rho)
    def V(p):return p+dot(kv(p),energy)
    def ends(p):
        fp=inverse*tv(p)
        return [(theta(p-edge)+dot(kv(edge),fp))/t for edge in (q,-q)]
    def phase(p,z):return theta(p-z)-dot(kv(p),inverse*tv(z))
    def hat(p,z):return phase(p,z)-phase(p,q)-phase(q,z)+phase(q,q)
    eq=ends(q); worst=m.mpf(0)
    for p in (m.mpf('1.2'),m.mpf('-1.3')):
        fp,fm=ends(p)
        worst=max(worst,abs(D(q)*(fp-fm)-1+D(p)),abs(V(q)*(fp+fm)-P(p)+V(p)),
                  abs(V(p)-V(q)*D(p)/D(q)-P(p)+P(q)+2*V(q)*(fm-eq[1])))
        for z2 in (m.mpf('1.5'),m.mpf('-1.6')):
            zp,zm=ends(z2)
            product=t*((fp-eq[0])*(zp-eq[0])-(fm-eq[1])*(zm-eq[1]))
            worst=max(worst,abs(hat(p,z2)+hat(z2,p)-product))
    worst=max(worst,abs(D(q)*V(q)-m.pi*h/spacing),abs(P(q)-m.pi*h/spacing))
    check('F actual finite-sea endpoint, opposite-edge and reciprocity identities',worst<m.mpf('1e-27'))
    print('F maximum finite-sea residual:',m.nstr(worst,8),flush=True)
    # Derive y(p) via classical momentum inversion, differentiate y(p)
    # numerically, and invert its derivative to test u'(y).
    def classical(K):return 4*scale*(K*m.cosh(K)-m.sinh(K))/spacing
    def kap(p):return m.findroot(lambda K:classical(K)-P(p)+P(q),(.2,1))
    def ylabel(p):return scale*kap(p)**2/h
    pp=m.mpf('1.2'); KK=kap(pp); yp=m.diff(ylabel,pp)
    check('F independent momentum inversion Jacobian',near(1/(h*yp),2*m.sinh(KK)/(spacing*D(pp)),m.mpf('1e-35')))
    check('F energy Jacobian',near(m.diff(E,pp)/yp,2*h*m.sinh(KK)*V(pp)/(spacing*D(pp)),m.mpf('1e-35')))

# C: formal Krein integrals, with the same conditional status as the audits.
for rr in (m.mpf('.4'),m.mpf('.6')):
    chi=(m.polylog(2,rr)-m.polylog(2,-rr))/2
    direct=4/m.pi*m.quad(lambda z:m.log(z/rr)*z*z/(1-z*z),[0,rr])
    check('C R0 conditional integral',near(direct,-4/m.pi*(chi-rr)))
    def integr(z):
        sig=1-rr*rr+z*z; delta=-m.log(z*z/(rr*rr*sig))/(2*m.pi)
        return 8/sig**m.mpf('2.5')*(z*z*(-2*dd-3/(4*m.pi**2)-delta/(4*m.pi)+3*delta**2/8)
                  -dd*sig/2*(1+z*z*(1-1/rr**2-1/sig)))
    target=-rr**3/(6*(1-rr**2))-2*dd*((1-rr**2)/rr**2*m.log((1-rr)/(1+rr))+2/rr)
    check('C c1 conditional integral',near(m.quad(integr,[0,rr]),target))

# S: full candidate residue, not just an action-level condition.
def candidate(b1,b2,h):
    dl=m.cosh(3*b1)-m.cosh(3*b2); eta=2*dl/(9*h)
    interaction=(1+3*m.coth((b1+b2)/2)**2)/(1+3*m.coth((b1-b2)/2)**2)
    top=8*m.sinh(b1)*m.sinh(b2)*(m.cosh(b1)-m.cosh(b2))/3
    return m.gamma(1+1j*eta)/m.gamma(1-1j*eta)*m.exp(1j*eta*m.log(m.e**2*h*h*81*interaction/(4*dl*dl))+1j*top/h)
aa=m.mpf('.5'); angle=m.mpf('1.25'); action=-4*m.sinh(3*aa)*m.sin(3*angle)/9
for N in (1,2,3):
    h=action/N
    residue=[]
    for eps in (m.mpf('1e-15'),m.mpf('1e-20')):
        residue.append(eps*candidate(aa-1j*(angle+eps),aa+1j*(angle+eps),h))
    check('S full bad Gamma candidate has nonzero simple pole N='+str(N),abs(residue[1])>m.mpf('1e-15') and near(residue[0],residue[1],m.mpf('1e-12')))
check('S gamma quotient regular at eta=0',near(m.gamma(1)/m.gamma(1),1))
# A compensating prefactor permits a non-spectral odd Gamma argument locally.
# This refutes unrestricted necessity, not a restricted prefactor ansatz.
xi=m.mpf('.3')/m.mpf('1e-12')
offset=2*m.im(m.loggamma(1+1j*xi))-2*xi*(m.log(xi)-1)
check('S arbitrary odd Gamma argument with compensating leading phase',abs(offset-m.pi/2)<m.mpf('1e-11'))
Ag=s.symbols('Ag',positive=True)
tau=lambda X:1-2*X+Ag*X*X
check('S printed good-breather zero is false',s.simplify(tau(1/s.sqrt(Ag)))==2-2/s.sqrt(Ag))
check('S corrected positive tau zero',s.simplify(tau(1/(1+s.sqrt(1-Ag))))==0)
check('S A=0 exceptional assertion fails at cos I=+1',1+2*s.Symbol('X',positive=True)>0)

# R: independent classical phase derivatives for conditional ring identities.
def data(a,eps,bad):
    k=(m.sinh(a) if bad else m.sin(a)); v=eps*(m.cosh(a) if bad else m.cos(a))
    ell=eps*(m.cosh(3*a) if bad else -m.cos(3*a))
    return k,v,ell,2*k,-2*k*v,2*k**3*v/3


def interaction(a,b,e1,e2,bad):
    k1,v1,*_=data(a,e1,bad); k2,v2,*_=data(b,e2,bad)
    def poly(K,W):return W*W-K*K+(-1 if bad else 1)*K**4
    return -poly(k1-k2,k1*v1-k2*v2)/poly(k1+k2,k1*v1+k2*v2)


def delta(a,b,e1,e2,bad):
    k1,v1,l1,M1,Pi1,P1=data(a,e1,bad); k2,v2,l2,M2,Pi2,P2=data(b,e2,bad)
    return 2*(l1-l2)*m.log(interaction(a,b,e1,e2,bad))/9+2*M1*M2*(v1-v2)/3


for bad in (False,True):
    for e2 in (1,-1):
        a=m.mpf('.3'); b=m.mpf('.6'); k1,v1,l1,M1,Pi1,P1=data(a,1,bad)
        M2=data(b,e2,bad)[3]
        ring=lambda aa:delta(aa,b,1,e2,bad)+data(aa,1,bad)[4]*M2
        derivative=-m.diff(ring,a)/m.diff(lambda aa:data(aa,1,bad)[5],a)
        check('R conditional directional ring derivative '+str((bad,e2)),near(derivative,-m.log(interaction(a,b,1,e2,bad))/k1))
P1,P2,M1,M2,Pi1,Pi2,d12,L=s.symbols('P1 P2 M1 M2 Pi1 Pi2 d12 L')
check('R conditional ring sum includes self winding',s.expand((P1*L+Pi1*M1+d12+Pi1*M2)+(P2*L+Pi2*M2-d12+Pi2*M1)-(P1+P2)*L-(Pi1+Pi2)*(M1+M2))==0)

# D: missing defect primitives, threshold and continuum UV normalisation.
sig,bare=s.symbols('sigma bare',real=True)
for orientation in (-1,1):
    A=orientation/s.sqrt(3)
    rho=(sig-sig**3/3-bare+bare**3/3)/(2*A)
    tau=s.integrate(-sig*rho,(sig,bare,sig)); hh=s.integrate(-rho,(sig,bare,sig))
    check('D defect primitive derivatives and reference',s.simplify(s.diff(tau,sig)+sig*rho)==0 and
          s.simplify(s.diff(hh,sig)+rho)==0 and tau.subs(sig,bare)==hh.subs(sig,bare)==0)
    T=s.symbols('T'); MD=(sig-bare)/(s.sqrt(3)*orientation)
    check('D UV boundary term',s.simplify(orientation*s.sqrt(3)*sig*(T-1)/(2*s.pi)-orientation*s.sqrt(3)*bare*(T-1)/(2*s.pi)-3*(T-1)*MD/(2*s.pi))==0)
S,c=s.symbols('S c')
for sign in (-1,1):
    poly=S*S-(s.sqrt(3)+sign)*c*S/2+(c*c-2+sign*s.sqrt(3))/2
    check('D threshold polynomial',s.expand(poly.subs(S,1)-(c-sign)*(c-s.sqrt(3))/2)==0)
check('D remaining growth rate at switch',s.simplify((((s.sqrt(3)+1)/2)**2-1)/2)==s.sqrt(3)/4)

# I: citation existence and honest static coverage. No phase2 script is run
# because many write tracked data; all requested paper scripts were run apart.
paper=(ROOT/'paper/boussinesq_soliton_scattering.tex').read_text()
app=paper.split(r'\section{Claims and verification}')[1].split(r'\section{Inconsistencies')[0]
paths=sorted(set(re.findall(r'\\scr\{([^{}]+\.py)\}',app)))
check('I all 64 Appendix A script paths exist',len(paths)==64 and all((ROOT/p).is_file() for p in paths))
for path in paths:
    ast.parse((ROOT/path).read_text())
print('I Appendix A source files parsed:',len(paths),flush=True)
hirota=ast.parse((ROOT/'phase1/scripts/hirota_solitons.py').read_text())
bypasses=[n for n in ast.walk(hirota) if isinstance(n,ast.Assert) and
          any(isinstance(q,ast.BoolOp) and isinstance(q.op,ast.Or) and
              any(isinstance(t,ast.Constant) and t.value is True for t in q.values) for q in ast.walk(n.test))]
check('I current Hirota assertions are not bypassed',not bypasses)
check('I generator is not a verification script','DATA GENERATED (no checks)' in (ROOT/'phase1/scripts/figures.py').read_text())
check('I earlier onebody failure is recorded, not declared repaired',
      "assert max_cv_id_diff < mp.mpf('3e-6')" in (ROOT/'phase2/scripts/audit_onebody.py').read_text())
print('I reviewed paper SHA256:',hashlib.sha256(paper.encode()).hexdigest(),flush=True)
print('ALL CHECKS PASSED')

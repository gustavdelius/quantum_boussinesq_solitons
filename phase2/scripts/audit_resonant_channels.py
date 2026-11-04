#!/usr/bin/env python3
"""Independent audit (c). No project imports, cache reads or file writes.
Riccati/WKB recursion and Darboux intertwining replace pseudodifferential
multiplication; exponential-coordinate beta integrals replace tanh integration.
All quantum results are conditional on the stated continuum prescription.
"""
import time
from functools import lru_cache
import sympy as s
import mpmath as m
START=time.perf_counter()
m.mp.dps=80
TOL=m.mpf('1e-60')
def close(x,y=0,tol=TOL):
    assert abs(x-y)<tol, (m.nstr(x,30),m.nstr(y,30),m.nstr(abs(x-y),8))
def say(*args): print(*args,flush=True)
def Dprint(x): return m.nstr(x,30)

# H_n = -n int [z^-n](psi'/psi - lambda) dx, modulo total derivatives.
# f''+3ff'+f^3+q f+r=z^3, lambda^3-lambda/4=z^3.
y=s.Symbol('y',positive=True); k,v=s.symbols('k v',real=True)
w=s.Symbol('w'); pa,pb=s.symbols('a b'); p=s.Symbol('p')
C=pa**2+pa*pb+pb**2-s.Rational(1,4)
red=lambda e:s.factor(s.rem(s.expand(e),C,pa))
delta=lambda j:(-pb)**j-(-pa)**j
lam=1/w
for j in range(1,9):
    c=s.Symbol('c'); trial=lam+c*w**j
    coef=s.expand(trial**3-trial/4-w**-3).coeff(w,j-2)
    lam+=s.solve(coef,c)[0]*w**j
assert s.series(lam**3-lam/4-w**-3,w,0,7).removeO()==0
H={n:s.expand(n*sum(delta(j)/j*s.series((1/lam)**j,w,0,9).removeO().coeff(w,n)
                    for j in range(1,n+1))) for n in range(1,9)}
expected={1:delta(1),2:delta(2),3:0,4:delta(4)-delta(2)/3,
5:delta(5)-s.Rational(5,72)*delta(1),6:0,
7:delta(7)-s.Rational(7,12)*delta(5)+s.Rational(7,324)*delta(1),
8:delta(8)-s.Rational(7,36)*delta(4)+s.Rational(11,324)*delta(2)}
for n in H: assert red(H[n]-expected[n])==0
# Independent Riccati recursion directly on rational exponential profile.
def riccati(q,r,dx,N):
    f={-1:s.Integer(1)}
    for n in range(1,N+1):
        target=n-2
        rhs=dx(dx(f.get(target,0)))+q*f.get(target,0)+(r if target==0 else 0)
        rhs+=3*sum(fi*dx(f.get(target-i,0)) for i,fi in f.items())
        rhs+=sum(fi*fj*f.get(target-i-j,0) for i,fi in f.items() for j,fj in f.items())
        f[n]=s.cancel(-rhs/3)
        say('Riccati coefficient',n,'constructed')
    return f
say('Formal generating series checked through H8')
for kval,vval in [(s.Rational(1,2),s.sqrt(3)/2),(s.Rational(97,103),-20*s.sqrt(3)/103)]:
    dx=lambda a:s.cancel(kval*y*s.diff(a,y))
    u=2*kval**2*y/(1+y)**2
    q=3*u/2-s.Rational(1,4);r=dx(q)/2-vval/(2*s.sqrt(3))*(q+s.Rational(1,4))
    fs=riccati(q,r,dx,8)
    for n in range(1,9):
        integrand=s.cancel(-n*(fs[n]-fs[n].subs(y,0))/(kval*y))
        num,den=s.fraction(integrand)
        if num==0: value=0
        else:
            N=s.degree(den,y); scale=den.subs(y,0)
            assert s.expand(den-scale*(1+y)**N)==0
            value=sum(coef*s.factorial(j)*s.factorial(N-j-2)/s.factorial(N-1)/scale
                      for (j,),coef in s.Poly(num,y).terms())
        a=(vval/s.sqrt(3)+kval)/2;b=(vval/s.sqrt(3)-kval)/2
        assert s.simplify(value-H[n].subs({pa:a,pb:b}))==0,(n,value)
    say('Riccati beta integration H1..H8:',kval,vval,'passed')
# Darboux operator coefficient identities, also applicable to time-dependent Y.
f0,f1,f2,f3,L0=s.symbols('f0 f1 f2 f3 L0')
q0=-s.Rational(1,4);qd=q0+3*f1;rd=3*f0*f1+3*f2
assert s.expand(-f3-qd*f1-rd*f0+(f3+3*f1**2+3*f0*f2+3*f0**2*f1+q0*f1))==0
# Physical chord potentials, independent beta integrals give these charges.
Qphys=[2*k,-2*k*v,s.Rational(2,3)*k**3*v,s.Rational(2,15)*k**3*(5-4*k**2)]
F=[-2*p,2*s.sqrt(3)*p**2,-2*s.sqrt(3)*p**4+p**2/s.sqrt(3),-s.Rational(24,5)*p**5+p/6]
for Q,FQ in zip(Qphys,F): assert red(Q.subs({k:pa-pb,v:s.sqrt(3)*(pa+pb)})-(FQ.subs(p,pb)-FQ.subs(p,pa)))==0
c0,c1=s.symbols('c0 c1')
assert s.solve(s.Poly(red(c0*delta(1)+c1*delta(5)-(pa-pb)**3),pa,pb).coeffs(),[c0,c1])==[]
say('Darboux generating function, printed H7/H8 and physical potentials passed')

# Kinematics using signed velocity as independent coordinate, no alpha inversion.
def sol(V):
    K=m.sqrt(1-V*V)
    return K,V,2*K,-2*K*V,m.mpf(2)/3*K**3*V,m.mpf(2)/15*K**3*(5-4*K*K)
def pair(phi):
    roots=[m.cos(phi+shift)/m.sqrt(3) for shift in (0,-2*m.pi/3,2*m.pi/3)]
    return roots,[sol(m.sqrt(3)*(roots[i]+roots[j])) for i,j in ((0,1),(1,2),(0,2))]
def A(S,T):
    k,v=S[:2];l,w=T[:2]
    disp=lambda K,W:W*W-K*K+K**4
    return -disp(k-l,k*v-l*w)/disp(k+l,k*v+l*w)
def ell(S): return S[1]*(4*S[0]**2-1)
def topo(S,T): return m.mpf(8)/3*S[0]*T[0]*(S[1]-T[1])
def phase(S,T):return m.mpf(2)/9*(ell(S)-ell(T))*m.log(A(S,T))+topo(S,T)
for phi in [m.mpf('0.03'),m.mpf('0.23'),m.mpf('0.49'),m.mpf('0.69'),m.mpf('1.01')]:
    roots,ss=pair(phi)
    for j in range(2,6): close(ss[0][j]+ss[1][j],ss[2][j])
    close(ss[0][1]-ss[1][1],m.sqrt(3)*ss[2][0])
    for i,j in ((0,1),(1,2),(0,2)):
        a,b=roots[i],roots[j];close(3*(a*a-b*b)**2-(a-b)**2+(a-b)**4)
    assert ss[0][1]>.5 and ss[1][1]<-.5 and abs(ss[2][1])<.5
# Independent symbolic curvature in velocity coordinate.
V=s.Symbol('V',positive=True); K=s.sqrt(1-V**2)
P=s.Rational(2,3)*K**3*V; E=s.Rational(2,15)*K**3*(5-4*K*K)
assert s.simplify(s.diff(E,V)-V*s.diff(P,V))==0
assert s.simplify(1/s.diff(P,V)+3/(2*K*(4*V**2-1)))==0
# Inverse of monotonically increasing light P as a function k, bracketed solve.
def light(P):
    if abs(P)==0:return m.mpf(0)
    l,h=m.mpf(0),m.sqrt(3)/2
    for _ in range(280):
        z=(l+h)/2
        if m.mpf(2)/3*z**3*m.sqrt(1-z*z)<abs(P):l=z
        else:h=z
    z=(l+h)/2
    return m.mpf(2)/15*z**3*(5-4*z*z)
for v1 in map(m.mpf,['.52','.7','.95']):
 for v2 in map(m.mpf,['.6','.8','.98']):
  for sign in (-1,1):
   S,T=sol(v1),sol(sign*v2);Pt=S[4]+T[4]
   if abs(Pt)<=m.sqrt(3)/8: assert S[5]+T[5]>light(Pt)
# Printed slope has the wrong sign; independent finite differences use v inversion.
a=m.mpf('.62');b=m.pi/3-a;S=sol(m.cos(a));T=sol(-m.cos(b));B=sol(m.cos(2*m.pi/3-a))
def heavy_energy(P):
    vr=m.findroot(lambda vv:sol(vv)[4]-P,(-m.mpf('.49'),m.mpf('.49')))
    assert abs(vr)<.5
    return sol(vr)[5]
def gap(h):
    T=sol(-m.cos(b-h));return S[5]+T[5]-heavy_energy(S[4]+T[4])
slope=m.diff(gap,0)
assert slope<0
close(slope,(T[1]-B[1])*m.mpf(2)/3*m.sin(b)*m.sin(3*b))
assert abs(slope+m.mpf('0.24107867527655644'))>m.mpf('.02')
for h in [m.mpf('1e-8'),m.mpf('1e-12')]:
    assert gap(h)<0 and gap(-h)>0
for av,bv in [(m.mpf('.2'),m.mpf('.3')),(m.mpf('.4'),m.mpf('.8')),(m.mpf('.7'),m.mpf('.6'))]:
    Sg,Tg=sol(m.cos(av)),sol(-m.cos(bv))
    dg=Sg[5]+Tg[5]-heavy_energy(Sg[4]+Tg[4])
    assert m.sign(dg)==m.sign(av+bv-m.pi/3)
say('Light concavity; all-heavy on-shell charges/Jacobian; signed slope',Dprint(slope))

# Heavy instability independently exposed by the exact Y family.
roots,ss=pair(m.mpf('.23'));kh,vh=ss[2][:2]
growth=m.sqrt(3)*ss[0][0]*ss[1][0]
close(growth,m.sqrt(3)*(m.mpf(1)/4-vh*vh))
profile=lambda x:kh*kh/(2*m.cosh(kh*x/2)**2)
seed=lambda x:m.exp(-m.sqrt(3)*vh*x/2)/(2*m.cosh(kh*x/2))
mode=lambda x:2*m.diff(seed,x,2)
for xx in map(m.mpf,['-2.1','.3','1.8']):
    residual=growth**2*mode(xx)-2*growth*vh*m.diff(mode,xx)+vh*vh*m.diff(mode,xx,2)-m.diff(mode,xx,2)+m.diff(mode,xx,4)+6*m.diff(lambda x:profile(x)*mode(x),xx,2)
    close(residual,0,m.mpf('1e-65'))
assert kh>m.sqrt(3)*abs(vh) and growth>0
say('Heavy localized growing mode rate',Dprint(growth))
# The printed unrestricted phase-space integral can have extra co-moving roots.
Hheavy=sol(m.cos(m.mpf('1.1')))
def equations(k1,k2):
    v1=m.sqrt(1-k1*k1);v2=m.sqrt(1-k2*k2)
    return (sol(v1)[4]+sol(v2)[4]-Hheavy[4],sol(v1)[5]+sol(v2)[5]-Hheavy[5])
k1,k2=m.findroot(equations,(m.mpf('.1'),m.mpf('.85')))
assert 0<k1<k2<m.sqrt(3)/2
assert abs(2*(k1+k2)-Hheavy[2])>m.mpf('.01')
for rr in equations(k1,k2):close(rr)
say('Additional E,P-only co-moving root widths',Dprint(k1),Dprint(k2),'mass mismatch',Dprint(2*(k1+k2)-Hheavy[2]))

# Direct Y integration, minus-sign tau, using probability central moments.
roots,ss=pair(m.mpf('.23'))
def densities(x,t):
    ex=[-q*x+m.sqrt(3)*q*q*t for q in roots]; mx=max(ex)
    weights=[m.exp(e-mx) for e in ex]; Z=sum(weights);weights=[e/Z for e in weights]
    mean=sum(w*q for w,q in zip(weights,roots))
    mu=[sum(w*(q-mean)**j for w,q in zip(weights,roots)) for j in (2,3)]
    u=2*mu[0];ux=-2*mu[1]
    pi=-2*m.sqrt(3)*(mu[1]+2*mean*mu[0])
    return [u,pi,-pi*u,(pi*pi+u*u+ux*ux)/2-u**3]
# Independent central-moment densities and different panels/times; precision comparison.
quad_results=[]
for dps in (45,65):
 with m.workdps(dps):
  rows=[]
  for t in (-5,1,7):
   vals=[]
   for j in range(4):
    # finite physical panels plus double-exponential tails, no tracked cache
    val=m.quad(lambda x:densities(x,t)[j],[-m.inf,-24,-6,3,18,m.inf],method='tanh-sinh')
    close(val,ss[2][j+2],m.mpf(10)**(-dps+8));vals.append(val)
   rows.append(vals)
  quad_results.append(rows)
spread=max(abs(a-b) for ra,rb in zip(*quad_results) for a,b in zip(ra,rb))
say('Y M,Pi,P,E quadrature precision/panel error:',Dprint(spread))
# All hierarchy across collision: direct numerical Riccati log-derivative integrals
# from dressed eigenfunction psi=(lambda-f) exp(lambda x). Integral is boundary log.
# Numerically integrate f'/(lambda-f) at non-asymptotic spectral values, all-order generator.
for t in (-5,1,7):
 for lamv in [m.mpf('1.3'),m.mpf('2.1')]:
    def g(x):
        ex=[-q*x+m.sqrt(3)*q*q*t for q in roots];mx=max(ex)
        ws=[m.exp(e-mx) for e in ex];Z=sum(ws)
        mean=-sum(w*q for w,q in zip(ws,roots))/Z
        fp=sum(w*q*q for w,q in zip(ws,roots))/Z-mean**2
        return -fp/(lamv-mean)
    val=m.quad(g,[-m.inf,-24,-6,3,18,m.inf])
    close(val,m.log((lamv+roots[2])/(lamv+roots[0])),m.mpf('1e-65'))
say('All-order Darboux generator integrated through Y collision')

# Independent mode trace from rational KP transmission, not angular log formulas.
def trans(q,S):
    k,v=S[:2];pa=(v/m.sqrt(3)+k)/2;pb=(v/m.sqrt(3)-k)/2
    R=m.sqrt(1+q*q);sg=m.sign(q)
    qa=(sg*R+1j*m.sqrt(3)*q)/(2*m.sqrt(3));qb=qa.conjugate()
    da=(sg*q/R+1j*m.sqrt(3))/(2*m.sqrt(3));db=da.conjugate()
    amp=(pa-qa)*(pb-qb)/((pa-qb)*(pb-qa))
    deriv=m.im(-da/(pa-qa)-db/(pb-qb)+db/(pa-qb)+da/(pb-qa))
    return amp,deriv

def trace(S,U,cuts=None,method="tanh-sinh"):
    L=m.sinh(U);R=m.cosh(U);k,v=S[:2]
    def integrand(t,momentum=False):
        q=m.sinh(t);R=m.cosh(t)
        dp=trans(q,S)[1];dn=trans(-q,S)[1]
        return (q*(dp-dn) if momentum else q*R*(dp+dn))*R/(4*m.pi)
    cuts=cuts or [0,1,3,6,U]
    eraw=m.quad(lambda t:integrand(t),cuts,method=method)
    # Only large-q endpoint phases needed; principal arg tends zero there.
    boundary=L*R*(m.arg(trans(L,S)[0])-m.arg(trans(-L,S)[0]))/(4*m.pi)
    energy=eraw-boundary+3*k/m.pi*(R-1)
    momentum=m.quad(lambda t:integrand(t,True),cuts,method=method)
    return energy,momentum,energy-v*momentum

loop_rows=[]
for U in (18,24,30):
    vals=[trace(S,m.mpf(U)) for S in ss]
    # Momentum has a slower finite-cutoff remainder as well; fit unrestricted below.
    loop_rows.append(vals)
    say('Root-transmission trace cutoff U',U,'delta F',Dprint(vals[0][2]+vals[1][2]-vals[2][2]))
with m.workdps(60):
    alternate=[trace(S,m.mpf(18),cuts=[0,m.mpf('.5'),2,5,10,18],method='gauss-legendre') for S in ss]
quadspread=max(abs(a-b) for aa,bb in zip(alternate,loop_rows[0]) for a,b in zip(aa,bb))
assert quadspread<m.mpf('1e-40')
say('Trace quadrature method/precision/panel spread',Dprint(quadspread))
xs=[1/m.sinh(U) for U in (18,24,30)]
def extrap(ys,inds):
    A=m.matrix([[1,xs[i],xs[i]**3][:len(inds)] for i in inds]);b=m.matrix([ys[i] for i in inds])
    return m.lu_solve(A,b)[0]
loop=[]
for j in range(3):
    loop.append([extrap([row[j][c] for row in loop_rows],[0,1,2]) for c in range(3)])
dF=loop[0][2]+loop[1][2]-loop[2][2]
close(dF,m.mpf('.38879244070035460274'),m.mpf('5e-21'))
close(loop[0][0]+loop[1][0],loop[2][0],m.mpf('1e-32'))
close(loop[0][1]+loop[1][1],loop[2][1],m.mpf('1e-32'))
ys=[row[0][2]+row[1][2]-row[2][2] for row in loop_rows]
say('Independent delta F',Dprint(dF),'two/three-cutoff spread',Dprint(abs(extrap(ys,[1,2])-dF)))
# At fixed incoming heavy P: outgoing + momentum moves down; no endpoint threshold.
dp=-dF/(ss[0][1]-ss[1][1]);assert dp<0
# Under finite u^2 coefficient lambda: dF -> dF - 2 lambda k12 k23 k13.
scheme_slope=-2*ss[0][0]*ss[1][0]*ss[2][0]
close(m.mpf(2)/3*(ss[0][0]**3+ss[1][0]**3-ss[2][0]**3),scheme_slope)
say('Fixed-heavy-P outgoing shift',Dprint(dp),'scheme slope',Dprint(scheme_slope))

# Resonances derived from bilinear polynomial; area computed by integrating slices.
for e1,e2 in [(1,1),(-1,-1),(1,-1),(-1,1)]:
 for a,b in [(m.mpf('.31'),m.mpf('.72')),(m.mpf('1.22'),m.mpf('.19'))]:
  S,T=sol(e1*m.cos(a)),sol(e2*m.cos(b)); aa=(S[1]/m.sqrt(3)+S[0])/2;ab=aa-S[0];ba=(T[1]/m.sqrt(3)+T[0])/2;bb=ba-T[0]
  cross=(aa-ba)*(ab-bb)/((aa-bb)*(ab-ba));close(A(S,T),cross)
  trig=(1-3*m.cot((a+b)/2)**2)/(1-3*m.cot((a-b)/2)**2) if e1==e2 else (1-3*m.tan((a-b)/2)**2)/(1-3*m.tan((a+b)/2)**2)
  close(A(S,T),trig)
close(m.quad(lambda x:max(0,x-m.pi/6),[0,m.pi/6,m.pi/2])/(m.pi*m.pi/4),m.mpf(2)/9)
ho_area=m.quad(lambda x:max(0,min(m.pi/2,x+m.pi/3)-max(0,m.pi/3-x,x-m.pi/3)),[0,m.pi/6,m.pi/3,m.pi/2])
close(ho_area/(m.pi*m.pi/4),m.mpf(2)/3)
# Transverse logarithmic coefficient by derivative (h * delta''), not two-point subtraction.
for a,case,sign in [(m.mpf('.75'),'ov',1),(m.mpf('.4'),'hp',-1),(m.mpf('.25'),'hm',1)]:
 def ST(h):
  if case=='ov':return sol(m.cos(a)),sol(m.cos(2*m.pi/3-a-h))
  if case=='hp':return sol(m.cos(a)),sol(-m.cos(m.pi/3-a-h))
  return sol(m.cos(a+m.pi/3+h)),sol(-m.cos(a))
 h=m.mpf('1e-25');S,T=ST(h)
 coeff=h*m.diff(lambda hh:phase(*ST(hh)),h,2)
 close(coeff,-m.mpf(2)/3*sign*m.sin(3*a),m.mpf('1e-21'))
 # generic coefficient of log A, computed from finite trace, with modest bound
 dp1=trace(S,m.mpf(30))[1];dp2=trace(T,m.mpf(30))[1]
 assert abs(dp1/S[0]-dp2/T[0])>m.mpf('.01')
 tab={'ov':('.898441539345','.074538040233','.392932242573'),
      'hp':('1.076226034357','.412963868475','1.271224247491'),
      'hm':('.787088643179','1.240490193422','.992253053483')}[case]
 close(topo(*ST(0)),m.mpf(tab[0]),m.mpf('6e-13'))
 close(A(S,T)/h**sign,m.mpf(tab[1]),m.mpf('6e-13'))
 close(-(dp1/S[0]-dp2/T[0]),m.mpf(tab[2]),m.mpf('6e-13'))
 say(case,'h log h coefficient',Dprint(coeff))
# negative ell is allowed for left-moving overtaking resonance.
assert ell(sol(-m.cos(m.mpf('.75'))))<0
# Abstract alpha13=alpha12+alpha23+pi/3 gives 2pi/3, outside stated range.
assert abs((m.pi/3-m.mpf('.23'))+m.mpf('.23')+m.pi/3-(m.pi/3+m.mpf('.23')))>.5

# WKB continuation: explicitly use both logarithm sheets and full phase.
S,T=sol(m.cos(m.mpf('.50'))),sol(-m.cos(m.mpf('.65')));Av=A(S,T);de=ell(S)-ell(T);hb=m.mpf('.1')
assert Av<0
mods=[]
for branch in (-1,1):
    dc=m.mpf(2)/9*de*(m.log(-Av)+branch*1j*m.pi)+topo(S,T)
    mods.append(abs(m.exp(1j*dc/hb))**2)
close(mods[0]*mods[1],1)
close(min(mods),m.exp(-4*m.pi*abs(de)/(9*hb)))
# Refute printed A, ell gap and WKB rounded example with discrepancies asserted.
assert abs(Av-m.mpf('-.354784'))>1
assert abs(abs(de)-m.mpf('.295758'))>m.mpf('.001')
assert abs(min(mods)-m.mpf('.0160837'))>m.mpf('.0001')
say('Singular example A, |ell gap|, bounded |S|^2:',*[Dprint(x) for x in (Av,abs(de),min(mods))])
# Candidate fusion computed with loggamma (full amplitudes), and comparison to quadrature-derived p.
Test=sol(m.cos(m.mpf('.31')))
close(A(Test,ss[0])*A(Test,ss[1]),A(Test,ss[2]))
close(phase(Test,ss[0])+phase(Test,ss[1]),phase(Test,ss[2]))
def logcandidate(T,S,h):
    eta=2*(ell(T)-ell(S))/(9*h);tau=m.sign(T[1]*S[1])
    return tau*(m.loggamma(1+1j*eta)-m.loggamma(1-1j*eta))+1j*eta*(m.log(A(T,S))+tau*(2-m.log(eta*eta)))+1j*topo(T,S)/h
for h in [m.mpf('.1'),m.mpf('1e-4'),m.mpf('1e-8')]:
 ratio=m.exp(logcandidate(Test,ss[0],h)+logcandidate(Test,ss[1],h)-logcandidate(Test,ss[2],h))
 assert abs(ratio-1)>1
 if h==m.mpf('.1'):
  close(m.arg(ratio),m.mpf('1.5158110455271881045'),m.mpf('5e-20'))
  et=2*(ell(Test)-ell(ss[2]))/(9*h)
  printed=m.exp(-(m.loggamma(1+1j*et)-m.loggamma(1-1j*et)-1j*et*(2-m.log(et*et))))
  assert abs(printed-ratio)>m.mpf('.1')
  say('Printed Gamma formula phase',Dprint(m.arg(printed)))
 say('Gamma fusion phase',h,Dprint(m.arg(ratio)))
rem=sum(sign*m.log(A(Test,S))*row[1]/S[0] for sign,S,row in zip((1,1,-1),ss,loop))
close(rem,m.mpf('.8755292547543045976'),m.mpf('5e-20'))
# Transmission itself telescopes at every nonzero q, stronger than differentiating an assumed sum.
for q in map(m.mpf,['-3','-.17','.11','.9','4']):
 close(trans(q,ss[0])[0]*trans(q,ss[1])[0],trans(q,ss[2])[0])
 close(trans(q,ss[0])[1]+trans(q,ss[1])[1],trans(q,ss[2])[1])
say('Classical/phonon fusion and target momentum remainder:',Dprint(rem))
def gamma(S):
    k,v=S[:2]
    return -3*(1-2*v*v)/(k*k*(1-4*v*v))
full_rem=rem-2*Test[0]*sum(sign*gamma(S)*row[1] for sign,S,row in zip((1,1,-1),ss,loop))
assert abs(full_rem-rem)>m.mpf('.01')
say('Full Darboux target Taylor remainder',Dprint(full_rem))
# A symbolic test of the package string/substitution pitfall, without importing it.
p_real=s.Symbol('p',real=True);expr=s.sympify('-p')
assert expr.subs(p_real,1)==expr
# Breather bound and polynomial interference minimum.
AA,zz=s.symbols('AA zz');poly=1-2*zz+AA*zz**2
assert s.simplify(poly.subs(zz,1/AA)-(1-1/AA))==0
for al,be in [(m.mpf('.8'),m.mpf('.3')),(m.mpf('1.4'),m.mpf('.7'))]:
    abr=(1-3*m.cot(al)**2)/(1+3*m.coth(be)**2)
    assert abr<m.mpf('.25')
# Independent asymptotic check of the minus Y family, and exact bilinear factorization.
for j in (0,1):close(densities(ss[j][1]*(-600),-600)[0],ss[j][0]**2/2,m.mpf('1e-65'))
close(densities(ss[2][1]*600,600)[0],ss[2][0]**2/2,m.mpf('1e-65'))
assert s.expand(3*(pa**2-pb**2)**2-(pa-pb)**2+(pa-pb)**4-4*(pa-pb)**2*C)==0

# The heavy+light -> other-light labels cannot be literal positive-mass fusion.
assert ss[0][2]+ss[2][2]>ss[1][2]
assert ss[1][2]+ss[2][2]>ss[0][2]
Oo=s.symbols('O12 O23 O13');et=s.Symbol('eta')
assert s.expand(et*Oo[0]+et*Oo[1]-et*Oo[2]-et*(Oo[0]+Oo[1]-Oo[2]))==0
# Good/bad equation soliton dispersion and unsigned defect scattering convention.
beta=m.mpf('.37');kb=m.sinh(beta);vb=m.cosh(beta);close((kb*vb)**2-kb**2-kb**4)
sigma=m.mpf('2.4');eta=-1;qD=-sigma/(2*m.sqrt(3));close(-2*m.sqrt(3)*qD,sigma)
assert -2*m.sqrt(3)*qD != eta*sigma
say('RUNTIME_SECONDS',round(time.perf_counter()-START,3))
print('ALL CHECKS PASSED')

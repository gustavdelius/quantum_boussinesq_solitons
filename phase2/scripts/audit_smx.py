#!/usr/bin/env python3
"""Independent audit, (c): no imports from any project computation.
Run from any directory. 60-digit arithmetic; asymptotic tolerances reflect the
finite step, not working precision. Original scripts are run only as subprocess
smoke tests, never to provide a numerical target. No files are written.
"""
from pathlib import Path
import os
import subprocess
import hashlib
import itertools
import mpmath as m
import sympy as s

m.mp.dps = 60
Q = m.mpf
TOL = Q('1e-48')
ROOT = Path(__file__).resolve().parents[2]
INPUTS = [ROOT/'phase2'/('smx_candidates'+ext) for ext in ('.tex','.pdf')]
INPUTS += list((ROOT/'phase2/scripts').glob('smx_*.py'))
HASHES = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}

def eq(a,b,tol=TOL):
    assert abs(a-b) < tol, (m.nstr(a,25),m.nstr(b,25),m.nstr(abs(a-b),8))

def data(z,e,g):
    # Tuple rather than the package's soliton dictionaries.
    k = m.sin(z) if g else m.sinh(z)
    v = e*(m.cos(z) if g else m.cosh(z))
    l = -e*m.cos(3*z) if g else e*m.cosh(3*z)
    return k,v,l,2*k,-2*k*v,Q(2)/3*k**3*v

def interaction(x,y,e,f,g):
    # Independent KP chord construction, not the package Hirota code path.
    k,v,*_ = data(x,e,g); h,w,*_ = data(y,f,g)
    vv = v if g else 1j*v; ww = w if g else 1j*w
    a,b = (vv/m.sqrt(3)+k)/2,(vv/m.sqrt(3)-k)/2
    c,d = (ww/m.sqrt(3)+h)/2,(ww/m.sqrt(3)-h)/2
    return (a-c)*(b-d)/((a-d)*(b-c))

def hirota(x,y,e,f,g):
    k,v,*_=data(x,e,g); h,w,*_=data(y,f,g)
    poly=lambda K,W: W*W-K*K+(1 if g else -1)*K**4
    return -poly(k-h,k*v-h*w)/poly(k+h,k*v+h*w)

def topo(x,y,e,f,g):
    a=data(x,e,g); b=data(y,f,g)
    return -Q(2)/3*(a[4]*b[3]-b[4]*a[3])

def phase(x,y,e,f,g):
    return Q(2)/9*(data(x,e,g)[2]-data(y,f,g)[2])*m.log(interaction(x,y,e,f,g))+topo(x,y,e,f,g)

def deriv(z,e,g,idx): return m.diff(lambda t:data(t,e,g)[idx],z)
def gamma(z,e,g): return deriv(z,e,g,4)/deriv(z,e,g,5)
def ring(x,y,e,f,g): return phase(x,y,e,f,g)+data(x,e,g)[4]*data(y,f,g)[3]
def delay(x,y,e,f,g): return -m.log(interaction(x,y,e,f,g))/data(x,e,g)[0]

def candidate(x,y,e,f,g,h,shift=1):
    # Direct Gamma quotient and complex power; no package logS implementation.
    if (e,f)==(-1,1): return 1/candidate(y,x,f,e,g,h,shift)
    eta=Q(2)/9*(data(x,e,g)[2]-data(y,f,g)[2])/h
    A=interaction(x,y,e,f,g)
    tau=1 if e==f else -1
    return (m.gamma(shift+1j*eta)/m.gamma(shift-1j*eta))**tau * m.exp(1j*eta*(m.log(A)+tau*(2-2*m.log(eta**2)/2))+1j*topo(x,y,e,f,g)/h)

def classical():
    errs=[]
    for g in (False,True):
        for e,f in itertools.product((1,-1),repeat=2):
            x,y=map(Q,('0.23','0.61'))
            a,b=data(x,e,g),data(y,f,g)
            eq(interaction(x,y,e,f,g),hirota(x,y,e,f,g))
            dxdy=m.diff(lambda z:delay(x,z,e,f,g),y)
            dydx=m.diff(lambda z:-delay(y,z,f,e,g),x)
            eq(deriv(x,e,g,5)*dxdy-deriv(y,f,g,5)*dydx,
               -deriv(y,f,g,4)*deriv(x,e,g,3)-deriv(x,e,g,4)*deriv(y,f,g,3))
            eq(deriv(x,e,g,5),2*a[0]*deriv(x,e,g,2)/9)
            dl=m.diff(lambda t:m.log(interaction(t,y,e,f,g)),x)
            eq(Q(2)/9*(a[2]-b[2])*dl,-deriv(x,e,g,4)*b[3]/3-2*deriv(x,e,g,3)*b[4]/3)
            actual=-m.diff(lambda t:phase(t,y,e,f,g),x)/deriv(x,e,g,5)
            eq(actual,delay(x,y,e,f,g)+gamma(x,e,g)*b[3])
            # Refuted printed minus sign: discrepancy is nonzero and exact.
            wrong=delay(x,y,e,f,g)-gamma(x,e,g)*b[3]
            eq(actual-wrong,2*gamma(x,e,g)*b[3]); assert abs(actual-wrong)>Q('.01')
            eq(-m.diff(lambda t:ring(t,y,e,f,g),x)/deriv(x,e,g,5),delay(x,y,e,f,g))
            shift=m.pi if g else 1j*m.pi
            eq(interaction(x,y+shift,1,1,g)*interaction(x,y,1,-1,g),1)
            eq(phase(x,y+shift,1,1,g)+phase(x,y,1,-1,g),0)
            eq(topo(x,y+shift,1,1,g)+topo(x,y,1,-1,g),0)
            # Physical charge energy derivative, and cubic root dictionary.
            energy=lambda z: Q(2)/15*data(z,e,g)[0]**3*(5+(-4 if g else 4)*data(z,e,g)[0]**2)
            eq(m.diff(energy,x)/deriv(x,e,g,5),a[1])
            for sign in (1,-1):
                root=((a[1] if g else 1j*a[1])/m.sqrt(3)+sign*a[0])/2
                eq(4*root**3+(-1 if g else 1)*root,(1 if g else 1j)*a[2]/(3*m.sqrt(3)))
        for n in (2,3,4):
            zs=[Q('.15')+Q('.11')*j for j in range(n)]
            es=[1 if j%2==0 else -1 for j in range(n)]
            ds=[data(z,e,g) for z,e in zip(zs,es)]
            for perm in itertools.permutations(range(n)):
                for i,j in itertools.combinations(perm,2):
                    eq(deriv(zs[j],es[j],g,5)*gamma(zs[j],es[j],g)*deriv(zs[i],es[i],g,3),deriv(zs[j],es[j],g,4)*deriv(zs[i],es[i],g,3))
            L=Q(50)
            B=[d[5]*L+d[4]*d[3]+sum(ring(zs[j],zs[l],es[j],es[l],g) for l in range(n) if l!=j) for j,d in enumerate(ds)]
            eq(sum(B),sum(d[5] for d in ds)*L+sum(d[4] for d in ds)*sum(d[3] for d in ds))
        # Missing self derivative in printed derivative of total Bethe phase.
        x=Q('.23'); missing=m.diff(lambda z:data(z,1,g)[4]*data(z,1,g)[3],x)/deriv(x,1,g,5)
        assert abs(missing)>1
        print('(c) missing self derivative, good=',g, m.nstr(missing,12))
    # The proposed seam generating function integrates Delta dP, not P dDelta.
    p,L,M=s.symbols('p L M'); C=s.Function('C')(p)
    assert s.diff(p*L+C*M,p)==L+M*s.diff(C,p)
    assert s.simplify(p*s.diff(L+M*s.diff(C,p),p)-s.diff(p*L+C*M,p))!=0
    print('(c) classical dictionary, corrected Darboux signs, conditional ring identities PASS')

def families():
    x,y=s.symbols('x y',positive=True)
    # Rational uniformisation x=exp(beta1), y=exp(beta2).
    block=lambda z: (4*(z+1)**2-4*z)/(z-1)**2
    A=block(x*y)/block(x/y)
    l=lambda z:(z**3+z**-3)/2
    dd=x*y*s.diff(s.Rational(2,9)*(l(x)-l(y))*s.log(A),x,y)
    assert not dd.has(s.log,s.polylog) and dd.is_rational_function(x,y)
    # Chain rule is tested using an argument independently different from A.
    R=lambda a,b: Q('.1')*(a+b+a*b)
    a,b=Q('.3'),Q('.7')
    lhs=m.diff(lambda u,v:m.polylog(2,R(u,v)),(a,b),(1,1))
    L=lambda u,v:m.log(R(u,v))
    rhs=R(a,b)/(1-R(a,b))*m.diff(lambda u:L(u,b),a)*m.diff(lambda v:L(a,v),b)-m.log(1-R(a,b))*m.diff(L,(a,b),(1,1))
    eq(lhs,rhs)
    # Counterexample to a universal arbitrary-argument/product weight exclusion.
    mixed=lambda u,v:m.polylog(2,-m.exp(u*v))+m.polylog(2,-m.exp(-u*v))
    eq(mixed(a,b),-m.pi**2/6-(a*b)**2/2)
    eq(m.diff(mixed,(a,b),(1,1)),-2*a*b)
    for g in (False,True):
        aa=Q('.43'); d=Q('1e-22')
        res=d*m.diff(lambda u,v:phase(u,v,1,1,g),(aa+d,aa),(1,1))
        expected=-Q(4)/3*(m.sin(3*aa) if g else m.sinh(3*aa))
        eq(res,expected,Q('1e-20'))
        print('(c) diagonal classical residue good=',g,m.nstr(res,16))
    for shift in (0,2*m.pi/3,-2*m.pi/3):
        aa=Q('.43'); d=Q('1e-22')
        u,v=aa+d+shift,aa+shift
        ratio=m.tan(u/2)/m.tan(v/2)
        residue=d*(-ratio/(1-ratio)/m.sin(u)/m.sin(v))
        eq(residue,1/m.sin(aa+shift),Q('1e-19'))
        assert abs(residue+1/(2*m.sin(aa+shift)))>Q('.1')
    # Independent exact three-pole residue argument: each csc has a distinct pole.
    for r in range(3):
        t=Q('1e-22'); angles=[2*m.pi*j/3 for j in range(3)]
        for j in range(3):
            eq(t/m.sin(-angles[r]+t+angles[j]),1 if r==j else 0,Q('1e-20'))
    D,S=s.symbols('D S'); F=s.Function('F'); G=s.Function('G')
    assert s.diff(F(D)*F(-D)*G(S)**2,S)==2*F(D)*F(-D)*G(S)*s.diff(G(S),S)
    # G constant is compatible, not inconsistent; unitarity does not exclude F.
    for d in (Q('.2'),Q('1.3')):
        eq(m.exp(1j*d**3)*m.exp(-1j*d**3),1)
    # Noncontinuous sign G is an excluded-by-regularity counterexample to 'alone'.
    assert all(v*v==1 for v in (-1,1))
    for g in (False,True):
        x,y=Q('.2'),Q('.5'); h=Q('.05'); cr=m.pi if g else 1j*m.pi
        for st in (Q(1),Q('.5'),Q('1.3')):
            for e,f in ((1,1),(1,-1)):
                z=candidate(x,y,e,f,g,h,st)
                eq(abs(z),1); eq(z*candidate(y,x,f,e,g,h,st),1)
            eq(candidate(x,y+cr,1,1,g,h,st)*candidate(x,y,1,-1,g,h,st),1)
    for g in (False,True):
        value=m.diff(lambda S,D:phase((S+D)/2,(S-D)/2,1,1,g)-topo((S+D)/2,(S-D)/2,1,1,g),(Q('1.2'),Q('.3')),(1,1))
        assert abs(value)>Q('.1')
        x,y,h,c=Q('.8'),Q('.3'),Q('1e-9'),Q('.4')
        ren=h*(1+c*h)
        ratio=candidate(x,y,1,1,g,ren)/candidate(x,y,1,1,g,h)
        eq(m.exp(1j*c*phase(x,y,1,1,g))*ratio,1,Q('1e-7'))
        dynratio=ratio*m.exp(1j*topo(x,y,1,1,g)*(1/h-1/ren))
        eq(m.exp(1j*c*(phase(x,y,1,1,g)-topo(x,y,1,1,g)))*dynratio,1,Q('1e-7'))
    z=s.symbols('z',positive=True)
    assert s.simplify(s.exp(s.I*(s.I*s.log(z)))-1/z)==0
    # Stirling expansion from loggamma, signed eta, all shifts and channels.
    for st in (Q(1),Q('.5'),Q('1.3')):
        for eta in (Q('1e7'),-Q('1e7')):
            val=2*m.im(m.loggamma(st+1j*eta))-2*eta*(m.log(abs(eta))-1)
            eq(val,(st-Q('.5'))*m.pi*m.sign(eta)-(st*st-st+Q(1)/6)/eta,Q('1e-20'))
    for eta in (Q('1e8'),-Q('1e8')):
        c=Q('.7'); z=(c+1j*eta)/(c-1j*eta)
        eq(m.arg(z),m.pi*m.sign(eta)-2*c/eta,Q('1e-23'))
        z=m.sinh(m.pi*(eta-Q('.35')*1j)/2)/m.sinh(m.pi*(eta+Q('.35')*1j)/2)
        eq(m.arg(z),-m.pi*Q('.35')*m.sign(eta))
    print('(c) restricted dilog exclusion repaired; mirror and Gamma algebra PASS')

def poles():
    for aa in map(Q,('.6','.9')):
        for n in (1,2,3):
            h=Q('.1'); b=(m.pi+m.asin(9*n*h/(4*m.sinh(3*aa))))/3
            x,y=aa-1j*b,aa+1j*b
            A=interaction(x,y,1,1,False); ieta=2j*(data(x,1,False)[2]-data(y,1,False)[2])/(9*h)
            eq(ieta,-n); assert m.re(A)>1 and m.im(data(x,1,False)[5]-data(y,1,False)[5])>0
            eq(A,(1+3*m.coth(aa)**2)/(1-3*m.cot(b)**2))
            residues=[]
            for t in (Q('1e-15'),Q('1e-19')):
                residues.append(t*candidate(aa-1j*(b+t),aa+1j*(b+t),1,1,False,h))
            eq(residues[0]/residues[1],1,Q('1e-11'))
        # Full threshold includes a logarithmic branch; finite limit, not analytic.
        z=[]
        for t in (Q('1e-15'),Q('1e-19')):
            b=m.pi/3+t; x,y=aa-1j*b,aa+1j*b
            z.append(candidate(x,y,1,1,False,Q('.1'))/m.exp(1j*topo(x,y,1,1,False)/Q('.1')))
        eq(z[1],1,Q('1e-14'))
    # Independently test half-shift and renormalised-hbar poles of full candidate.
    aa=Q('.6'); h=Q('.1')
    for st,hh in ((Q('.5'),h),(Q(1),h*(1+Q('.3')*h-Q('.1')*h*h))):
        b=(m.pi+m.asin(9*st*hh/(4*m.sinh(3*aa))))/3
        residues=[t*candidate(aa-1j*(b+t),aa+1j*(b+t),1,1,False,hh,st) for t in (Q('1e-15'),Q('1e-19'))]
        eq(residues[0]/residues[1],1,Q('1e-11'))
    # The old comment that Gamma(z)/Gamma(-z) has a zero-level pole is false.
    t=Q('1e-25'); eq(m.gamma(t)/m.gamma(-t),-1,Q('1e-23'))
    for aa in map(Q,('.2','.65','1.2','1.5')):
        for b in map(Q,('.1','.8','2')):
            A=interaction(aa-1j*b,aa+1j*b,1,1,True)
            eq(A,(1-3*m.cot(aa)**2)/(1+3*m.coth(b)**2)); assert m.re(A)<Q('.25')
    eq(1-3*m.cot(m.pi/3)**2,0)
    assert m.re(interaction(Q('.65')-Q('.1')*1j,Q('.65')+Q('.1')*1j,1,1,True))<0
    # Tau positivity requires A>1. Counterexample to the printed A>1/4 rule.
    A=Q('.5'); X=1/A; assert 1-2*X+A*X**2<0
    for aa in map(Q,('.4','.65')):
        h=Q('.05'); b=-m.asinh(9*h/(4*m.sin(3*aa)))/3
        x,y=aa-1j*b,aa+1j*b
        A=interaction(x,y,1,1,True); imP=m.im(data(x,1,True)[5]-data(y,1,True)[5])
        eq(2j*(data(x,1,True)[2]-data(y,1,True)[2])/(9*h),-1)
        eq(imP,m.sinh(2*b)*(m.cos(4*aa)*m.cosh(2*b)-m.cos(2*aa))/3)
        assert imP>0 and m.re(A)<0
        residues=[t*candidate(aa-1j*(b+t),aa+1j*(b+t),1,1,True,h) for t in (Q('1e-15'),Q('1e-19'))]
        eq(residues[0]/residues[1],1,Q('1e-11'))
        print('(c) good pole h=.05:',m.nstr(aa,4),m.nstr(b,13),m.nstr(imP,13),m.nstr(A,13))
    # False resonance |D|=pi/3, which is not a pole of the denominator.
    x=Q('.2'); y=x+m.pi/3
    assert abs(data(x,1,True)[2]-data(y,1,True)[2])>1
    assert abs(interaction(x,y,1,1,True))<10
    print('(c) false resonance A, ell gap:',m.nstr(interaction(x,y,1,1,True),12),m.nstr(data(x,1,True)[2]-data(y,1,True)[2],12))
    # Actual resonances have finite S limit but logarithmically divergent slope.
    for e,f,x,y0 in ((1,-1,Q('.4'),m.pi/3-Q('.4')),(1,1,Q('.75'),2*m.pi/3-Q('.75'))):
        eq(data(x,e,True)[2]-data(y0,f,True)[2],0)
        slopes=[]
        for t in (Q('1e-10'),Q('1e-20')):
            y=y0-t
            z=candidate(x,y,e,f,True,Q('.1'))/m.exp(1j*topo(x,y,e,f,True)/Q('.1'))
            assert abs(z-1)<Q('1e-6')
            slopes.append(abs(m.log(z)/t))
        assert slopes[1]>Q('1.8')*slopes[0]
    # Conditional fusion is charge additivity and an analytically continued delay.
    a,b,z=Q('.75'),Q('1.25'),Q('.4'); x,y=a-1j*b,a+1j*b
    eq(data(x,1,False)[3]+data(y,1,False)[3],4*m.sinh(a)*m.cos(b))
    eq(data(x,1,False)[4]+data(y,1,False)[4],-2*m.sinh(2*a)*m.cos(2*b))
    eq(m.im(ring(x,z,1,1,False)+ring(y,z,1,1,False)),0)
    eq(-m.diff(lambda t:ring(t,x,1,1,False)+ring(t,y,1,1,False),z)/deriv(z,1,False,5),-m.log(interaction(z,x,1,1,False)*interaction(z,y,1,1,False))/m.sinh(z))
    print('(c) poles, breather threshold, resonance discrepancies PASS')

W=(1,1,-2)
def angles(a): return (a+2*m.pi/3,a-2*m.pi/3,a)
def theta(q,a,e):
    # Elementary scattering phase from oneloop_phase definitions. No dilogs.
    r=abs(q); v=m.sqrt(1+q*q)
    return -sum(c*(m.sign(q)*m.atan(m.sin(p)/r)+e*m.atan(m.sin(p)*m.cos(p)/(r*v+r*r+m.sin(p)**2))) for p,c in zip(angles(a),W))
def dt(q,a,e):
    return sum(c*m.sin(p)/(q*q+m.sin(p)**2)*(1+e*m.sign(q)*m.cos(p)/m.sqrt(1+q*q)) for p,c in zip(angles(a),W))
def raw(a,b,e,f):
    cuts=sorted(set([Q(0),Q('.01'),Q('.1'),Q(1),Q(10)]+[abs(m.sin(p)) for p in angles(a)+angles(b)]))+[m.inf]
    sa=[(m.sin(p),m.cos(p),c) for p,c in zip(angles(a),W)]
    sb=[(m.sin(p),m.cos(p),c) for p,c in zip(angles(b),W)]
    # Pair the two half-lines algebraically; cache angle trigonometry only.
    # theta(+q)=-O_a-e E_a; theta(-q)=O_a-e E_a,
    # theta_b'(+q)=U_b+f V_b; theta_b'(-q)=U_b-f V_b.
    def paired(q):
        root=m.sqrt(1+q*q)
        odd=sum(c*m.atan(sn/q) for sn,cs,c in sa)
        even=sum(c*m.atan(sn*cs/(q*root+q*q+sn*sn)) for sn,cs,c in sa)
        even_d=sum(c*sn/(q*q+sn*sn) for sn,cs,c in sb)
        odd_d=sum(c*sn*cs/(q*q+sn*sn) for sn,cs,c in sb)/root
        return -2*(f*odd*odd_d+e*even*even_d)
    # Local check of the paired integrand against both literal half-lines.
    for q in (Q('.07'),Q('.8'),Q(5)):
        eq(paired(q),theta(q,a,e)*dt(q,b,f)+theta(-q,a,e)*dt(-q,b,f))
    return m.quad(paired,cuts)/(4*m.pi)
def ij(a,b,e=1,f=1):
    v=raw(a,b,e,f); return v,(v-raw(b,a,f,e))/2

def quadrature():
    for a,b in ((Q('.2'),Q('.55')),(Q('.35'),Q('.6'))):
        for q in (Q('-.8'),Q('.3')):
            eq(m.diff(lambda t:theta(t,a,1),q),dt(q,a,1))
            eq(theta(q,b+m.pi,1),-theta(q,b,-1))
        I,J=ij(a,b); Ih,Jh=ij(a,b,1,-1); Ic,Jc=ij(a,b+m.pi)
        eq(J-I,m.pi/2); eq(Ih,Jh); eq(Ic+Ih,0); eq(Jc+Jh,0)
        # Explicitly assert the discrepancy, instead of the false +pi/2 target.
        eq((Jc+Jh)-m.pi/2,-m.pi/2)
        print('(c) quadrature a,b,I,J,Ih,Jcross+Jh:',*[m.nstr(t,15) for t in (a,b,I,J,Ih,Jc+Jh)])
        # Literal printed J is neither this antisymmetrized integral nor its continuation.
        sig=(1,1,-1)
        f=lambda u,v:sum(c*m.sign(m.cos(p))*m.atan(m.tan(u)/m.tan(p)) for p,c in zip(angles(v),sig))
        Jprinted=f(a,b)-f(b,a)
        assert abs(Jprinted-J)>Q('.01')
        print('(c) printed J minus quadrature:',m.nstr(Jprinted-J,15))
        printed_I=0
        for p,c in zip(angles(a),sig):
            for q,d in zip(angles(b),sig):
                z=m.tan(p/2)/m.tan(q/2)
                printed_I+=c*d*(m.polylog(2,-z)-m.polylog(2,z)+m.log(abs(z))*m.log(abs(1-z)/(1+abs(z))))/m.pi
        assert abs(printed_I-I)>Q('.01')
        print('(c) literal printed I minus quadrature:',m.nstr(printed_I-I,15))
    # Resolve the narrow Lorentzian at the cusp explicitly with scaled cuts.
    a=Q('.2'); t=Q('1e-7')
    im,jm=ij(a,m.pi/3-t); ip,jp=ij(a,m.pi/3+t)
    # Initial audit expectation of a -pi/2 raw same-direction J jump failed.
    # Direct quadrature and unchanged endpoint product instead give continuity.
    eq(ip-im,0,Q('1e-4')); eq(jp-jm,0,Q('1e-4'))
    print('(c) upper-minus-lower cusp jumps I,J:',m.nstr(ip-im,12),m.nstr(jp-jm,12))
    # Algebraic constant conversion if a continuation C were imposed.
    cm,cl,C=s.symbols('cm cl C')
    assert s.expand(cm*s.pi/2-cl*s.pi/2-(cm-cl)*s.pi/2)==0
    eq(m.atan(Q(2))-m.atan(Q('.5')),Q('0.643501108793284386802809228717322638041510591115312382865606'))
    assert abs(m.atan(2)-m.atan(Q('.5'))-m.pi/2)>Q('.5')
    print('(c) direct I crossing confirmed; direct J offset and printed cancellation refuted')

def interfaces():
    x,y,r,Y=s.symbols('x y r Y',positive=True)
    A=x*y*(x-y)/(x+y)**3
    assert s.cancel(A+A.xreplace({x:y,y:x}))==0
    assert s.simplify(A.subs({x:Y,y:r*r*Y})-r*r*(1-r*r)/(1+r*r)**3)==0
    assert s.diff(A,x,y)!=0
    assert s.limit(A,x,0)==0 and s.limit(A,y,0)==0
    # Subleading noninteger power is not excluded by the classical limit.
    h,H=s.symbols('h H',positive=True)
    assert s.limit(h*H/s.sqrt(h),h,0)==0
    assert s.limit(H/s.sqrt(h),h,0)==s.oo
    # Explicit entire crossing-real residual H=sinh(beta1-beta2).
    assert s.simplify(s.sinh(x-y-s.I*s.pi)+s.sinh(x-y))==0
    assert s.simplify(s.sinh(x-y)+s.sinh(y-x))==0
    # Small-amplitude frame interface from the actual derivative of the action.
    r=Q('.6'); lam=Q('1e-12'); x,y=lam*r,lam
    for cl,cn in ((1/m.pi,(-m.log(2)-1+m.log(3)/4)/m.pi),(Q(0),-4/m.pi)):
        dp=lambda z:cl*z*m.log(z)+cn*z
        H=delay(x,y,1,1,True)*dp(x)-delay(y,x,1,1,True)*dp(y)
        tail=gamma(x,1,True)*2*m.sin(y)*dp(x)-gamma(y,1,True)*2*m.sin(x)*dp(y)
        head=(-m.diff(lambda z:phase(z,y,1,-1,True),x)/deriv(x,1,True,5)*dp(x)
              +m.diff(lambda z:phase(x,z,1,-1,True),y)/deriv(y,-1,True,5)*dp(y))
        eq(H,-2*cl*m.log(r)*m.log((1-r)/(1+r)),Q('1e-20'))
        eq(tail,-2*(1/r-r)*(cl*m.log(lam)+cn)-2*cl*m.log(r)/r,Q('1e-20'))
        eq(head,-2*(1/r+r)*(cl*m.log(lam)+cn)-2*cl*m.log(r)/r,Q('1e-20'))
        print('(c) frame Hirota, tail, head:',*[m.nstr(t,15) for t in (H,tail,head)])
    eq((interaction(x,y,1,-1,True)-1)/lam**2,3*r,Q('1e-20'))
    # Verify the mode-sum coefficient by independent charge expression.
    dpm=lambda z:-sum(c*m.sin(p)*m.log(abs(m.tan(p/2))) for p,c in zip(angles(z),W))/(2*m.pi)
    eq(dpm(lam)/lam, (m.log(lam)-m.log(2)-1+m.log(3)/4)/m.pi,Q('1e-20'))
    # KdV integral from elementary quadrature, not the package dilog evaluator.
    Lr=m.quad(lambda t:2*m.log(t)/(1-t*t),[0,r])
    Ik=2*Lr/m.pi; Jk=m.pi/2+Ik
    assert abs(Jk)>Q('.1')
    eq(Ik,-4/m.pi*(m.polylog(2,r)-m.polylog(2,-r))/2-2/m.pi*m.log(r)*m.log((1-r)/(1+r)))
    print('(c) KdV I,J:',m.nstr(Ik,15),m.nstr(Jk,15))
    print('(c) frame antisymmetric freedom matches locally; printed small-amplitude tails refuted')

def smoke():
    names=['smx_dictionary_and_families','smx_poles_good_vs_bad','smx_residual_ambiguity_interface','smx_twisted_ring_bethe']
    for name in names:
        proc=subprocess.run(['python3',name+'.py'],cwd=ROOT/'phase2/scripts',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=180)
        assert proc.returncode==0 and proc.stdout.strip().endswith('ALL CHECKS PASSED'),(name,proc.stdout,proc.stderr)
        print('(c) original smoke:',name,'ALL CHECKS PASSED')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in HASHES.items()),'Concurrent source change: re-audit needed'
    print('(c) audited source hashes:')
    for p,h in HASHES.items(): print(p.name,h)

if __name__=='__main__':
    classical()
    families()
    poles()
    quadrature()
    interfaces()
    smoke()
    print('ALL CHECKS PASSED')

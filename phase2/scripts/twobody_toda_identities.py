#!/usr/bin/env python3
"""70-digit independent finite-sea identities and symbolic edge bookkeeping.
Only twobody_toda_identities.json is written. No old checker is imported.
"""
from pathlib import Path
import json,time
import sympy as s
import mpmath as m
m.mp.dps=70
OUT=Path(__file__).with_suffix('.json')

def finite(N):
    h=m.mpf('.7');q=m.mpf(1);t=2*m.pi*h
    xx,ww=m.gauss_quadrature(N,'legendre');x=[q*z for z in xx];w=[q*z for z in ww]
    K=lambda d:2*m.re(m.digamma(1+1j*d/h))+2*m.log(h)
    Th=lambda d:2*h*m.im(m.loggamma(1+1j*d/h))+2*d*m.log(h)
    A=m.matrix(N);one=m.matrix([1]*N)
    for i in range(N):
        for j in range(N):A[i,j]=(t if i==j else 0)-w[j]*K(x[i]-x[j])
    Ai=A**-1;rho=Ai*one;eta=Ai*m.matrix(x)
    dot=lambda f,g:sum(wj*fj*gj for wj,fj,gj in zip(w,f,g))
    P=lambda p:p+dot([Th(p-z) for z in x],rho)
    D=lambda p:1+dot([K(p-z) for z in x],rho)
    V=lambda p:p+dot([K(p-z) for z in x],eta)
    F=lambda p:Ai*m.matrix([Th(p-z) for z in x])
    def Fe(p,e):return (Th(p-e)+dot([K(e-z) for z in x],F(p)))/t
    Phi=lambda p,z:Th(p-z)-dot([K(p-v) for v in x],F(z))
    n=dot([1]*N,rho);a=1/n;b=m.exp(-a/2)
    errs=[];algebra=[]
    for p in map(m.mpf,['1.2','1.5','-1.3']):
        for e in [q,-q]:algebra.append(abs(Phi(e,p)+t*Fe(p,e)))
        errs.extend([abs(t*(Fe(p,q)-Fe(p,-q))*D(q)/t-(1-D(p))),
                     abs(V(q)*(Fe(p,q)+Fe(p,-q))-(P(p)-V(p)))])
    errs.append(abs(P(q)-m.pi*h*n))
    pairs=[]
    for p,z in [(m.mpf('1.5'),m.mpf('1.2')),(m.mpf('1.5'),m.mpf('-1.3'))]:
        ep=m.sign(p)*q;ez=m.sign(z)*q
        four=lambda p,z,ep,ez:Phi(p,z)-Phi(p,ez)-Phi(ep,z)+Phi(ep,ez)
        fp=[Fe(p,e)-Fe(ep,e) for e in [q,-q]]
        fz=[Fe(z,e)-Fe(ez,e) for e in [q,-q]]
        sym=t*(fp[0]*fz[0]-fp[1]*fz[1])
        err=abs(four(p,z,ep,ez)+four(z,p,ez,ep)-sym);errs.append(err)
        pairs.append(dict(p=str(p),z=str(z),reciprocity_error=str(err)))
    # Independently invert physical momentum and differentiate that inverted map.
    y=m.mpf('.8');eps=m.mpf('1e-10')
    def root(yy):
        k=m.sqrt(yy*h/b);target=4*b*(k*m.cosh(k)-m.sinh(k))/a
        return m.findroot(lambda p:P(p)-P(q)-target,(q+m.mpf('.1'),q+2))
    pp=root(y);numerical=(root(y+eps)-root(y-eps))/(2*eps*h)
    rhs=2*m.sinh(m.sqrt(y*h/b))/(a*D(pp))
    jacerr=abs(numerical-rhs)
    assert jacerr<m.mpf('1e-18')
    bound=m.mpf('1e-15') if N==32 else m.mpf('1e-23')
    assert max(errs)<bound,(N,max(errs))
    assert max(algebra)<m.mpf('1e-65')
    return dict(N=N,a=str(a),max_fredholm_error=str(max(errs)),endpoint_evaluation_error=str(max(algebra)),jacobian_error=str(jacerr),pairs=pairs)

def symbolic():
    y,r,A,ep=s.symbols('y r A ep',positive=True)
    # Momentum inversion's derivative, and its one-body and two-body consequences.
    u=y-A*s.log(y)+s.Symbol('c')
    assert s.simplify(s.diff(u,y)-(1-A/y))==0
    prod=4*r/ep**2/((1-A*ep**2)*(1-A*ep**2/r**2))
    const=s.expand(s.series(prod,ep,0,2).removeO()).coeff(ep,0)
    assert s.simplify(const-4*A*(r+1/r))==0
    ss=s.Symbol('s1')
    G=4*y**s.Rational(3,2)/3
    primitive=8*y**s.Rational(5,2)/15-8*A*y**s.Rational(3,2)/9
    assert s.simplify(s.diff(primitive,y)-G*s.diff(u,y))==0
    C=s.simplify(ss+3*primitive/(4*G)-3*y/10)
    assert C==ss-A/2
    assert s.simplify(C.subs({ss:3/(4*s.pi),A:1/(2*s.pi)})-1/(2*s.pi))==0
    # Reference bookkeeping, including the source missing from the audit's PDE.
    B1,B2,B0,f1,f2,H=s.symbols('B1 B2 B0 f1 f2 H')
    U2=2*B0*(B2-B0);U1=2*B0*(B1-B0)
    N21=2*B1*B2-2*B0**2-H
    K12=H-U2;K21=N21-U1
    assert s.expand(K12+K21-2*(B1-B0)*(B2-B0))==0
    assert s.expand((K12-K21)/2-((H-N21)/2+(U1-U2)/2))==0
    canonical=K12-(B1-B0)*(B2-B0)
    assert s.expand(canonical-(H-B1*B2+B0*(B1-B2)+B0**2))==0
    b1=2/ep*(1+A*ep**2);b2=2*r/ep*(1+A*ep**2/r**2)
    head=s.series((b1-s.sqrt(s.pi))*(b2-s.sqrt(s.pi))-4*r/ep**2,ep,0,1).removeO()
    assert s.simplify(head-(-2*s.sqrt(s.pi)*(1+r)/ep+s.pi+4*A*(r+1/r)))==0
    actual_source=2*f1*B2-2*B0*f2
    printed_source=2*f1*(B2-B0)
    assert s.expand(actual_source-printed_source-2*B0*(f1-f2))==0
    return dict(momentum_log_coefficient=str(1/(2*m.pi)),onebody_C=str(1/(2*m.pi)),sound=str(3/(4*m.pi)),sound_subtracted=str(-1/(4*m.pi)),symmetric_formula=str(const-s.pi),four_term_PDE_discrepancy=str(s.factor(actual_source-printed_source)))

def integral():
    rows=[]
    for r in map(m.mpf,['.2','.4','.5','.6','.8','.9']):
        val=4/m.pi*m.quad(lambda t:m.log(t/r)*t*t/(1-t*t),[0,r])
        closed=-4/m.pi*((m.polylog(2,r)-m.polylog(2,-r))/2-r)
        err=abs(val-closed);assert err<m.mpf('1e-65')
        # Independently integrate before the nontrivial rational substitution.
        original=-1/m.pi*m.quad(lambda t:m.sqrt(t)/(1-r*r+t)**m.mpf('1.5')*m.log((1-r*r+t)*r*r/t),[0,r*r])
        assert abs(original-closed)<m.mpf('1e-65')
        rows.append(dict(r=str(r),integral=str(val),closed=str(closed),error=str(err)))
    return rows

def dictionary():
    b=m.exp(-m.mpf('.5'));rows=[]
    for k1,k2 in [(m.mpf('.2'),m.mpf('.7')),(m.mpf('.8'),m.mpf('.3'))]:
        for sign in [1,-1]:
            w1=2*b*m.sinh(k1);w2=sign*2*b*m.sinh(k2)
            pt=lambda k,w:w*w-4*b*b*m.sinh(k/2)**2
            cross=-pt(2*k1-2*k2,w1-w2)/pt(2*k1+2*k2,w1+w2)
            f=m.sinh if sign==1 else m.cosh
            closed=(f((k1-k2)/2)/f((k1+k2)/2))**2
            assert abs(cross-closed)<m.mpf('1e-65')
    # Leading scaled classical phases, using independent hyperbolic formulas.
    h=m.mpf('1e-30');Y=m.mpf(16);r=m.mpf('.5');k1=m.sqrt(h*Y/b);k2=r*k1
    for sign in [1,-1]:
        f=m.sinh if sign==1 else m.cosh
        la=2*m.log(f((k1-k2)/2)/f((k1+k2)/2))
        delta=(2*b*m.cosh(k1)-sign*2*b*m.cosh(k2))*la+4*k1*k2*b*(m.sinh(k1)/k1-sign*m.sinh(k2)/k2)
        target=2*Y*(1-r*r)*m.log((1-r)/(1+r)) if sign==1 else 4*r*Y
        err=abs(delta/h-target);assert err<m.mpf('1e-25')
        rows.append(dict(sign=sign,scaled_classical_error=str(err)))
    beta=m.mpf('1e-20');k=m.sqrt(3)*beta
    P=4*b*(k*m.cosh(k)-m.sinh(k))/(6*m.sqrt(3)*b)
    assert abs(P/beta**3-m.mpf(2)/3)<m.mpf('1e-25')
    return rows

def main():
    start=time.monotonic();result=dict(symbolic=symbolic(),integral=integral(),dictionary=dictionary(),finite=[finite(32),finite(48)])
    result['seconds']=time.monotonic()-start
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['finite'],indent=2));print('ALL CHECKS PASSED')

if __name__=='__main__':main()

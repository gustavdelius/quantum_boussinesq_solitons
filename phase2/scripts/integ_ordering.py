#!/usr/bin/env python3
"""One-loop local Wick-to-Weyl mixing in the stated contraction-cutoff scheme.

Cut off vacuum contractions, not the continuum canonical Poisson operator.
This checks the formal local Ward identity, not a finite-mode regulator or
existence of an interacting continuum operator on a Hilbert space.
"""
import sympy as sp
import mpmath as mp
from integ_charges import u,p,s,h,q,j,dx,euler,bracket,zero_functional

def checks():
    A,B,C,D,E=sp.symbols('A B C D E',real=True)
    # Covariance/2 times the second derivative, including EVERY jet in H,Q.
    # Symmetrized mixed u-p covariances vanish. <u u_xx>/hbar=-B;
    # this off-diagonal entry is retained even though these densities do not use it.
    jets=[u[0],u[1],u[2],p[0],p[1]]
    cov=sp.Matrix([[A,0,-B,0,0],[0,B,0,0,0],[-B,0,C,0,0],
                   [0,0,0,D,0],[0,0,0,0,E]])
    def contraction(f):
        return sp.expand(sum(cov[a,b]*sp.diff(f,jets[a],jets[b])/2
                             for a in range(5) for b in range(5)))
    th=contraction(h); tq=contraction(q)
    assert sp.expand(th-((D+s*A+B)/2-3*A*u[0]))==0
    assert sp.expand(tq-(18*A*u[0]**2-(3*D+9*B+3*s*A)*u[0]+E+C+s*B))==0
    assert contraction(-u[0]**3)==-3*A*u[0]
    # General second-contraction parity, with independent symbolic derivatives.
    f20,f11,f02,g20,g11,g02=sp.symbols('f20 f11 f02 g20 g11 g02')
    second_fg=f20*g02-2*f11*g11+f02*g20
    second_gf=g20*f02-2*g11*f11+g02*f20
    assert sp.expand(second_fg-second_gf)==0
    # A formal Weyl commutator has no hbar^1 term after division by i*hbar.
    x,y,z,w=sp.symbols('x y z w'); hh=sp.Symbol('hbar')
    f=x**4+x*y**2+y**2; g=w**2/2-z**3+z**2/2
    def star2(f,g):
        ans=0
        for n in range(3):
            ans+=(sp.I*hh/2)**n/sp.factorial(n)*sum(
                (-1)**r*sp.binomial(n,r)*sp.diff(f,x,n-r,y,r)*
                sp.diff(g,z,r,w,n-r) for r in range(n+1))
        return sp.expand(ans.subs({z:x,w:y}))
    rev=star2(g.subs({z:x,w:y}),f.subs({x:z,y:w}, simultaneous=True))
    reduced=sp.expand((star2(f,g)-rev)/(sp.I*hh))
    assert sp.expand(reduced).coeff(hh,1)==0
    # Direct two-contraction Wick calculation for the only surviving nonlinear
    # term: [-3 u(x)pi(x)^2, -u(y)^3]. Cuu is real, Cpi,u=i delta'/2.
    cc,dd=sp.symbols('Cuu delta_prime',real=True)
    left=-3*x*y**2; right=-z**3
    def two(cross):
        return sp.expand(sum(cross[a,b]*cross[c,d]*sp.diff(left,(x,y)[a],(x,y)[c])*
                              sp.diff(right,(z,w)[b],(z,w)[d])/2
                              for a in range(2) for b in range(2)
                              for c in range(2) for d in range(2)))
    cross=sp.Matrix([[cc,sp.I*dd/2],[sp.I*dd/2,sp.Symbol('Cpp')]])
    source=sp.expand((two(cross)-two(sp.conjugate(cross)))/sp.I)
    assert source==36*cc*dd*y*z
    # Complete double-contraction sum, including every derivative jet.
    from math import comb
    specifications=[('phi',1),('phi',2),('phi',3),('pi',0),('pi',1)]
    moments={('phi',2):-A,('phi',4):B,('phi',6):-C,
             ('pi',0):D,('pi',2):-E}
    def moment(kind,n):
        if n%2: return sp.Integer(0)
        return moments[(kind,n)]
    direct=0
    for a,ja in enumerate(jets):
        for b,jb in enumerate(jets):
            species_a,order_a=specifications[a]; species_b,order_b=specifications[b]
            if species_a!=species_b: continue
            for c,jc in enumerate(jets):
                for d,jd in enumerate(jets):
                    species_c,order_c=specifications[c]; species_d,order_d=specifications[d]
                    if species_c==species_d: continue
                    left2=sp.diff(q,ja,jc); right2=sp.diff(h,jb,jd)
                    if left2==0 or right2==0: continue
                    n=order_c+order_d
                    sign=(-1)**(order_b+order_d)*(1 if species_c=='phi' else -1)
                    # Product C^(a+b)(r) delta^(n)(r), integrated against right2.
                    for ell in range(n+1):
                        right_derivative=right2
                        for _ in range(n-ell): right_derivative=dx(right_derivative)
                        if right_derivative==0: continue
                        direct+=sign*(-1)**ell*comb(n,ell)*moment(species_a,order_a+order_b+ell)*left2*right_derivative
    assert sp.expand(direct-36*A*p[0]*u[1])==0
    # C(r) delta'(r)=C(0) delta'(r), since C is even. This yields +36 A pi u_x.
    zero_functional(-18*A*bracket(u[0]**2,h)-36*A*p[0]*u[1])
    assert euler(-18*A*bracket(u[0]**2,h),p)==36*A*u[1]
    la=sp.Symbol('lambda')
    # Weyl symbols of :H:+hbar lambda:I2: and corrected :Q:.
    h1=-th+la*u[0]**2
    q1=-tq+18*A*u[0]**2+2*la*j
    zero_functional(bracket(q,h1)+bracket(q1,h))
    zero_functional(bracket(q,h1)+bracket(-tq+2*la*j,h)+18*A*bracket(u[0]**2,h))
    # Thus merely deforming the normal-ordered charge does NOT cancel the Wick source.
    assert euler(bracket(q,h1)+bracket(-tq+2*la*j,h),p)==36*A*u[1]
    # The source is independent of lambda. Adding conserved charges cannot cancel it.
    for density in (u[0],p[0],-u[0]*p[0],h,q):
        zero_functional(bracket(density,h))
    # Finite-cutoff primitive identities and independent 60-digit quadratures.
    L=sp.Symbol('Lambda',positive=True); R=sp.sqrt(1+L**2)
    formulas=[(R-1)/(2*sp.pi),(R**3/3-R+sp.Rational(2,3))/(2*sp.pi),
              (R**5/5-2*R**3/3+R-sp.Rational(8,15))/(2*sp.pi),
              (R**3-1)/(6*sp.pi),(R**5/5-R**3/3+sp.Rational(2,15))/(2*sp.pi)]
    integrands=[L/R,L**3/R,L**5/R,L*R,L**3*R]
    for form,integrand in zip(formulas,integrands):
        assert sp.simplify(sp.diff(form,L)-integrand/(2*sp.pi))==0
        assert form.subs(L,0)==0
    assert sp.simplify(formulas[3]-formulas[0]-formulas[1])==0
    assert sp.simplify(formulas[4]-formulas[1]-formulas[2])==0
    assert sp.simplify(((D+A+B)/2).subs(D,A+B)-(A+B))==0
    # Q linear and vacuum terms at s=1: 6A+12B and 2(B+C).
    assert sp.expand((3*D+9*B+3*A).subs(D,A+B)-(6*A+12*B))==0
    assert sp.expand((E+C+B).subs(E,B+C)-2*(B+C))==0
    assert sp.limit(formulas[0]/L,L,sp.oo)==1/(2*sp.pi)
    assert sp.limit(18*formulas[0]-9*L/sp.pi,L,sp.oo)==-9/sp.pi
    mp.mp.dps=60; worst=mp.mpf(0)
    numerical=[sp.lambdify(L,z,'mpmath') for z in formulas]
    functions=[lambda z:z/mp.sqrt(1+z*z),lambda z:z**3/mp.sqrt(1+z*z),
               lambda z:z**5/mp.sqrt(1+z*z),lambda z:z*mp.sqrt(1+z*z),
               lambda z:z**3*mp.sqrt(1+z*z)]
    for cutoff in map(mp.mpf,('.4','2','7')):
        for closed,integrand in zip(numerical,functions):
            num=mp.quad(integrand,[0,cutoff])/(2*mp.pi)
            error=abs(num-closed(cutoff)); worst=max(worst,error)
            assert error<mp.mpf('1e-53')
    print('All one-contraction terms and the local two-contraction Ward source checked.')
    print('Uncorrected Wick charge fails (retained negative check); local mixing cancels it for every lambda.')
    print('60-digit contraction quadrature maximum absolute error:',mp.nstr(worst,8))
    print('No claim of finite-mode integrability, continuum operator convergence, or higher Moyal orders.')

if __name__=='__main__':
    checks()
    print('ALL CHECKS PASSED')

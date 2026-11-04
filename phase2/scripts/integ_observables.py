#!/usr/bin/env python3
"""Static vacuum matching and sound-unit invariants, conditional on the trace.

The finite part of the absolute two-body phase is NOT computed here.
Independent phase formulas below are copied as mathematical inputs from the
stated Darboux convention, then differentiated and tested by inverse momenta.
"""
import sympy as sp
import mpmath as mp

def checks():
    U,L,la,eps,rho=sp.symbols('U Lambda lambda eps rho',real=True)
    W=-((1-6*U)**sp.Rational(3,2)-1+9*U)/(6*sp.pi)
    assert W.subs(U,0)==0 and sp.diff(W,U).subs(U,0)==0
    assert sp.simplify(sp.diff(W,U,2)+9/(2*sp.pi*sp.sqrt(1-6*U)))==0
    energy=U**2/2-U**3+eps*(W+la*U**2)
    assert sp.diff(energy,U).subs(U,0)==0
    modulus=sp.diff(energy,U,2)
    assert sp.simplify(modulus.subs(U,0)-1-eps*(2*la-9/(2*sp.pi)))==0
    s1=la-9/(4*sp.pi); Cg=la-2/sp.pi
    assert sp.simplify(Cg-s1-1/(4*sp.pi))==0
    assert sp.solve(s1,la)==[9/(4*sp.pi)]
    assert sp.simplify(sp.diff(sp.sqrt(modulus.subs(U,eps*rho)),eps).subs(eps,0)
                       -(s1-3*rho))==0
    k=sp.Symbol('k',positive=True)
    F0=sp.sqrt(3)/sp.pi*(sp.Rational(3,4)-k**2)*sp.atanh(2*k/sp.sqrt(3))-3*k/(2*sp.pi)
    F=F0+2*la*k**3/3
    P=2*k**3*sp.sqrt(1-k**2)/3
    E=2*k**3/3-8*k**5/15
    v=sp.sqrt(1-k**2)
    assert sp.simplify(sp.diff(E,k)-v*sp.diff(P,k))==0
    for orientation in (-1,1):
        assert sp.simplify(sp.diff(E,k)-(orientation*v)*sp.diff(orientation*P,k))==0
    assert sp.expand(5*E-4*P*v-2*k**3/3)==0
    assert sp.simplify(sp.limit(F/P,k,0)-Cg)==0
    FR=sp.simplify(F-s1*(5*E-4*P*v))
    assert sp.diff(FR,la)==0
    assert sp.simplify(sp.limit(FR/P,k,0)-1/(4*sp.pi))==0
    # The action, symplectic form, energy and momentum all rescale.
    c=sp.Symbol('c',positive=True)
    assert sp.simplify(2*c**2*(c*k)**3/3-8*(c*k)**5/15-c**5*E)==0
    assert sp.simplify(2*(c*k)**3*sp.sqrt(c**2-(c*k)**2)/3-c**4*P)==0
    length=1/c; time=1/c**2; field=c
    coefficients=[field**2*length/time,c**2*field**2*time/length,
                  field**3*time/length**2,field**2*time/length**3]
    assert all(sp.simplify(a-c**3)==0 for a in coefficients)
    assert sp.simplify(field**2/time-c**4)==0  # momentum
    assert sp.simplify(field**2*length/time**2-c**5)==0  # energy
    assert sp.simplify((field/time)*field*length-c**3)==0  # symplectic form
    # PDE scaling: u=c^2 w(cx,c^2t); all three terms carry c^6.
    assert c**2/time**2==c**2*c**2/length**2==c**2/length**4==c**4/length**2
    p1,p2=sp.symbols('P1 P2'); delta=sp.Function('delta0')
    scaled=c**3*delta(p1/c**4,p2/c**4)
    K=sp.diff(scaled,c).subs(c,1).doit()
    expected=3*delta(p1,p2)-4*(p1*sp.diff(delta(p1,p2),p1)+p2*sp.diff(delta(p1,p2),p2))
    assert sp.simplify(K-expected)==0
    assert sp.diff(sp.sqrt(1+2*eps*la),eps).subs(eps,0)==la
    cR=1+eps*s1
    phase_numerator=delta(cR**4*p1,cR**4*p2)-cR**3*delta(p1,p2)
    assert sp.simplify(sp.diff(phase_numerator,eps).subs(eps,0).doit()+s1*K)==0
    seed,kk=sp.symbols('unknown_seed K')
    assert sp.simplify(seed+la*kk-s1*kk-(seed+9*kk/(4*sp.pi)))==0
    # Independent 70-digit fixed-momentum inversion for the phase response.
    mp.mp.dps=70
    def sol(a,e):
        k=mp.sin(a); v=e*mp.cos(a)
        return dict(a=a,e=e,k=k,v=v,M=2*k,P=2*v*k**3/3,
                    ell=-e*mp.cos(3*a),pa=(k+v/mp.sqrt(3))/2,pb=(v/mp.sqrt(3)-k)/2,
                    gamma=-3*mp.cos(2*a)/(mp.sin(a)*mp.sin(3*a)))
    def phase(a,b):
        cross=(a['pa']-b['pa'])*(a['pb']-b['pb'])/((a['pa']-b['pb'])*(a['pb']-b['pa']))
        assert cross>0
        log=mp.log(cross)
        d=2*(a['ell']-b['ell'])*log/9+2*a['M']*b['M']*(a['v']-b['v'])/3
        x1=-log/a['k']+a['gamma']*b['M']; x2=log/b['k']-b['gamma']*a['M']
        return d,x1,x2
    worst=mp.mpf(0); values=[]
    for e1,e2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        a=sol(mp.mpf('.3'),e1); b=sol(mp.mpf('.6'),e2)
        d,x1,x2=phase(a,b); response=3*d+4*(a['P']*x1+b['P']*x2)
        for index,item in enumerate((a,b)):
            deriv=mp.diff(lambda angle:phase(sol(angle,e1),b)[0] if index==0 else phase(a,sol(angle,e2))[0],item['a'])
            dp=mp.diff(lambda angle:sol(angle,item['e'])['P'],item['a'])
            assert abs(deriv/dp+(x1,x2)[index])<mp.mpf('1e-60')
        def scaled_at_fixed_P(cc):
            aa=mp.findroot(lambda angle:cc**4*sol(angle,e1)['P']-a['P'],a['a'],tol=mp.mpf('1e-65'))
            bb=mp.findroot(lambda angle:cc**4*sol(angle,e2)['P']-b['P'],b['a'],tol=mp.mpf('1e-65'))
            return cc**3*phase(sol(aa,e1),sol(bb,e2))[0]
        step=mp.mpf('1e-18')
        error=abs((scaled_at_fixed_P(1+step)-scaled_at_fixed_P(1-step))/(2*step)-response)
        worst=max(worst,error); assert error<mp.mpf('1e-32')
        dd,y1,y2=phase(b,a)
        assert abs(response+3*dd+4*(b['P']*y1+a['P']*y2))<mp.mpf('1e-60')
        values.append((e1,e2,mp.nstr(response,25)))
    # Curvature is checked by an independent convergent integral.
    for uu in map(mp.mpf,('-.08','0','.02','.10')):
        integral=-9/(2*mp.pi)*mp.quad(lambda z:z/(z*z+1-6*uu)**mp.mpf('1.5'),[0,1,mp.inf])
        assert abs(integral+9/(2*mp.pi*mp.sqrt(1-6*uu)))<mp.mpf('1e-60')
    print('Static pressure leaves lambda free; modulus matching and sound-unit identities passed.')
    print('70-digit inverse-momentum phase response maximum error:',mp.nstr(worst,8))
    print('K_lambda at alpha=(.3,.6):',values)
    print('Invariance is restricted to the single elastic direction and the fixed boundary/frame convention.')

if __name__=='__main__':
    checks()
    print('ALL CHECKS PASSED')

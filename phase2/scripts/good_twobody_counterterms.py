"""Classical scaling computes local coupling responses at fixed momentum.

H=(1+2*h*kappa)pi^2/2+(1+2*h*lambda)u^2/2
  -(1-h*eta)u^3+(1+2*h*zeta)u_x^2/2.
No quantum determinant or boundary convention is inferred from this test.
"""
import sys
sys.dont_write_bytecode=True
import sympy as sp
import mpmath as mp
from good_twobody_kernels import sol, classical, save, momentum_mode

if __name__=='__main__':
    mp.mp.dps=75
    a,b,d,g=sp.symbols('a b d g',positive=True)
    length=sp.sqrt(d/b)
    time=sp.sqrt(d/a)/b
    field=sp.sqrt(b*d)/g
    scale=d*b**sp.Rational(3,2)/(sp.sqrt(a)*g**2)
    pscale=b**2*sp.sqrt(d)/(sp.sqrt(a)*g**2)
    # Four independently transformed coefficients of the action.
    coeffs=[field**2*length/(a*time),b*field**2*time/length,
            g*field**3*time/length**2,d*field**2*time/length**3]
    assert all(sp.simplify(z-scale)==0 for z in coeffs)
    assert sp.simplify(field**2/(a*time)-pscale)==0
    h,la,eta,ze,ka=sp.symbols('h lambda eta zeta kappa')
    replacement={a:1+2*h*ka,b:1+2*h*la,d:1+2*h*ze,g:1-h*eta}
    assert sp.simplify(sp.diff(scale.subs(replacement),h).subs(h,0)-(-ka+3*la+2*eta+2*ze))==0
    assert sp.simplify(sp.diff(pscale.subs(replacement),h).subs(h,0)-(-ka+4*la+2*eta+ze))==0
    k=sp.symbols('k',positive=True)
    v2=1-k**2
    E=2*k**3/3-8*k**5/15
    vP=2*v2*k**3/3
    assert sp.expand(5*E-4*vP-2*k**3/3)==0
    assert sp.expand(2*E-2*vP-4*k**5/15)==0
    assert sp.expand(E-vP-2*k**5/15)==0
    assert sp.simplify(sp.diff(E,k)-sp.sqrt(1-k*k)*sp.diff(2*sp.sqrt(1-k*k)*k**3/3,k))==0
    # Profile moments and physical Hamiltonian insertions (independent integrals).
    moments=[]
    for kval in (mp.mpf('.3'),mp.mpf('.7')):
        u=lambda x:kval*kval/(2*mp.cosh(kval*x/2)**2)
        vals=[mp.quad(lambda x:u(x)**n,[-mp.inf,0,mp.inf]) for n in (1,2,3)]
        ux2=mp.quad(lambda x:mp.diff(u,x)**2,[-mp.inf,0,mp.inf])
        targets=[2*kval,2*kval**3/3,4*kval**5/15]
        assert max(abs(v-w) for v,w in zip(vals,targets))<mp.mpf('1e-65')
        assert abs(ux2-2*kval**5/15)<mp.mpf('1e-65')
        moments.append(dict(k=kval,integrals=vals,gradient_integral=ux2))
    results=[]
    worst=mp.mpf(0)
    for e,f in ((1,1),(1,-1),(-1,1),(-1,-1)):
        s,t=sol('.3',e),sol('.6',f)
        c=classical(s,t)
        # Check the Darboux gradient directly, without using it to define delta0.
        for index,z in enumerate((s,t)):
            dp=mp.diff(lambda x:sol(x,z['e'])['P'],z['a'])
            deriv=mp.diff(lambda x:classical(sol(x,e),t)['delta0'] if index==0
                          else classical(s,sol(x,f))['delta0'],z['a'])
            assert abs(deriv/dp+c['X1' if index==0 else 'X2'])<mp.mpf('1e-65')
        # Log coefficient of lambda response; finite part is independently retained.
        finite=3*c['top']+4*(s['P']*s['gamma']*t['M']-t['P']*t['gamma']*s['M'])
        assert abs(c['lambda_coefficient']-(2*(t['v']-s['v'])/3*c['logA']+finite))<mp.mpf('1e-65')
        for name in ('lambda','eta','zeta','kappa'):
            def exact_phase(hh):
                aa=1+2*hh*(name=='kappa')
                bb=1+2*hh*(name=='lambda')
                dd=1+2*hh*(name=='zeta')
                gg=1-hh*(name=='eta')
                ss=dd*bb**mp.mpf('1.5')/(mp.sqrt(aa)*gg**2)
                pp=bb**2*mp.sqrt(dd)/(mp.sqrt(aa)*gg**2)
                al1=mp.findroot(lambda x:pp*sol(x,e)['P']-s['P'],s['a'],tol=mp.mpf('1e-72'))
                al2=mp.findroot(lambda x:pp*sol(x,f)['P']-t['P'],t['a'],tol=mp.mpf('1e-72'))
                return ss*classical(sol(al1,e),sol(al2,f))['delta0']
            # Finite differences of exact inverse momenta, not derivative of the response formula.
            step=mp.mpf('1e-18')
            derivative=(exact_phase(step)-exact_phase(-step))/(2*step)
            error=abs(derivative-c[name+'_coefficient'])
            worst=max(worst,error)
            assert error<mp.mpf('1e-32'),(name,e,f,error)
            assert abs(c[name+'_coefficient']+classical(t,s)[name+'_coefficient'])<mp.mpf('1e-65')
        def converted(hh):
            al1=mp.findroot(lambda x:sol(x,e)['P']+hh*momentum_mode(sol(x,e))-s['P'],s['a'])
            al2=mp.findroot(lambda x:sol(x,f)['P']+hh*momentum_mode(sol(x,f))-t['P'],t['a'])
            return classical(sol(al1,e),sol(al2,f))['delta0']
        step=mp.mpf('1e-18')
        response=(converted(step)-converted(-step))/(2*step)
        assert abs(response-c['X1']*momentum_mode(s)-c['X2']*momentum_mode(t))<mp.mpf('1e-32')
        results.append(dict(epsilon=[e,f],alpha=['.3','.6'],**c))
    # One-loop UV expansion, evaluated independently through the frequency integral.
    q,U=sp.symbols('q U',positive=True)
    w=sp.sqrt(q*q+q**4-6*U*q*q)/2
    assert sp.simplify(sp.diff(w,U).subs(U,0)+3*q/(2*sp.sqrt(1+q*q)))==0
    assert sp.simplify(sp.diff(w,U,2).subs(U,0)/2+9*q/(4*(1+q*q)**sp.Rational(3,2)))==0
    assert sp.limit(sp.diff(w,U).subs(U,0),q,sp.oo)==-sp.Rational(3,2)
    assert sp.limit(q*q*sp.diff(w,U,2).subs(U,0)/2,q,sp.oo)==-sp.Rational(9,4)
    for n in range(1,5):
        coefficient=sp.diff(w,U,n).subs(U,0)/sp.factorial(n)
        assert sp.limit(coefficient/q**(2-2*n),q,sp.oo)==sp.binomial(sp.Rational(1,2),n)*(-6)**n/2
    save('good_twobody_counterterms.json',dict(precision=75,finite_difference_step='1e-18',
          max_inverse_momentum_error=worst,moments=moments,samples=results,
          scope='bulk coupling response in the stipulated classical Darboux frame; no boundary counterterms fixed'))
    print('max fixed-momentum finite-difference error',mp.nstr(worst,8))
    print('ALL CHECKS PASSED')

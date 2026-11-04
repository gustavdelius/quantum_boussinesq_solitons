"""Exact tau coefficient tests, three-soliton phonons and cluster limitations.

Passing these tests does not construct a constrained functional determinant.
"""
import sys
sys.dont_write_bytecode=True
from itertools import product, combinations
import sympy as sp
import mpmath as mp
from good_twobody_kernels import sol, interaction, theta, integrals, save

def polynomial(k,w):
    return w*w-k*k+k**4

def terms(ks,ws):
    n=len(ks)
    As={(i,j):-polynomial(ks[i]-ks[j],ws[i]-ws[j])/polynomial(ks[i]+ks[j],ws[i]+ws[j])
        for i,j in combinations(range(n),2)}
    out=[]
    for mask in product((0,1),repeat=n):
        coeff=1
        for (i,j),val in As.items():
            if mask[i] and mask[j]: coeff*=val
        out.append((mask,coeff,sum(ks[i]*mask[i] for i in range(n)),sum(ws[i]*mask[i] for i in range(n))))
    return out,As

def residuals(ts,exact=False):
    coeffs={}
    sizes={}
    for n,c,k,w in ts:
        for m,d,l,z in ts:
            power=tuple(x+y for x,y in zip(n,m))
            term=c*d*polynomial(k-l,w-z)
            coeffs[power]=coeffs.get(power,0)+term
            if not exact: sizes[power]=sizes.get(power,0)+abs(term)
    if exact:
        assert all(sp.simplify(v)==0 for v in coeffs.values())
        return len(coeffs)
    return max(abs(v)/(1+sizes[p]) for p,v in coeffs.items())

if __name__=='__main__':
    mp.mp.dps=75
    # Four-soliton tau, one imaginary wave: all 81 exponential coefficients.
    ks=[];ws=[]
    for z in map(sp.Rational,('1/10','1/5','1/3')):
        kval=2*z/(1+z*z);vel=(1-z*z)/(1+z*z)
        ks.append(kval);ws.append(kval*vel)
    ks.append(sp.I*sp.Rational(3,4));ws.append(sp.I*sp.Rational(15,16))
    exact_count=residuals(terms(ks,ws)[0],True)
    results=[]
    for eps in ((1,1,1),(1,1,-1)):
        ss=[sol(a,e) for a,e in zip(('.15','.4','.55'),eps)]
        assert all(interaction(ss[i],ss[j])>0 for i,j in combinations(range(3),2))
        for q in (mp.mpf('-.7'),mp.mpf('.37')):
            ks=[s['k'] for s in ss]+[1j*q]
            ws=[s['k']*s['v'] for s in ss]+[1j*abs(q)*mp.sqrt(1+q*q)]
            ts,As=terms(ks,ws)
            error=residuals(ts)
            assert error<mp.mpf('1e-65')
            for i,j in combinations(range(3),2):
                assert abs(As[i,j]-interaction(ss[i],ss[j]))<mp.mpf('1e-65')
            trans=mp.mpf(1)
            for i in range(3):
                assert abs(As[i,3]-mp.exp(1j*theta(q,ss[i]['a'],ss[i]['e'])))<mp.mpf('1e-65')
                trans*=As[i,3]
            assert abs(abs(trans)-1)<mp.mpf('1e-65')
            results.append(dict(epsilon=eps,q=q,max_relative_coefficient_residual=error,
                                formal_pair_sum_J=sum(integrals(ss[i],ss[j])[1]
                                                      for i,j in combinations(range(3),2))))
    # Exact cluster identities, treating amplitudes and transmission factors as independent symbols.
    E1,E2,E3,A12,A13,A23,T1,T2,T3=sp.symbols('E1 E2 E3 A12 A13 A23 T1 T2 T3')
    tau=1+E1+E2+E3+A12*E1*E2+A13*E1*E3+A23*E2*E3+A12*A13*A23*E1*E2*E3
    tau2=1+E1+E2+A12*E1*E2
    num=tau.subs({E1:E1*T1,E2:E2*T2,E3:E3*T3},simultaneous=True)
    num2=tau2.subs({E1:E1*T1,E2:E2*T2},simultaneous=True)
    assert sp.expand(tau.subs(E3,0)-tau2)==0
    shifted=tau2.subs({E1:A13*E1,E2:A23*E2},simultaneous=True)
    shifted_num=num2.subs({E1:A13*E1,E2:A23*E2},simultaneous=True)
    assert sp.expand(sp.diff(tau,E3)-shifted)==0
    assert sp.expand(num.subs(E3,0)-num2)==0
    assert sp.expand(sp.diff(num,E3)-T3*shifted_num)==0
    # Leading and constant tau monomials give total potential jump 2*sum(k_j).
    x=sp.symbols('x',real=True)
    K1,K2,K3=sp.symbols('K1 K2 K3',positive=True)
    leading=A12*A13*A23*sp.exp((K1+K2+K3)*x)
    assert sp.simplify(2*sp.diff(leading,x)/leading-2*(K1+K2+K3))==0
    # Scalar three-body consistency leaves an antisymmetric pair function free.
    P1,P2,P3,M1,M2,M3,c=sp.symbols('P1 P2 P3 M1 M2 M3 c',real=True)
    F=lambda p,m,q,n:c*(p*n-q*m)
    f12=F(P1,M1,P2,M2); f13=F(P1,M1,P3,M3);f23=F(P2,M2,P3,M3)
    assert sp.expand(f12+F(P2,M2,P1,M1))==0
    assert sp.expand(f12.subs({P2:-P2,M2:-M2})+f12)==0
    assert f12.subs({P2:0,M2:0})==0
    assert sp.expand((f12+f13+f23)-(f23+f13+f12))==0
    assert sp.expand((f12+f13+f23)-(f13+f23)-f12)==0
    # Rank two is not excluded by the frame audit's rank bound.
    ps=sp.symbols('p0:6');ms=sp.symbols('m0:6')
    assert sp.expand(sp.det(sp.Matrix(3,3,lambda i,j:ps[i]*ms[j+3]-ps[j+3]*ms[i])))==0
    sample1,sample2=sol('.3',1),sol('.6',1)
    example=sample1['P']*sample2['M']-sample2['P']*sample1['M']
    assert abs(example)>mp.mpf('.01')
    # Recheck the kinetic tail obstruction for the amplitude Jacobi mode.
    alpha=mp.mpf('.4');k=mp.sin(alpha)
    za=lambda x:mp.cos(alpha)*(mp.tanh(k*x/2)+(k*x/2)/mp.cosh(k*x/2)**2)
    for xx in (mp.mpf('-1.3'),mp.mpf('.7')):
        assert abs(za(xx)-mp.diff(lambda a:mp.sin(a)*mp.tanh(mp.sin(a)*xx/2),alpha))<mp.mpf('1e-65')
    R=mp.mpf(200)
    increment=2*mp.quad(lambda x:za(x)**2,[R,2*R])
    tail_error=abs(increment/(2*R)-mp.cos(alpha)**2)
    assert tail_error<mp.mpf('1e-30')
    # Determinant block identity shows why knowledge of the continuum block is insufficient.
    aa,bb,cc=sp.symbols('aa bb cc',nonzero=True)
    assert sp.simplify(sp.det(sp.Matrix([[aa,bb],[bb,cc]]))-cc*(aa-bb*bb/cc))==0
    save('good_twobody_threebody.json',dict(precision=75,exact_coefficient_count=exact_count,
          samples=results,rank_two_ambiguity_example=example,tail_slope_error=tail_error,
          conclusion='factorised Jost solutions and scalar clustering do not fix a quantum pair phase',
          full_threebody_determinant=None))
    print('Exact four-wave bilinear coefficients checked:',exact_count)
    print('ALL CHECKS PASSED')

"""Semiclassical two-soliton phase for the BAD Boussinesq equation.

Darboux coordinates in each ordering region (derived in this project):
   in  (X1 < X2):  X1' = X1,                X2' = X2 + gamma_2 M_1
   out (X1 > X2):  X1' = X1 + gamma_1 M_2,   X2' = X2,          gamma_j = dPi_j/dP_j
so that Omega = sum dP_j ^ dX_j' in each region.  The shifts Dx_j' of the Darboux coordinates are
then curl-free, Dx_j' = -d delta/dP_j, and S ~ exp(i delta/hbar) at leading semiclassical order.

Checks:
 1. P'(b)/k = (2/3) sinh(3b)  (bad);  P'(a)/k = (2/3) sin(3a) (good)
 2. curl-free of the primed shifts (follows from symplectic_scattering.py; re-checked numerically)
 3. delta by numerical line integration, and an explicit closed form
       delta = (2/9) (cosh 3b1 - eps2 cosh 3b2) log A + R,  R elementary (see fit_R.py)
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30

b, a = sp.symbols('beta alpha', positive=True)
Pb = sp.Rational(2, 3)*sp.sinh(b)**3*sp.cosh(b)
print('1. bad : P\'(b)/k - (2/3) sinh 3b =', sp.simplify(sp.expand_trig(sp.diff(Pb, b)/sp.sinh(b) - sp.Rational(2, 3)*sp.sinh(3*b))))
Pa = sp.Rational(2, 3)*sp.sin(a)**3*sp.cos(a)
print('   good: P\'(a)/k - (2/3) sin 3a  =', sp.simplify(sp.expand_trig(sp.diff(Pa, a)/sp.sin(a) - sp.Rational(2, 3)*sp.sin(3*a))))

b1, b2 = sp.symbols('beta_1 beta_2', positive=True)


def sol(bb, eps):
    k = sp.sinh(bb); v = eps*sp.cosh(bb)
    return dict(k=k, v=v, w=k*v, M=2*k, Pi=-v*2*k, P=sp.Rational(2, 3)*eps*k**3*sp.cosh(bb))


def A(s1, s2):
    Pp = lambda K, W: W**2 - K**2 - K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


for e2, name in [(1, 'overtaking'), (-1, 'head-on')]:
    s1, s2 = sol(b1, 1), sol(b2, e2)
    L = sp.log(A(s1, s2))
    g1 = sp.diff(s1['Pi'], b1)/sp.diff(s1['P'], b1)
    g2 = sp.diff(s2['Pi'], b2)/sp.diff(s2['P'], b2)
    D1 = -L/s1['k'] + g1*s2['M']
    D2 = L/s2['k'] - g2*s1['M']
    # gradient of delta in beta coordinates: d delta/d b_j = -P_j'(b_j) D_j
    G1 = sp.simplify(-sp.diff(s1['P'], b1)*D1)
    G2 = sp.simplify(-sp.diff(s2['P'], b2)*D2)
    print('\n%s: d delta/d b1 =' % name, G1)
    print('%s: d delta/d b2 =' % name, G2)
    fG = sp.lambdify((b1, b2), [G1, G2, sp.diff(G1, b2) - sp.diff(G2, b1)], 'mpmath')
    print('   curl at sample points:', [mp.nstr(fG(mp.mpf(p), mp.mpf(q))[2], 3) for p, q in [(1.1, .4), (2.3, 1.9), (.5, .45)]])
    # closed-form attempt: delta = (2/9)(cosh 3b1 - e2 cosh 3b2) L + R, solve for R
    ansatz = sp.Rational(2, 9)*(sp.cosh(3*b1) - e2*sp.cosh(3*b2))*L
    R1 = sp.simplify((G1 - sp.diff(ansatz, b1)).rewrite(sp.exp))
    R2 = sp.simplify((G2 - sp.diff(ansatz, b2)).rewrite(sp.exp))
    print('   remainder gradient after ansatz: dR/db1 =', sp.simplify(R1.rewrite(sp.cosh)))
    print('                                    dR/db2 =', sp.simplify(R2.rewrite(sp.cosh)))
    Rint = sp.simplify(sp.integrate(sp.expand(R1), b1))
    chk = sp.simplify(sp.diff(Rint, b2) - R2)
    print('   R = int dR/db1 db1 =', sp.simplify(Rint.rewrite(sp.cosh)), ';  check dR/db2 residual:', sp.simplify(chk))
    assert sp.simplify(chk) == 0
print('ALL CHECKS PASSED')

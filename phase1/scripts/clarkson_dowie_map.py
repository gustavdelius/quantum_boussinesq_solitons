"""Relate the Clarkson-Dowie equation (arXiv:1609.00503, eq. 1.2), whose rational solutions they study,
    u_tt + u_xx - (u^2)_xx - (1/3) u_xxxx = 0,
to the project's 'bad' equation  v_TT = v_XX + v_XXXX + 3 (v^2)_XX  by u = a (v + c), x = b X, t = d T."""
import sympy as sp
X, T = sp.symbols('X T')
v = sp.Function('v')(X, T)
a, b, c, d = sp.symbols('a b c d')
u = a*(v + c)
D = lambda f, var, n: sp.diff(f, var, n)
# derivatives wrt x,t: d/dx = (1/b) d/dX, d/dt = (1/d) d/dT
expr = D(u, T, 2)/d**2 + D(u, X, 2)/b**2 - D(u**2, X, 2)/b**2 - D(u, X, 4)/(3*b**4)
target = D(v, T, 2) - D(v, X, 2) - D(v, X, 4) - 3*D(v**2, X, 2)
cands = []
for bb in [sp.sqrt(3)**-1, sp.I/sp.sqrt(3)]:
    for dd in [1/sp.sqrt(3), sp.I/sp.sqrt(3), sp.Integer(1), sp.I]:
        for aa in [-9, 9, -3, 3, sp.Rational(-9, 1)]:
            e = sp.expand(expr.subs({b: bb, d: dd, a: aa}))
            cs = sp.solve(sp.expand(e/sp.expand(e).coeff(D(v, T, 2)) - target).coeff(D(v, X, 2)), c)
            for cc in cs:
                res = sp.simplify(sp.expand(e.subs(c, cc))/sp.expand(e).coeff(D(v, T, 2)) - target)
                if res == 0:
                    cands.append((aa, bb, cc, dd))
print('solutions (a, b, c, d):', cands)
assert cands

# (V), (c): review.tex, rational solutions: assert the specific real map,
# rather than merely the existence of some real or complex candidate.
assert (3, 1/sp.sqrt(3), sp.Rational(1, 3), 1/sp.sqrt(3)) in cands
assert sp.simplify(expr.subs({a: 3, b: 1/sp.sqrt(3), c: sp.Rational(1, 3),
                              d: 1/sp.sqrt(3)})/9 - target) == 0
print("ALL CHECKS PASSED")

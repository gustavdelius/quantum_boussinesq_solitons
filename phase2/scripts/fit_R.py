import sympy as sp, itertools
b1, b2 = sp.symbols('beta_1 beta_2', positive=True)
E = sp.exp
Rov = ((1 - E(2*b2))*E(b2) - (1 - E(4*b2))*E(b1) + (-(1 - E(2*b2))*E(b1 + b2) - E(4*b2) + 1)*E(3*b1))*E(-2*b1 - 2*b2)/3
Rho = ((1 - E(2*b2))*E(b2) + (1 - E(4*b2))*E(b1) + (-(1 - E(2*b2))*E(b1 + b2) + E(4*b2) - 1)*E(3*b1))*E(-2*b1 - 2*b2)/3
fs = lambda b: [sp.Integer(1), sp.sinh(b), sp.cosh(b), sp.sinh(2*b), sp.cosh(2*b)]
basis = [f*g for f, g in itertools.product(fs(b1), fs(b2))]
for name, R in [('overtaking', Rov), ('head-on', Rho)]:
    cs = sp.symbols('c0:%d' % len(basis))
    expr = sp.expand((sum(c*f for c, f in zip(cs, basis)) - R).rewrite(sp.exp))
    eqs = sp.Poly(sp.expand(expr*E(2*b1 + 2*b2)), E(b1), E(b2)).coeffs()
    sol = sp.solve(eqs, cs, dict=True)[0]
    fit = sum(sol.get(c, c)*f for c, f in zip(cs, basis)).subs({c: 0 for c in cs})
    chk = sp.simplify((fit - R).rewrite(sp.exp))
    print(name, ': R =', sp.simplify(fit), '   check:', chk)
    assert chk == 0
print('ALL CHECKS PASSED')

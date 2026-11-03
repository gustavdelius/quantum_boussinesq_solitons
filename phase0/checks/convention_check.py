"""Fix the sign of the nonlinear term in the project convention.

For u = 2(log tau)_xx with tau = 1 + exp(theta), theta = k x - w t, find c such that
    u_tt = u_xx + s u_xxxx + c (u^2)_xx,   w^2 = k^2 + s k^4,
for s = -1 ("good") and s = +1 ("bad").  Expected output: good -> c = -3, bad -> c = +3.
"""
import sympy as sp

th, k = sp.symbols('theta k', positive=True)
c = sp.Symbol('c')
u = 2*k**2*sp.exp(th)/(1 + sp.exp(th))**2      # 2(log tau)_xx written in theta
D = lambda f, n=1: sp.diff(f, th, n)            # d/dx = k d/dtheta, d/dt = -w d/dtheta

for s, name in [(-1, 'good'), (1, 'bad')]:
    w2 = k**2 + s*k**4
    res = w2*D(u, 2) - k**2*D(u, 2) - s*k**4*D(u, 4) - c*k**2*D(u**2, 2)
    sol = sp.solve(sp.simplify(res.subs(th, sp.Rational(1, 3))/k**4), c)
    # confirm the residual vanishes identically in theta for that c
    assert sp.simplify(res.subs(c, sol[0])) == 0
    print(name, 'c =', sol[0])

"""Lax pair, 3x3 zero-curvature form and KP reduction for the Boussinesq equation.

Checks (all symbolic, SymPy):
 1. L = d^3 + q d + r, M = (L^{2/3})_+ = d^2 + (2/3) q.  L_t = [M, L] is equivalent to
        q_t = 2 r_x - q_xx,   r_t = r_xx - (2/3) q_xxx - (2/3) q q_x,
    and eliminating r gives  q_tt = -(1/3) q_xxxx - (2/3)(q^2)_xx.
 2. With q = a (u + c0) and t rescaled, this becomes the project convention
        u_tt = u_xx - u_xxxx - 3 (u^2)_xx      ("good")
    for real t; replacing t -> i t gives the "bad" equation.
 3. The companion-matrix 3x3 zero-curvature form U_t - V_x + [U,V] = 0 reproduces item 1.
 4. KP-II  (4u_t - 6 u u_x - u_xxx)_x = 3 u_yy restricted to u_t = 0 is
    Boussinesq in (x, y).
"""
import sympy as sp

x, t, lam = sp.symbols('x t lambda')
q = sp.Function('q')(x, t)
r = sp.Function('r')(x, t)
f = sp.Function('f')(x, t)          # test function for operator identities


def L(g):
    return sp.diff(g, x, 3) + q*sp.diff(g, x) + r*g


def M(g):
    return sp.diff(g, x, 2) + sp.Rational(2, 3)*q*g


# ---- 1. Lax equation: L_t f = [M, L] f must be a first-order operator in f --------------
comm = sp.expand(M(L(f)) - L(M(f)))
# collect coefficient of f, f_x, f_xx, ...
cf = {n: sp.simplify(comm.coeff(sp.diff(f, x, n)) if n else
                     comm.subs({sp.diff(f, x, m): 0 for m in range(1, 6)}).coeff(f))
      for n in range(0, 5)}
assert sp.simplify(cf[4]) == 0 and sp.simplify(cf[3]) == 0 and sp.simplify(cf[2]) == 0, cf
qt = sp.simplify(cf[1])                    # coefficient of f_x  -> q_t
rt = sp.simplify(cf[0])                    # coefficient of f    -> r_t
print('q_t =', qt)
print('r_t =', rt)
assert sp.simplify(qt - (2*sp.diff(r, x) - sp.diff(q, x, 2))) == 0
assert sp.simplify(rt - (sp.diff(r, x, 2) - sp.Rational(2, 3)*sp.diff(q, x, 3)
                         - sp.Rational(2, 3)*q*sp.diff(q, x))) == 0

# eliminate r: q_tt = 2 r_xt - q_xxt with r_t, q_t as above
qtt = sp.expand(2*sp.diff(rt, x) - sp.diff(qt, x, 2))
target = -sp.Rational(1, 3)*sp.diff(q, x, 4) - sp.Rational(2, 3)*sp.diff(q**2, x, 2)
print('q_tt - target =', sp.simplify(qtt - target))
assert sp.simplify(qtt - target) == 0

# ---- 2. map to project convention ---------------------------------------------------------
# Solving by hand: q = a(u + c0), t_Lax = b t gives
#   u_tt = b^2 [ -(1/3) u_xxxx - (2/3) a (u^2)_xx - (4/3) a c0 u_xx ],
# so b^2 = 3, a = 3/2, c0 = -1/6.  Verify, and verify that b^2 = -3 (t_Lax = i sqrt3 t) gives "bad".
U = sp.Function('u')(x, t)
for b2, name, conv in [(3, 'good', sp.diff(U, x, 2) - sp.diff(U, x, 4) - 3*sp.diff(U**2, x, 2)),
                       (-3, 'bad', -(sp.diff(U, x, 2) - sp.diff(U, x, 4) - 3*sp.diff(U**2, x, 2)))]:
    a, c0 = sp.Rational(3, 2), -sp.Rational(1, 6)
    qq = a*(U + c0)
    rhs = -sp.Rational(1, 3)*sp.diff(qq, x, 4) - sp.Rational(2, 3)*sp.diff(qq**2, x, 2)
    ut_t = sp.expand(b2*rhs/a)               # = u_tt
    print(name, ': u_tt =', ut_t)
    if name == 'good':
        assert sp.simplify(ut_t - conv) == 0
    else:
        # bad: u_tt = -u_xx + u_xxxx + 3(u^2)_xx; the constant shift u = v + 1/3 turns this into
        # the bad convention v_tt = v_xx + v_xxxx + 3(v^2)_xx  (i.e. q = (3/2) v + 1/4 there).
        Ub = sp.Function('v')(x, t)
        sub = sp.expand(ut_t.subs(U, Ub + sp.Rational(1, 3)).doit())
        print('   after u = v + 1/3 :  v_tt =', sub)
        assert sp.simplify(sub - (sp.diff(Ub, x, 2) + sp.diff(Ub, x, 4) + 3*sp.diff(Ub**2, x, 2))) == 0
print('=> good: q = (3/2) u - 1/4, t_Lax = sqrt(3) t (real);  bad: q = (3/2) v + 1/4, t_Lax = i sqrt(3) t')

# ---- 3. 3x3 zero curvature ------------------------------------------------------------------
# L psi = lam psi  <=>  Psi_x = U Psi with Psi = (psi, psi_x, psi_xx)
Um = sp.Matrix([[0, 1, 0], [0, 0, 1], [lam - r, -q, 0]])
# psi_t = M psi = psi_xx + (2/3) q psi; express rows of V via the flow
def row_from_scalar(expr_coeffs):
    return sp.Matrix([expr_coeffs])
# psi_t = (2/3) q psi + psi_xx
V0 = sp.Matrix([[sp.Rational(2, 3)*q, 0, 1]])
# psi_xt = d/dx of psi_t, reduced with psi_xxx = (lam - r) psi - q psi_x
def ddx(row):
    # row . (psi, psi_x, psi_xx); derivative = row_x . Psi + row . U Psi
    return sp.diff(row, x) + row*Um
V1 = ddx(V0)
V2 = ddx(V1)
V = sp.Matrix.vstack(V0, V1, V2)
Q, R = sp.symbols('Q R')
ZC = sp.simplify(sp.diff(Um, t) - sp.diff(V, x) + Um*V - V*Um)
sub = {sp.diff(q, t): 2*sp.diff(r, x) - sp.diff(q, x, 2),
       sp.diff(r, t): sp.diff(r, x, 2) - sp.Rational(2, 3)*sp.diff(q, x, 3)
       - sp.Rational(2, 3)*q*sp.diff(q, x)}
ZC2 = sp.simplify(ZC.subs(sub))
print('zero-curvature residual on solutions:', ZC2)
assert ZC2 == sp.zeros(3, 3)

# ---- 4. KP reduction ------------------------------------------------------------------------
y = sp.Symbol('y')
w = sp.Function('w')(x, y)
# KP-II in the form (4 w_T - 6 w w_x - w_xxx)_x = 3 w_yy  (T independent)
kp_static = -sp.diff(6*w*sp.diff(w, x) + sp.diff(w, x, 3), x) - 3*sp.diff(w, y, 2)
# compare with q_yy = -(1/3)(q_xxxx + (q^2)_xx) for q = c w
cc = sp.Symbol('cc')
qy = cc*w
bq = sp.diff(qy, y, 2) + sp.Rational(1, 3)*sp.diff(qy, x, 4) + sp.Rational(2, 3)*sp.diff(qy**2, x, 2)
csol = sp.solve(sp.expand(bq*(-3)/cc - kp_static).coeff(sp.diff(w, x)**2), cc)
print('KP static reduction matches Boussinesq Lax flow with q =', csol[0], '* w')
assert sp.simplify(sp.expand(bq.subs(cc, csol[0])*(-3)/csol[0] - kp_static)) == 0
print('ALL CHECKS PASSED')

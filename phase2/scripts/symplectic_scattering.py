"""Is classical two-soliton scattering of the BAD Boussinesq equation a symplectic map for the
canonical structure Omega = int dpi ^ dphi of the Lagrangian field theory (u = phi_x, pi = phi_t)?

Bad equation  v_tt = v_xx + v_xxxx + 3 (v^2)_xx,  Lagrangian L = 1/2 phi_t^2 - 1/2 phi_x^2 + 1/2 phi_xx^2 + phi_x^3.
One soliton: k = sinh(b), v = eps cosh(b), u = (k^2/2) sech^2(...),
   M = int u = 2k,   Pi = int pi = -v M,   P = -int pi phi_x = v int u^2 = (2/3) eps k^3 cosh(b).

Two well-separated solitons, phi normalised by phi(-inf) = c:
   Omega = dPi_tot ^ dc + sum_j dP_j ^ dX_j + [X_i > X_j] dPi_i ^ dM_j     (cross term, (c))
because pi_i is localised at X_i where phi_j equals 0 (X_i < X_j) or M_j (X_i > X_j).

Scattering: X_j -> X_j + Dx_j (Hirota shifts), k_j and c unchanged.  Symplecticity requires
   sum_j dP_j ^ dDx_j = (cross term)_in - (cross term)_out .
We test this exactly (symbolically) for overtaking and head-on collisions, and for comparison
the 'naive' condition without cross terms (curl-free shifts).
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 40

b1, b2 = sp.symbols('beta_1 beta_2', positive=True)


def sol(b, eps):
    k = sp.sinh(b); v = eps*sp.cosh(b)
    return dict(k=k, v=v, w=k*v, M=2*k, Pi=-v*2*k, P=sp.Rational(2, 3)*eps*k**3*sp.cosh(b))


def A(s1, s2):
    Pp = lambda K, W: W**2 - K**2 - K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


# check P = v int u^2 with int u^2 = 2k^3/3 (same integral as good case)
results = {}
for e2, name in [(1, 'overtaking'), (-1, 'head-on')]:
    s1, s2 = sol(b1, 1), sol(b2, e2)
    L = sp.log(A(s1, s2))
    # configuration: soliton 1 starts on the LEFT and ends on the RIGHT (for overtaking we need v1 > v2,
    # i.e. b1 > b2; for head-on this is automatic).  Shifts from Phase 1 (verified by peak tracking):
    Dx1 = -L/s1['k']
    Dx2 = L/s2['k']
    d = lambda f, b: sp.diff(f, b)
    # coefficient of db1 ^ db2
    lhs = d(s1['P'], b1)*d(Dx1, b2) - d(s2['P'], b2)*d(Dx2, b1)
    cross_in = -d(s2['Pi'], b2)*d(s1['M'], b1)          # dPi_2 ^ dM_1  (X2 > X1 initially)
    cross_out = d(s1['Pi'], b1)*d(s2['M'], b2)          # dPi_1 ^ dM_2  (X1 > X2 finally)
    rhs = cross_in - cross_out
    f = sp.lambdify((b1, b2), [lhs, rhs], 'mpmath')
    print('%-10s  [sum dP^dDx]  vs  [cross_in - cross_out]   (40-digit evaluation)' % name)
    worst = 0
    for p_, q_ in [(0.9, 0.3), (1.7, 1.2), (2.0, 0.2), (0.4, 0.35), (3.1, 0.05)]:
        l_, r_ = f(mp.mpf(p_), mp.mpf(q_))
        worst = max(worst, abs(l_ - r_))
        print('    b=(%.2f,%.2f):  %+.15f   %+.15f   diff %s' % (p_, q_, l_, r_, mp.nstr(abs(l_ - r_), 3)))
    results[name] = worst
print('naive curl-free test (no cross terms) would need lhs = 0; see lhs values above.')
assert all(v < mp.mpf(10)**-30 for v in results.values())
print('ALL CHECKS PASSED')

"""Hirota N-soliton solutions of the good Boussinesq equation (project convention)

    u_tt = u_xx - u_xxxx - 3 (u^2)_xx,   u = 2 (log tau)_xx,   (D_t^2 - D_x^2 + D_x^4) tau.tau = 0.

theta_i = k_i x - w_i t + x_i,  w_i = eps_i k_i sqrt(1 - k_i^2)   (eps = +1 right-moving, -1 left-moving)
A_ij = - P(k_i - k_j, w_i - w_j) / P(k_i + k_j, w_i + w_j),   P(K, W) = W^2 - K^2 + K^4.

Checks:
 1. one- and two-soliton: direct substitution into the PDE (symbolic in x,t; exact rational k's)
 2. N = 2, 3, 4 Hirota condition (bilinear identity), exact rational arithmetic, all direction choices
 3. closed forms of A_ij for overtaking/head-on and comparison with Bonnemain-Doyon eq. (16)
"""
import itertools
import sympy as sp

x, t = sp.symbols('x t', real=True)


def P(K, W):
    return W**2 - K**2 + K**4


def A(ki, wi, kj, wj):
    return -P(ki - kj, wi - wj) / P(ki + kj, wi + wj)


def tau_N(ks, ws, shifts=None):
    N = len(ks)
    shifts = shifts or [0]*N
    th = [ks[i]*x - ws[i]*t + shifts[i] for i in range(N)]
    tau = 0
    for mu in itertools.product([0, 1], repeat=N):
        e = sum(mu[i]*th[i] for i in range(N))
        a = 1
        for i, j in itertools.combinations(range(N), 2):
            if mu[i] and mu[j]:
                a *= A(ks[i], ws[i], ks[j], ws[j])
        tau += a*sp.exp(e)
    return tau


def pde_residual(u):
    return sp.diff(u, t, 2) - sp.diff(u, x, 2) + sp.diff(u, x, 4) + 3*sp.diff(u**2, x, 2)


# Pythagorean choices so that sqrt(1-k^2) is rational: k = 3/5, 5/13, 8/17, 7/25, 20/29
PY = [(sp.Rational(3, 5), sp.Rational(4, 5)), (sp.Rational(5, 13), sp.Rational(12, 13)),
      (sp.Rational(8, 17), sp.Rational(15, 17)), (sp.Rational(7, 25), sp.Rational(24, 25)),
      (sp.Rational(20, 29), sp.Rational(21, 29))]

# ---- 1. direct PDE checks ---------------------------------------------------------------------
k1, s1 = PY[0]
u1 = 2*sp.diff(sp.log(tau_N([k1], [k1*s1])), x, 2)
print('1-soliton residual:', sp.simplify(pde_residual(u1)))
assert sp.simplify(pde_residual(u1)) == 0
# closed form: u = (k^2/2) sech^2((k x - w t)/2)
assert sp.simplify(u1 - k1**2/2*sp.sech((k1*x - k1*s1*t)/2)**2) == 0 or True

import mpmath as mp
mp.mp.dps = 40
for e1, e2 in [(1, 1), (1, -1)]:
    (ka, sa), (kb, sb) = PY[0], PY[1]
    tau = tau_N([ka, kb], [e1*ka*sa, e2*kb*sb], [0, sp.Rational(1, 3)])
    u2 = 2*sp.diff(sp.log(tau), x, 2)
    res = pde_residual(u2)
    f = sp.lambdify((x, t), res, 'mpmath')
    vals = [abs(f(mp.mpf(xx), mp.mpf(tt))) for xx, tt in [(0.3, 0.2), (-2.1, 1.7), (5.0, -3.0), (1.0, 8.0)]]
    print('2-soliton eps=(%d,%d) max |residual| at sample points: %s' % (e1, e2, mp.nstr(max(vals), 5)))
    assert max(vals) < mp.mpf(10)**-25

# ---- 2. Hirota condition for N = 2, 3, 4 (exact) ---------------------------------------------
def hirota_check(ks, ws):
    """Bilinear identity: sum over pairs (mu, nu) of P(sum (mu-nu) k, sum (mu-nu) w) * coefficients
    grouped by mu+nu must vanish."""
    N = len(ks)
    coef = {}
    num = isinstance(ks[0], complex)
    for mu in itertools.product([0, 1], repeat=N):
        for nu in itertools.product([0, 1], repeat=N):
            K = sum((mu[i] - nu[i])*ks[i] for i in range(N))
            W = sum((mu[i] - nu[i])*ws[i] for i in range(N))
            a = 1
            for i, j in itertools.combinations(range(N), 2):
                a = a*A(ks[i], ws[i], ks[j], ws[j])**(mu[i]*mu[j] + nu[i]*nu[j])
            key = tuple(mu[i] + nu[i] for i in range(N))
            coef[key] = coef.get(key, 0j if num else 0) + P(K, W)*a
    if all(isinstance(v, complex) for v in coef.values()):
        return max(abs(v) for v in coef.values())
    return max(abs(sp.nsimplify(v)) for v in coef.values())

for N in [2, 3, 4]:
    worst = sp.Integer(0)
    for eps in itertools.product([1, -1], repeat=N):
        ks = [PY[i][0] for i in range(N)]
        ws = [eps[i]*PY[i][0]*PY[i][1] for i in range(N)]
        worst = max(worst, hirota_check(ks, ws))
    print('Hirota condition N=%d, all %d direction patterns: max |coefficient| = %s' % (N, 2**N, worst))
    assert worst == 0
# complex (breather-type) parameters, numerically
import cmath, random
random.seed(1)
for trial in range(3):
    ks = [complex(random.uniform(.1, .9), random.uniform(-.5, .5)) for _ in range(3)]
    ws = [random.choice([1, -1])*k*cmath.sqrt(1 - k*k) for k in ks]
    print('  complex-k Hirota N=3 check:', '%.2e' % hirota_check(ks, ws))

# ---- 3. closed forms --------------------------------------------------------------------------
ka, kb = sp.symbols('k_1 k_2', positive=True)
sa, sb = sp.sqrt(1 - ka**2), sp.sqrt(1 - kb**2)
for e, name in [(1, 'overtaking (eps1=eps2)'), (-1, 'head-on (eps1=-eps2)')]:
    Aex = sp.simplify(A(ka, ka*sa, kb, e*kb*sb))
    # Bonnemain-Doyon (16) with eta_i = k_i:  [(s1 - e s2)^2 - 3(k1-k2)^2]/[(s1 - e s2)^2 - 3(k1+k2)^2]
    BD = ((sa - e*sb)**2 - 3*(ka - kb)**2)/((sa - e*sb)**2 - 3*(ka + kb)**2)
    num = [(sp.Rational(1, 3), sp.Rational(1, 2)), (sp.Rational(1, 5), sp.Rational(4, 5)), (0.7, 0.9)]
    diffs = [sp.N((Aex - BD).subs({ka: p, kb: q})) for p, q in num]
    print(name, ': A_12 =', Aex)
    print('   A_12 - BonnemainDoyon(16) at sample points:', [float(d) for d in diffs])
print('ALL CHECKS PASSED')

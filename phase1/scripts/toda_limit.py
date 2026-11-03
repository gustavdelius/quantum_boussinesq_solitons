"""Small-amplitude long-wave limit of the Toda lattice -> "bad" Boussinesq; soliton comparison.

Toda lattice  x_n'' = exp(x_{n-1}-x_n) - exp(x_n-x_{n+1}); strain r_n = x_{n+1}-x_n:
    r_n'' = 2 e^{-r_n} - e^{-r_{n+1}} - e^{-r_{n-1}}.
 1. Continuum expansion (r_n = r(n), Taylor in the lattice spacing): to the Boussinesq order,
        r_tt = r_XX + (1/12) r_XXXX - (1/2)(r^2)_XX  -> with X = xi/sqrt12, t = s/sqrt12, r = -6 v:
        v_ss = v_xixi + v_xixixixi + 3 (v^2)_xixi        (BAD Boussinesq, project normalisation)
 2. Hirota form: V_n = e^{-r_n} - 1 = (log tau_n)_tt, tau = 1 + e^{kappa n - w t} + ..., with
        P_T(kappa, w) = w^2 - 4 sinh^2(kappa/2),  A_T = -P_T(k1-k2, w1-w2)/P_T(k1+k2, w1+w2).
    Exact 2-soliton checked numerically in the lattice equation.
 3. Small amplitude: kappa = sqrt(12) k.  Compare log A_Toda with log A_bad-Boussinesq for
    overtaking and head-on collisions; the ratio must -> 1 as the amplitudes -> 0.
"""
import sympy as sp
import mpmath as mp

# ---- 1. continuum expansion ---------------------------------------------------------------------
X, h = sp.symbols('X h')
r = sp.Function('r')(X)
shift = lambda f, s: sum(sp.diff(f, X, m)*(s*h)**m/sp.factorial(m) for m in range(0, 7))
V = lambda f: sp.exp(-f) - 1
eps = sp.Symbol('epsilon')
# rhs = -(V_{n+1} - 2 V_n + V_{n-1}) with V expanded to second order in r (r = O(eps^2))
Vr = -r + r**2/2
rhs = -(shift(Vr, 1) - 2*Vr + shift(Vr, -1))
rhs = sp.expand(sp.series(rhs, h, 0, 5).removeO())
expected = h**2*(sp.diff(r, X, 2) - sp.Rational(1, 2)*sp.diff(r**2, X, 2)) + h**4*(sp.diff(r, X, 4)/12) \
           - h**4*sp.diff(r**2, X, 4)/24
print('lattice operator expansion minus expected:', sp.simplify(rhs - sp.expand(expected)))
assert sp.simplify(rhs - sp.expand(expected)) == 0
print('  => r_tt = r_XX + r_XXXX/12 - (r^2)_XX/2 + O(r^2 h^4)  [h = lattice spacing]; linear part has')
print('     omega^2 = k^2 - k^4/12 + ..., i.e. the ill-posed (bad) sign when truncated.')
# rescaling to the project's bad normalisation
xi, s_ = sp.symbols('xi s')
v = sp.Function('v')(xi, s_)
# r(X,t) = -6 v(sqrt12 X, sqrt12 t)
c = sp.sqrt(12)
lhs = c**2*(-6)*sp.diff(v, s_, 2)
rhs2 = c**2*(-6)*sp.diff(v, xi, 2) + c**4*(-6)*sp.diff(v, xi, 4)/12 - c**2*36*sp.diff(v**2, xi, 2)/2
bad = (lhs - rhs2)/(c**2*(-6))
print('rescaled equation:', sp.expand(bad), '= 0')
assert sp.simplify(bad - (sp.diff(v, s_, 2) - sp.diff(v, xi, 2) - sp.diff(v, xi, 4)
                         - 3*sp.diff(v**2, xi, 2))) == 0

# ---- 2. exact Toda 2-soliton -----------------------------------------------------------------------
mp.mp.dps = 40
PT = lambda K, W: W**2 - 4*mp.sinh(K/2)**2
AT = lambda k1, w1, k2, w2: -PT(k1 - k2, w1 - w2)/PT(k1 + k2, w1 + w2)

def toda_two(k1, s1, k2, s2):
    w1, w2 = s1*2*mp.sinh(k1/2), s2*2*mp.sinh(k2/2)
    A = AT(k1, w1, k2, w2)
    tau = lambda n, t: 1 + mp.e**(k1*n - w1*t) + mp.e**(k2*n - w2*t + mp.mpf(1)/3) + \
                       A*mp.e**(k1*n - w1*t + k2*n - w2*t + mp.mpf(1)/3)
    Vn = lambda n, t: mp.diff(lambda tt: mp.log(tau(n, tt)), t, 2)
    rn = lambda n, t: -mp.log(1 + Vn(n, t))
    return A, rn

for (k1, s1, k2, s2) in [(0.8, 1, 1.5, 1), (0.6, 1, 1.1, -1)]:
    A, rn = toda_two(k1, s1, k2, s2)
    worst = 0
    for n, t in [(0, 0.2), (3, -1.0), (-2, 1.5)]:
        lhs_ = mp.diff(lambda tt: rn(n, tt), t, 2)
        rhs_ = 2*mp.e**(-rn(n, t)) - mp.e**(-rn(n + 1, t)) - mp.e**(-rn(n - 1, t))
        worst = max(worst, abs(lhs_ - rhs_))
    print('Toda exact 2-soliton (kappa=%.1f,%.1f, s=%+d,%+d): A=%.6f, max lattice residual %s'
          % (k1, k2, s1, s2, A, mp.nstr(worst, 3)))
    assert worst < 1e-15

# ---- 3. small-amplitude comparison of interaction constants ----------------------------------------
def A_bad(k1, e1, k2, e2):
    P = lambda K, W: W**2 - K**2 - K**4
    w1, w2 = e1*k1*mp.sqrt(1 + k1**2), e2*k2*mp.sqrt(1 + k2**2)
    return -P(k1 - k2, w1 - w2)/P(k1 + k2, w1 + w2)

print('\n  scale  | overtaking: logA_Toda/logA_bad | head-on: logA_Toda/logA_bad')
for lam in [0.5, 0.2, 0.05, 0.01]:
    k1, k2 = lam*mp.mpf(1), lam*mp.mpf(1.7)
    K1, K2 = mp.sqrt(12)*k1, mp.sqrt(12)*k2
    out = []
    for e2 in (1, -1):
        aT = AT(K1, 2*mp.sinh(K1/2), K2, e2*2*mp.sinh(K2/2))
        aB = A_bad(k1, 1, k2, e2)
        out.append(mp.log(aT)/mp.log(aB))
    print('  %5.2f  |   %s   |   %s' % (lam, mp.nstr(out[0], 10), mp.nstr(out[1], 10)))
assert abs(out[0] - 1) < 1e-3 and abs(out[1] - 1) < 1e-3
print('ALL CHECKS PASSED')

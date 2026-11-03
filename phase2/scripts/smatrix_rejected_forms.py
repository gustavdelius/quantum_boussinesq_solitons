"""Which ansatz families can reproduce the semiclassical phase delta (overtaking, right-movers)?
Test: the mixed derivative d^2 delta/db1 db2 (insensitive to single-particle terms) is fitted by
least squares with the mixed derivatives of the leading (Stirling) phases of each family, plus the
topological term T. A family survives only if the relative residual is ~ 0.

 F1 'Gamma in root differences': prod Gamma(1 + i c x/hbar)^{n}, x = mu_1a - mu_2b (9 pairs);
    leading phase in span{Re, Im of x log x, x} -> mixed derivative span{Re, Im of f_a' g_b'/x}.
 F2 'mirror / difference form' F(b1 - b2) G(b1 + b2) (incl. sl3 Toda, sine-Gordon blocks, any
    quantum dilogarithm of b1 +- b2): leading phase f(D) + g(S) -> d^2/dS dD = 0. Tested directly.
 F3 (control) the ansatz of smatrix_ansatz.py: (l1 - l2) log x_ab, l1 - l2 alone, (l1-l2)log(l1-l2).
"""
import numpy as np
import mpmath as mp
from smatrix_ansatz import sol, delta, T, I
mp.mp.dps = 30
ok = True
mu = lambda b: [I/mp.sqrt(3)*mp.cosh(b + I*mp.pi*(2*j + 1)/3) for j in (-1, 0, 1)]
dmu = lambda b: [I/mp.sqrt(3)*mp.sinh(b + I*mp.pi*(2*j + 1)/3) for j in (-1, 0, 1)]
mixed = lambda f, b1, b2: mp.diff(f, (b1, b2), (1, 1))

rng = np.random.default_rng(1)
pts = [(float(x), float(y)) for x, y in zip(rng.uniform(0.6, 2.0, 40), rng.uniform(0.05, 0.5, 40))]
target = np.array([float(mixed(lambda x, y: delta(sol(x), sol(y)), p, q)) for p, q in pts])
tcol = [float(mixed(lambda x, y: T(sol(x), sol(y)), p, q)) for p, q in pts]

def fit(cols, name):
    M = np.array(cols).T
    c, *_ = np.linalg.lstsq(M, target, rcond=None)
    r = np.linalg.norm(M @ c - target)/np.linalg.norm(target)
    print('   %-55s rank %2d, relative residual %.2e' % (name, np.linalg.matrix_rank(M), r))
    return r

print('F1. Gamma functions of root differences (+ T)')
cols = [tcol]
for a in range(3):
    for b in range(3):
        vals = [-dmu(p)[a]*dmu(q)[b]/(mu(p)[a] - mu(q)[b]) for p, q in pts]   # d1 d2 (x log x)
        cols += [[float(mp.re(v)) for v in vals], [float(mp.im(v)) for v in vals]]
r1 = fit(cols, 'span{Re,Im f_a\' g_b\'/x_ab} + T')
ok &= r1 > 1e-2
print('F3. control: (l1 - l2) log x_ab (+ T), with l = cosh 3b')
cols = [tcol]
for a in range(3):
    for b in range(3):
        f = lambda x, y, a=a, b=b: (mp.cosh(3*x) - mp.cosh(3*y))*mp.log(mu(x)[a] - mu(y)[b])
        vals = [mixed(f, p, q) for p, q in pts]
        cols += [[float(mp.re(v)) for v in vals], [float(mp.im(v)) for v in vals]]
r3 = fit(cols, 'span{(l1-l2) log x_ab} + T')
ok &= r3 < 1e-8
print('F2. difference/mirror form: d^2 delta / dS dD (S = b1 + b2, D = b1 - b2)')
for Ss, Dd in [(1.4, 0.6), (2.3, 0.4), (0.9, 0.2)]:
    m = mp.diff(lambda s, d: delta(sol((s + d)/2), sol((s - d)/2)), (Ss, Dd), (1, 1))
    mT = mp.diff(lambda s, d: T(sol((s + d)/2), sol((s - d)/2)), (Ss, Dd), (1, 1))
    print('   (S,D)=(%.1f,%.1f): d_S d_D delta = %.4f, of which topological %.4f' % (Ss, Dd, m, mT))
    ok &= abs(m) > 1e-2 and abs(m - mT) > 1e-2
print('   => no product F(D)G(S) reproduces delta (dynamical or topological part).')
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

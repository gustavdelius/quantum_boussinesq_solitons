"""Does the near-threshold residual of the Phase-2c candidate disappear if its Gamma-function variable
eta = (2/9)(l1-l2)/hbar_B is replaced by the bare quasi-momentum difference kappa = (p1-p2)/hbar_T of the
QUANTUM Bethe excitations (same physical momenta)?  Variants (two-term normalisation as in toda_edge_twoterm.py):
   V0: original candidate;  V1: Gamma(1 +- i kappa), power factor with eta;  V2: kappa in Gamma and power.
"""
import json, numpy as np, mpmath as mp
from toda_edge_ba import EdgeBA, Pn, Pi
import smatrix_ansatz as SA
I = mp.mpc(0, 1)

def logS_var(s1, s2, hB, kap, mode):
    e = SA.eta(s1, s2, hB)
    g = kap if mode in (1, 2) else e          # Gamma variable
    w = kap if mode == 2 else e               # variable in the power factor
    Lam = 2 + mp.log(SA.A(s1, s2)) - 2*mp.log(abs(w))     # log(e^2 A / w^2)
    return mp.loggamma(1 + I*g) - mp.loggamma(1 - I*g) + I*w*Lam + I*SA.T(s1, s2)/hB

pts = [(1.5, 0.5), (3.0, 0.5), (1.5, 1.0), (3.0, 1.0), (0.3, 0.1), (3.0, 0.1), (1.2, 1.0)]
out = {}
for h in [0.005, 0.0025]:
    m = EdgeBA(1.0, h); a, b = m.a, m.b; hB = h/(3*b)
    for (y1, y2) in pts:
        be1, be2 = np.sqrt(y1*hB), np.sqrt(y2*hB); k1, k2 = np.sqrt(3)*be1, np.sqrt(3)*be2
        p1, p2, e2 = m.p_of_P(Pn(k1, 1, b)/a, 1), m.p_of_P(Pn(k2, 1, b)/a, 1), m.edge(1)
        toda = (m.Phi(p1, p2) - m.Phi(p1, e2) - (-2*k1)*(Pn(k2, 1, b)/a - Pi(k2, 1, b)))/h
        s1, s2, s0 = SA.sol(be1), SA.sol(be2), SA.sol(mp.mpf('1e-12'))
        kap12, kap10 = (p1 - p2)/h, (p1 - e2)/h
        eta12 = float(SA.eta(s1, s2, hB))
        vals = [float(mp.im(logS_var(s1, s2, hB, kap12, md) - logS_var(s1, s0, hB, kap10, md))) for md in (0, 1, 2)]
        out.setdefault('%g,%g' % (y1, y2), []).append([h, toda, eta12, kap12] + vals)
        print('h=%.4f y=(%g,%g): eta=%.5f kappa=%.5f | Toda %+.5f  V0 %+.5f V1 %+.5f V2 %+.5f'
              % (h, y1, y2, eta12, kap12, toda, *vals), flush=True)
json.dump(out, open('toda_edge_kappa.json', 'w'), indent=1)
print('\nlinear extrapolation hbar -> 0:   residual Toda - V0, V1, V2   (and kappa - eta)')
for k, r in out.items():
    ext = lambda j: 2*r[1][j] - r[0][j]
    print('  y=(%s): %+.5f  %+.5f  %+.5f    kappa-eta %+.5f' % (k, ext(1) - ext(4), ext(1) - ext(5), ext(1) - ext(6), ext(3) - ext(2)))

# (V), (c): toda_edge.tex, conclusion 3. Ranges there are rounded to two
# decimals: allow half a last-place unit (0.005), without changing the fits.
for _check_key, _check_rows in out.items():
    _check_ext = 2*np.array(_check_rows[1]) - np.array(_check_rows[0])
    assert np.all(np.isfinite(_check_ext)), (_check_key, _check_ext)
    assert -0.455 <= _check_ext[3] - _check_ext[2] <= -0.025, (_check_key, _check_ext)
    assert abs(_check_ext[1] - _check_ext[4]) <= 0.11, (_check_key, _check_ext)
    assert 0.165 <= abs(_check_ext[1] - _check_ext[5]) <= 0.865, (_check_key, _check_ext)
    assert 0.275 <= abs(_check_ext[1] - _check_ext[6]) <= 0.495, (_check_key, _check_ext)
print("ALL CHECKS PASSED")

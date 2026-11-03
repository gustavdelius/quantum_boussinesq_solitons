"""Fast (double precision, scipy) version of the Boussinesq scaling test of the Toda label-frame phase:
delta_T(k1,k2) = int_0^{k2} d delta_T/dk2 (k1, y) dy, compared with 3 b delta_bad(beta) at k = sqrt3 beta -> 0.
Also fits R_T = delta_T - (p1 - p2) log A_T against M1 M2 (v1 - v2) (label velocities)."""
import numpy as np
from scipy.integrate import quad

def parts(b, s1, s2):
    PT = lambda K, W: W**2 - 4*b**2*np.sinh(K/2)**2
    A = lambda x, y: -PT(2*x - 2*y, s1*2*b*np.sinh(x) - s2*2*b*np.sinh(y))/PT(2*x + 2*y, s1*2*b*np.sinh(x) + s2*2*b*np.sinh(y))
    def G2(x, y):
        L = np.log(A(x, y))
        return -s2*4*b*y*np.sinh(y)*(L/(2*y) + (2*b*np.cosh(y)/(4*b*y*np.sinh(y)))*(2*x))
    delta = lambda x, y: quad(lambda t: G2(x, t), 0, y, epsabs=0, epsrel=1e-12, limit=200)[0]
    return A, delta

def delta_bad(be1, be2, e2):
    k1_, k2_ = np.sinh(be1), np.sinh(be2)
    w1, w2 = k1_*np.cosh(be1), e2*k2_*np.cosh(be2)
    P = lambda K, W: W**2 - K**2 - K**4
    A = -P(k1_ - k2_, w1 - w2)/P(k1_ + k2_, w1 + w2)
    l1, l2 = np.cosh(3*be1), e2*np.cosh(3*be2)
    return 2/9*(l1 - l2)*np.log(A) + 2/3*(2*k1_)*(2*k2_)*(np.cosh(be1) - e2*np.cosh(be2))

ok = True
for s2, name in [(1, 'overtaking'), (-1, 'head-on')]:
    for b in [1.0, np.exp(-0.5)]:
        A, delta = parts(b, 1, s2)
        x, y = 1.1, 0.6
        R = delta(x, y) - (2*b*np.cosh(x) - s2*2*b*np.cosh(y))*np.log(A(x, y))
        cand = (-2*x)*(-2*y)*(b*np.sinh(x)/x - s2*b*np.sinh(y)/y)
        print('%-10s b=%.4f: R_T=%.10f, R_T/[M1 M2 (v1-v2)]=%.10f' % (name, b, R, R/cand))
    A, delta = parts(1.0, 1, s2)
    print('  scaling (b=1):')
    for lam in [0.1, 0.03, 0.01, 0.003]:
        be1, be2 = 1.7*lam, lam
        r = delta(np.sqrt(3)*be1, np.sqrt(3)*be2)/(3*delta_bad(be1, be2, s2))
        print('    beta scale %.3f: delta_T/(3 b delta_bad) = %.8f' % (lam, r))
    ok &= abs(r - 1) < 1e-2
print('ALL CHECKS PASSED' if ok else 'FAILED')

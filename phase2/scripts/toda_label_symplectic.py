"""Exact lattice test (classical Toda, label frame) of the topological symplectic mechanism of Prop. 2.2:
   sum_j dPn_j ^ dDn_j  =  dPi_2 ^ dM_1 (in, soliton 1 on the left)  -  dPi_1 ^ dM_2 (out)
with  Pn = s 4b (k cosh k - sinh k)   [momentum conjugate to the label position; Hamilton: dE/dPn = b sinh k / k]
      Pi = sum p_n = s 2b sinh k,  M = sum (r_n - a) = -2k,  Dn1 = -log A_T/(2k1), Dn2 = +log A_T/(2k2).
Also verify Pn as the symplectic pull-back  Omega(d_k, d_n0) = sum (dp_n/dk dx_n/dn0 - dp_n/dn0 dx_n/dk).
"""
import mpmath as mp
mp.mp.dps = 30

def A_T(k1, s1, k2, s2, b):
    PT = lambda K, W: W**2 - 4*b**2*mp.sinh(K/2)**2
    w1, w2 = s1*2*b*mp.sinh(k1), s2*2*b*mp.sinh(k2)
    return -PT(2*k1 - 2*k2, w1 - w2)/PT(2*k1 + 2*k2, w1 + w2)

def Pn(k, s, b): return s*4*b*(k*mp.cosh(k) - mp.sinh(k))
def Pi(k, s, b): return s*2*b*mp.sinh(k)
def M(k): return -2*k

ok = True
for a in [0.0, 1.0]:
    b = mp.e**(-a/2)
    for (k1, s1, k2, s2) in [(1.2, 1, 0.5, 1), (0.6, 1, 0.9, -1), (0.3, 1, 0.2, -1), (2.0, 1, 1.1, 1)]:
        D1 = lambda x, y: -mp.log(A_T(x, s1, y, s2, b))/(2*x)
        D2 = lambda x, y: mp.log(A_T(x, s1, y, s2, b))/(2*y)
        lhs = mp.diff(lambda x: Pn(x, s1, b), k1)*mp.diff(lambda y: D1(k1, y), k2) \
            - mp.diff(lambda y: Pn(y, s2, b), k2)*mp.diff(lambda x: D2(x, k2), k1)
        rhs = -mp.diff(lambda y: Pi(y, s2, b), k2)*mp.diff(M, k1) - mp.diff(lambda x: Pi(x, s1, b), k1)*mp.diff(M, k2)
        print('a=%.0f k=(%.1f,%+d; %.1f,%+d):  lhs=%+.15f  rhs=%+.15f  diff=%.1e' % (a, k1, s1, k2, s2, lhs, rhs, abs(lhs - rhs)))
        ok &= abs(lhs - rhs) < 1e-20

# symplectic pull-back for one soliton (a = 1): Omega(d_k, d_n0) vs dPn/dk
import numpy as np
def fields(k, n0, a, nmax=3000):
    b = np.exp(-a/2); n = np.arange(-nmax, nmax + 1, dtype=float); X = k*(n - n0)
    with np.errstate(over='ignore'):
        sech2 = 1/np.cosh(X)**2
    sh2 = np.sinh(k)**2; g = 1 + sh2*sech2; r = a - np.log(g)
    rdot = -(sh2*(-2)*np.tanh(X)*sech2*(-b*np.sinh(k)))/g
    p = -np.cumsum(rdot[::-1])[::-1]
    x = -np.cumsum(r[::-1])[::-1]
    return p, x
a = 1.0; b = np.exp(-a/2); h = 1e-5
for k in [0.4, 1.0]:
    pk = (fields(k + h, 0, a)[0] - fields(k - h, 0, a)[0])/(2*h); xk = (fields(k + h, 0, a)[1] - fields(k - h, 0, a)[1])/(2*h)
    pn = (fields(k, h, a)[0] - fields(k, -h, a)[0])/(2*h); xn = (fields(k, h, a)[1] - fields(k, -h, a)[1])/(2*h)
    Om = np.sum(pk*xn - pn*xk)
    print('pull-back Omega(d_k, d_n0) = %.8f   dPn/dk = 4b k sinh k = %.8f' % (Om, 4*b*k*np.sinh(k)))
print('ALL CHECKS PASSED' if ok else 'FAILED')
assert ok

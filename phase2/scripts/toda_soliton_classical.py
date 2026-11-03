"""Exact classical Toda soliton on a background of spacing a (b = e^{-a/2}):
     e^{-r_n} = b^2 [1 + sinh^2(kap) sech^2(kap n - s b t sinh kap)],   s = +-1.
Compute numerically: compression M_T = sum (r_n - a), momentum P = sum p_n, energy (fixed-length / enthalpy)
     E = sum [p_n^2/2 + e^{-r_n} - b^2 + b^2 (r_n - a)],
label velocity b sinh(kap)/kap, physical velocity a b sinh(kap)/kap (+ frame).
"""
import numpy as np
np.seterr(over='ignore')

def soliton_data(kap, a, s=1, nmax=4000):
    b = np.exp(-a/2)
    n = np.arange(-nmax, nmax + 1, dtype=float)
    X = kap*n                                   # at t = 0
    sh2 = np.sinh(kap)**2
    g = 1 + sh2/np.cosh(X)**2
    r = a - np.log(g)
    # rdot = -d/dt log g ; d/dt X = -s b sinh(kap)
    dgdt = sh2*(-2)*np.tanh(X)/np.cosh(X)**2*(-s*b*np.sinh(kap))
    rdot = -dgdt/g
    # p_n = xdot_n with far-right particles at rest: x_n = x_R - sum_{m>=n} r_m  => p_n = -sum_{m>=n} rdot_m
    p = -np.cumsum(rdot[::-1])[::-1]
    M = np.sum(r - a)
    P = np.sum(p)
    E = np.sum(p**2/2 + np.exp(-r) - b**2 + b**2*(r - a))
    return dict(M=M, P=P, E=E, vlab=b*np.sinh(kap)/kap)

if __name__ == '__main__':
    for kap in [0.2, 0.7, 1.5]:
        for a in [0.0, 1.0]:
            d = soliton_data(kap, a)
            b = np.exp(-a/2)
            print('kap=%.1f a=%.1f: M=%.10f (-2kap=%.10f)  P=%.10f  (2b sinh kap=%.10f)  E=%.10f  (2b^2(sinh k cosh k - k)=%.10f)'
                  % (kap, a, d['M'], -2*kap, d['P'], 2*b*np.sinh(kap), d['E'], 2*b**2*(np.sinh(kap)*np.cosh(kap) - kap)))
            assert abs(d['M'] + 2*kap) < 1e-9 and abs(d['P'] - 2*b*np.sinh(kap)) < 1e-9
            assert abs(d['E'] - 2*b**2*(np.sinh(kap)*np.cosh(kap) - kap)) < 1e-9
    print('ALL CHECKS PASSED')

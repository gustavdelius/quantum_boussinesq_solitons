"""Finite-hbar asymptotic Bethe ansatz of the quantum Toda chain (Sutherland/Mertens), soliton sector.

Two-body problem H_rel = q^2 + e^{-x} (hbar): psi = K_{2iq/hbar}(2 e^{-x/2}/hbar), reflection amplitude
   R = Gamma(i kappa)/Gamma(-i kappa) hbar^{2 i kappa},  kappa = (p2 - p1)/hbar.
Bethe equations in momentum units:  p_j L + sum_l Th(p_j - p_l) = 2 pi hbar I_j with
   Th(D) = 2 hbar Im log Gamma(1 + i D/hbar) + 2 D log hbar,   Th'(D) = 2 Re psi(1 + i D/hbar) + 2 log hbar,
   and Th -> 2 (D log|D| - D) as hbar -> 0 (classical limit used in toda_bethe_classical.py).
Ground state:  2 pi hbar rho - K rho = 1 on [-q, q],  int rho = 1/a.
Excitation at p1 (|p1| > q):  (2 pi hbar - K) F = -Th(. - p1);  P_dr = p1 + int Th(p1-p') rho;
   E_dr = p1^2/2 + int p F;  Phi(1|2) = Th(p1-p2) - int Th'(p1-p') F(p'|p2).
Nystrom method with Gauss-Legendre nodes (node spacing << hbar).
"""
import numpy as np
from scipy.special import loggamma, digamma
from scipy.optimize import brentq

class TodaBAq:
    def __init__(self, a, hbar, N=None):
        self.a, self.h = a, hbar
        self.d = 1/a
        self.N = N or int(max(400, 8*2.5/hbar))
        self._solve_ground()

    def Th(self, D):
        h = self.h
        return 2*h*np.imag(loggamma(1 + 1j*D/h)) + 2*D*np.log(h)

    def K(self, D):
        h = self.h
        return 2*np.real(digamma(1 + 1j*D/h)) + 2*np.log(h)

    def _nodes(self, q):
        x, wts = np.polynomial.legendre.leggauss(self.N)
        return q*x, q*wts

    def _rho(self, q):
        p, wq = self._nodes(q)
        Km = self.K(p[:, None] - p[None, :])*wq[None, :]
        A = 2*np.pi*self.h*np.eye(self.N) - Km
        rho = np.linalg.solve(A, np.ones(self.N))
        return p, wq, A, rho

    def _solve_ground(self):
        def f(q):
            _, wq, _, rho = self._rho(q)
            return np.sum(wq*rho) - self.d
        b = np.exp(-self.a/2)
        # scan upward from below the classical edge 2b to the first sign change (avoid the
        # spurious resonance of (2 pi hbar - K) far above the edge)
        qs = np.linspace(1.0*b, 2*b + 6*self.h, 40)
        fs = [f(q) for q in qs]
        i = next(i for i in range(len(qs) - 1) if fs[i] < 0 <= fs[i + 1])
        self.q = brentq(f, qs[i], qs[i + 1], xtol=1e-12)
        self.p, self.wq, self.A, self.rho = self._rho(self.q)

    def F(self, p1):
        return np.linalg.solve(self.A, -self.Th(self.p - p1))

    def P_dr(self, p1):
        return p1 + np.sum(self.wq*self.Th(p1 - self.p)*self.rho)

    def E_dr(self, p1):
        return p1**2/2 + np.sum(self.wq*self.p*self.F(p1))

    def Phi(self, p1, p2):
        return self.Th(p1 - p2) - np.sum(self.wq*self.K(p1 - self.p)*self.F(p2))

if __name__ == '__main__':
    from toda_bethe_classical import TodaBA
    a = 1.0; cl = TodaBA(a)
    print('classical band edge w = %.6f' % cl.w)
    for h in [0.2, 0.1, 0.05, 0.025]:
        ba = TodaBAq(a, h)
        print('hbar=%.3f N=%d: Fermi edge q=%.6f ; P_dr(1.6)=%.6f (cl %.6f) ; E_dr(1.6)-E_dr(q)=%.6f (cl %.6f); Phi(1.6|0.9*... )'
              % (h, ba.N, ba.q, ba.P_dr(1.6), cl.P_dr(1.6), ba.E_dr(1.6) - ba.E_dr(ba.q*(1+1e-9)),
                 cl.E_dr(1.6) - cl.E_dr(cl.w*(1+1e-12))))

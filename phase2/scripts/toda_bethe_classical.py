"""Classical (hbar -> 0) limit of the asymptotic Bethe ansatz of the quantum Toda chain, and the soliton sector.

Bethe equations (particles on a line of length L, density d = 1/a):
     p_j L + sum_l th(p_j - p_l) = 2 pi hbar I_j,   th(D) = 2 (D log|D| - D)   [classical limit of
     th_q(k) = 2 arg Gamma(1 + i k) + 2 k log hbar, k = D/hbar; derived from the e^{-x} two-body problem]
Ground state as hbar -> 0: 1 + 2 int log|p-p'| rho(p') dp' = 0 on the support  =>  arcsine density of
total weight d on [-w, w], w = 2 e^{-a/2} = 2b  (= Lax band of the static lattice).
Excitation at p1 outside the band: backflow F(p|p1) solves 2 int log|p-p'| F(p') dp' = th(p - p1) on the band;
   P_dr(p1) = p1 + int th(p1 - p') rho(p') dp'     (from its Bethe equation)
   P_tot(p1) = p1 + int F(p|p1) dp                   (sum of momenta; must agree)
   E_dr(p1)  = p1^2/2 + int p F(p|p1) dp
Two excitations: dressed phase  Phi(p1,p2) = th(p1-p2) - 2 int log|p1-p'| F(p'|p2) dp'.
Chebyshev method: log kernel is diagonal on T_n / (pi sqrt(1-x^2)).
"""
import numpy as np
from numpy.polynomial import chebyshev as C

NCH = 400

def th(D):
    D = np.asarray(D, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = 2*(D*np.log(np.abs(D)) - D)
    return np.where(D == 0, 0.0, out)

class TodaBA:
    def __init__(self, a):
        self.a, self.b = a, np.exp(-a/2)
        self.w, self.d = 2*self.b, 1/a

    def backflow(self, p1):
        """Chebyshev coefficients c_n of f(x) = sum c_n T_n(x) / (pi sqrt(1-x^2)), F(p)dp = f(x)dx."""
        g = C.chebinterpolate(lambda x: th(self.w*x - p1)/2, NCH)
        c = np.zeros_like(g)
        c[0] = g[0]/np.log(self.w/2)
        n = np.arange(1, len(g))
        c[1:] = -n*g[1:]
        return c

    def int_log_outside(self, c, p):
        """int log|p - p'| F(p') dp' for p outside the band (|p| > w)."""
        x = p/self.w
        z = abs(x) + np.sqrt(x*x - 1)
        n = np.arange(1, len(c))
        sgn = np.sign(x)**n                      # T_n(-x) = (-1)^n T_n(x)
        return c[0]*(np.log(self.w) + np.log(z/2)) - np.sum(c[1:]*sgn/(n*z**n))

    def P_dr(self, p1):
        # int th(p1 - p') rho(p') dp' with arcsine rho, by Gauss-Chebyshev quadrature
        xs = np.cos(np.pi*(np.arange(2000) + 0.5)/2000)
        return p1 + self.d*np.mean(th(p1 - self.w*xs))

    def P_tot(self, p1):
        return p1 + self.backflow(p1)[0]

    def E_dr(self, p1):
        c = self.backflow(p1)
        return p1**2/2 + self.w*c[1]/2

    def Phi(self, p1, p2):
        c = self.backflow(p2)
        return th(p1 - p2) - 2*self.int_log_outside(c, p1)

if __name__ == '__main__':
    # dictionary: soliton kappa <-> p1 = 2b cosh kappa; P_dr = Pn/a = (4b/a)(k cosh k - sinh k);
    # E_dr(p1) - E_dr(edge) = 2 b^2 (sinh k cosh k - k)
    for a in [1.0, 2.0]:
        ba = TodaBA(a); b = ba.b
        Eedge = ba.E_dr(ba.w*(1 + 1e-12))
        print('a = %.1f (b = %.4f, band [-%.4f, %.4f]); edge energy %.8f (2b^2 = %.8f)' % (a, b, ba.w, ba.w, Eedge, 2*b*b))
        for kap in [0.3, 0.8, 1.5]:
            p1 = 2*b*np.cosh(kap)
            Pn = 4*b/a*(kap*np.cosh(kap) - np.sinh(kap)); Es = 2*b**2*(np.sinh(kap)*np.cosh(kap) - kap)
            print('  kap=%.1f: P_dr=%.10f P_tot=%.10f Pn/a=%.10f | E_dr-E_edge=%.10f E_sol=%.10f'
                  % (kap, ba.P_dr(p1), ba.P_tot(p1), Pn, ba.E_dr(p1) - Eedge, Es))
            assert abs(ba.P_dr(p1) - Pn) < 1e-8 and abs(ba.P_tot(p1) - Pn) < 1e-8 and abs(ba.E_dr(p1) - Eedge - Es) < 1e-8
    print('ALL CHECKS PASSED')

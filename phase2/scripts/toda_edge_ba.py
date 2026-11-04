"""Fermi-edge double-scaling limit of the quantum Toda asymptotic Bethe ansatz (module).

Model: toda_bethe_quantum.py (digamma kernel), solved here with LU reuse and a secant search for the
Fermi edge q.  Soliton sizes are fixed by PHYSICAL dressed momenta:  P_dr(p) - P_dr(q) = Pn(kappa)/a,
Pn = s 4b (k cosh k - sinh k),  kappa = sqrt3 beta,  hbar_T = 3 b hbar_B  (action scale of the Boussinesq
limit, toda_route.tex).  Double scaling: y_j = beta_j^2 / hbar_B fixed, hbar_T -> 0.

Normalised label-frame phase (vanishes when either soliton shrinks to the edge):
   Psi_T(1,2) = [Phi(1|2) - Phi(1|e2) - Phi(e1|2) + Phi(e1|e2)] - M1 (Pn2/a - Pi2),
where e_j is the edge excitation on soliton j's side, and the last term is the classical frame change
(physical -> label frame, toda_ba_vs_label.py).  Exact S-matrix phase in the corner:  Psi_T / hbar_T.
"""
import numpy as np
from scipy.special import loggamma, digamma
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq


class EdgeBA:
    def __init__(self, a, hbar, npts=None):
        self.a, self.h, self.b = a, hbar, np.exp(-a/2)
        self.w = 2*self.b
        self.N = npts or int(4*np.pi*self.w/hbar) + 300
        self.x, self.wx = np.polynomial.legendre.leggauss(self.N)
        self._ground()

    def Th(self, D):
        h = self.h
        return 2*h*np.imag(loggamma(1 + 1j*D/h)) + 2*D*np.log(h)

    def K(self, D):
        h = self.h
        return 2*np.real(digamma(1 + 1j*D/h)) + 2*np.log(h)

    def _system(self, q):
        p, wq = q*self.x, q*self.wx
        A = 2*np.pi*self.h*np.eye(self.N) - self.K(p[:, None] - p[None, :])*wq[None, :]
        return p, wq, A

    def _count(self, q):
        p, wq, A = self._system(q)
        rho = np.linalg.solve(A, np.ones(self.N))
        return np.sum(wq*rho) - 1/self.a

    def _ground(self):
        q0, q1 = self.w + 0.8*self.h, self.w + 1.6*self.h
        f0, f1 = self._count(q0), self._count(q1)
        for _ in range(30):
            q2 = q1 - f1*(q1 - q0)/(f1 - f0)
            q0, f0, q1 = q1, f1, q2
            f1 = self._count(q1)
            if abs(q1 - q0) < 1e-13:
                break
        self.q = q1
        self.p, self.wq, A = self._system(self.q)
        self.lu = lu_factor(A)
        self.rho = lu_solve(self.lu, np.ones(self.N))

    def F(self, p1):
        return lu_solve(self.lu, -self.Th(self.p - p1))

    def P_dr(self, p1):
        return p1 + np.sum(self.wq*self.Th(p1 - self.p)*self.rho)

    def Phi(self, p1, p2, F2=None):
        F2 = self.F(p2) if F2 is None else F2
        return self.Th(p1 - p2) - np.sum(self.wq*self.K(p1 - self.p)*F2)

    def edge(self, s):
        return s*self.q*(1 + 1e-10)

    def p_of_P(self, P, s):
        e = self.edge(s); P0 = self.P_dr(e)
        f = lambda p: self.P_dr(p) - P0 - P
        lo, hi = (e, 3*self.w) if s > 0 else (-3*self.w, e)
        return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)


def Pn(k, s, b): return s*4*b*(k*np.cosh(k) - np.sinh(k))
def Pi(k, s, b): return s*2*b*np.sinh(k)


def Psi_T(m, y1, s1, y2, s2):
    """Normalised label-frame Toda phase / hbar_T for Boussinesq scaling variables y = beta^2/hbar_B."""
    a, b = m.a, m.b
    hB = m.h/(3*b)
    be1, be2 = np.sqrt(y1*hB), np.sqrt(y2*hB)
    k1, k2 = np.sqrt(3)*be1, np.sqrt(3)*be2
    p1, p2 = m.p_of_P(Pn(k1, s1, b)/a, s1), m.p_of_P(Pn(k2, s2, b)/a, s2)
    e1, e2 = m.edge(s1), m.edge(s2)
    F2, Fe2 = m.F(p2), m.F(e2)
    norm = m.Phi(p1, p2, F2) - m.Phi(p1, e2, Fe2) - m.Phi(e1, p2, F2) + m.Phi(e1, e2, Fe2)
    frame = (-2*k1)*(Pn(k2, s2, b)/a - Pi(k2, s2, b))
    return (norm - frame)/m.h, (be1, be2, hB)

# (V), (c): standalone verification of the equations and edge normalisation
# in toda_edge.tex. Imports retain their original module-only behaviour.
if __name__ == '__main__':
    # Use the existing nodecheck.py configuration; do not change solver defaults.
    _check_m = EdgeBA(1.0, 0.04, npts=700)
    # Secant stopping is 1e-13 in q; 1e-10 allows conditioning of the count.
    assert abs(np.sum(_check_m.wq*_check_m.rho) - 1/_check_m.a) < 1e-10
    assert np.all(_check_m.rho > 0), "Ground-state density must be positive"
    assert np.max(np.abs(_check_m._system(_check_m.q)[2] @ _check_m.rho - 1)) < 1e-10
    for _check_s in (1, -1):
        _check_P = Pn(np.sqrt(3*0.5*_check_m.h/(3*_check_m.b)), _check_s, _check_m.b)/_check_m.a
        assert abs(_check_m.P_dr(_check_m.p_of_P(_check_P, _check_s))
                   - _check_m.P_dr(_check_m.edge(_check_s)) - _check_P) < 1e-12
        # Four-term phase must vanish on either edge. Cancellation is divided
        # by h=0.04; 1e-10 comfortably resolves a nonzero normalisation error.
        assert abs(Psi_T(_check_m, 0.0, 1, 0.5, _check_s)[0]) < 1e-10
        assert abs(Psi_T(_check_m, 0.5, 1, 0.0, _check_s)[0]) < 1e-10
    print("ALL CHECKS PASSED")

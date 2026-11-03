"""O(hbar) corrections in the quantum Toda chain soliton sector, at fixed physical (dressed) momenta.
For hbar -> 0:  E(P) = E_cl(P) + hbar E_1(P) + ...,  Phi(P1,P2) = Phi_cl + hbar Phi_1 + ...  (S ~ exp(i Phi/hbar)).
Richardson extrapolation over hbar = 0.1, 0.05, 0.025 checks linear convergence.
"""
import numpy as np
from scipy.optimize import brentq
from toda_bethe_classical import TodaBA
from toda_bethe_quantum import TodaBAq

a = 1.0
cl = TodaBA(a)
def invert(model, P, sign, edge):
    P0 = model.P_dr(sign*edge*(1 + 1e-9))          # momentum of the edge excitation (0 classically)
    f = lambda p: model.P_dr(p) - P0 - P
    lo, hi = (edge*(1 + 1e-9), 6*edge) if sign > 0 else (-6*edge, -edge*(1 + 1e-9))
    return brentq(f, lo, hi, xtol=1e-13)

cases = [('overtaking', 0.6, 1, 0.15, 1), ('head-on', 0.4, 1, -0.3, -1)]
hs = [0.1, 0.05, 0.025]
models = {h: TodaBAq(a, h) for h in hs}
for name, P1, s1, P2, s2 in cases:
    pc1, pc2 = invert(cl, P1, s1, cl.w), invert(cl, P2, s2, cl.w)
    Ecl = cl.E_dr(pc1) - cl.E_dr(cl.w*(1 + 1e-12))
    Phi_cl, Phi_cl21 = cl.Phi(pc1, pc2), cl.Phi(pc2, pc1)
    print('%s: P=(%.2f,%.2f)  classical: E1=%.8f  Phi(1|2)=%.8f  Phi(2|1)=%.8f' % (name, P1, P2, Ecl, Phi_cl, Phi_cl21))
    rows = []
    for h in hs:
        m = models[h]
        p1, p2 = invert(m, P1, s1, m.q), invert(m, P2, s2, m.q)
        E = m.E_dr(p1) - m.E_dr(m.q*(1 + 1e-9))
        rows.append(((E - Ecl)/h, (m.Phi(p1, p2) - Phi_cl)/h, (m.Phi(p2, p1) - Phi_cl21)/h))
        print('   hbar=%.3f: (E-E_cl)/hbar=%+.6f  (Phi12-cl)/hbar=%+.6f  (Phi21-cl)/hbar=%+.6f' % (h, *rows[-1]))
    r = np.array(rows)
    rich = 2*r[2] - r[1]
    print('   Richardson (hbar->0):   E_1=%+.5f   Phi_1(1|2)=%+.5f   Phi_1(2|1)=%+.5f ; antisym part %+.5f, sym part %+.5f'
          % (rich[0], rich[1], rich[2], (rich[1] - rich[2])/2, (rich[1] + rich[2])/2))
    assert abs(r[2, 0] - r[1, 0]) < 0.6*abs(r[1, 0] - r[0, 0]) + 1e-6      # O(hbar) convergence of E
print('ALL CHECKS PASSED')

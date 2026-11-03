"""Link between the classical limit of the quantum-Toda Bethe phases and the label-frame phase delta_T.
Prediction (frame change x = a n + displacement, P = Pn/a):
   Phi(1|2) - delta_T(1,2) - M1 (Pn2/a - Pi2)  depends on kappa1 only,
   Phi(2|1) + delta_T(1,2) - M2 (Pn1/a - Pi1)  depends on kappa2 only,
with delta_T = (p1 - p2) log A_T + M1 M2 (v1 - v2)  (closed form, toda_scaling_fast.py)."""
import numpy as np
from toda_bethe_classical import TodaBA
a = 1.0; ba = TodaBA(a); b = ba.b
pof = lambda k, s: s*2*b*np.cosh(k)
Pn = lambda k, s: s*4*b*(k*np.cosh(k) - np.sinh(k)); Pi = lambda k, s: s*2*b*np.sinh(k); M = lambda k: -2*k
v = lambda k, s: s*b*np.sinh(k)/k
def A_T(k1, s1, k2, s2):
    PT = lambda K, W: W**2 - 4*b**2*np.sinh(K/2)**2
    return -PT(2*k1 - 2*k2, s1*2*b*np.sinh(k1) - s2*2*b*np.sinh(k2))/PT(2*k1 + 2*k2, s1*2*b*np.sinh(k1) + s2*2*b*np.sinh(k2))
def dT(k1, s1, k2, s2):
    return (pof(k1, s1) - pof(k2, s2))*np.log(A_T(k1, s1, k2, s2)) + M(k1)*M(k2)*(v(k1, s1) - v(k2, s2))
ok = True
for s2, name in [(1, 'overtaking'), (-1, 'head-on')]:
    k1 = 1.3 if s2 == 1 else 0.7
    r1 = [ba.Phi(pof(k1, 1), pof(k2, s2)) - dT(k1, 1, k2, s2) - M(k1)*(Pn(k2, s2)/a - Pi(k2, s2)) for k2 in [0.2, 0.5, 0.8]]
    k2 = 0.4
    r2 = [ba.Phi(pof(k2, s2), pof(k1, 1)) + dT(k1, 1, k2, s2) - M(k2)*(Pn(k1, 1)/a - Pi(k1, 1)) for k1 in ([0.9, 1.3, 1.8] if s2 == 1 else [0.3, 0.7, 1.1])]
    print('%-10s: [Phi(1|2) - ...] at k2=0.2,0.5,0.8: %s   spread %.1e' % (name, np.round(r1, 10), np.ptp(r1)))
    print('%-10s: [Phi(2|1) + ...] at three k1:      %s   spread %.1e' % (name, np.round(r2, 10), np.ptp(r2)))
    ok &= np.ptp(r1) < 1e-8 and np.ptp(r2) < 1e-8
print('ALL CHECKS PASSED' if ok else 'FAILED'); assert ok

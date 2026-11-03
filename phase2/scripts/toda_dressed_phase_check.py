"""Classical checks of the Toda Bethe-ansatz dressed two-soliton phase Phi (toda_bethe_classical.py).

Dictionary (verified in this script): soliton kappa, direction s  <->  excitation at p = s 2b cosh(kappa);
   canonical (physical-frame) momentum  P = s (4b/a)(kappa cosh kappa - sinh kappa);  E = 2b^2(sinh k cosh k - k).
Classical Toda two-soliton (Hirota, label n): A_T = -P_T(k1-k2, w1-w2)/P_T(k1+k2, w1+w2), P_T = w^2 - 4b^2 sinh^2(K/2),
   kH = 2 kappa, w = s 2b sinh(kappa).  Label shifts (soliton 1 starts on the left):
   Dn1 = -log A_T / kH1,  Dn2 = +log A_T / kH2.
Physical shifts include the displacement caused by the other soliton's compression M = -2 kappa:
   Dx1 = a Dn1 - 2 kappa2,   Dx2 = a Dn2 + 2 kappa1.
Result: Phi is NOT antisymmetric; instead Dx2 = -dPhi(1|2)/dP2 and Dx1 = +dPhi(2|1)/dP1 exactly.
"""
import numpy as np
from toda_bethe_classical import TodaBA

def classical_shifts(k1, s1, k2, s2, a):
    b = np.exp(-a/2)
    kH1, kH2 = 2*k1, 2*k2
    w1, w2 = s1*2*b*np.sinh(k1), s2*2*b*np.sinh(k2)
    PT = lambda K, W: W**2 - 4*b**2*np.sinh(K/2)**2
    A = -PT(kH1 - kH2, w1 - w2)/PT(kH1 + kH2, w1 + w2)
    Dn1, Dn2 = -np.log(A)/kH1, np.log(A)/kH2
    return a*Dn1 - 2*k2, a*Dn2 + 2*k1, A

def Pcan(k, s, a):
    b = np.exp(-a/2)
    return s*(4*b/a)*(k*np.cosh(k) - np.sinh(k))

if __name__ == '__main__':
    a = 1.0; ba = TodaBA(a); b = ba.b
    pof = lambda k, s: s*2*b*np.cosh(k)
    ok = True
    for (k1, s1, k2, s2, name) in [(1.2, 1, 0.5, 1, 'overtaking'), (0.6, 1, 0.9, -1, 'head-on'), (0.3, 1, 0.2, -1, 'head-on small')]:
        p1, p2 = pof(k1, s1), pof(k2, s2)
        anti = ba.Phi(p1, p2) + ba.Phi(p2, p1)
        h = 1e-5
        # dPhi/dP1 at fixed p2: chain rule through kappa1
        dPhi1 = (ba.Phi(pof(k1 + h, s1), p2) - ba.Phi(pof(k1 - h, s1), p2))/(2*h)
        dP1 = (Pcan(k1 + h, s1, a) - Pcan(k1 - h, s1, a))/(2*h)
        dPhi2 = (ba.Phi(p1, pof(k2 + h, s2)) - ba.Phi(p1, pof(k2 - h, s2)))/(2*h)
        dP2 = (Pcan(k2 + h, s2, a) - Pcan(k2 - h, s2, a))/(2*h)
        Dx1, Dx2, A = classical_shifts(k1, s1, k2, s2, a)
        print('%-13s A_T=%.6f  Phi(1,2)+Phi(2,1)=%.2e | -dPhi/dP1=%+.6f  Dx1=%+.6f | -dPhi/dP2=%+.6f  Dx2=%+.6f'
              % (name, A, anti, -dPhi1/dP1, Dx1, -dPhi2/dP2, Dx2))
        dPhi21 = (ba.Phi(p2, pof(k1 + h, s1)) - ba.Phi(p2, pof(k1 - h, s1)))/(2*h)
        print('%-13s  +dPhi(2|1)/dP1=%+.6f (= Dx1)' % ('', dPhi21/dP1))
        ok &= abs(-dPhi2/dP2 - Dx2) < 1e-6 and abs(dPhi21/dP1 - Dx1) < 1e-6
    print('=> Dx2 = -dPhi(1|2)/dP2 and Dx1 = +dPhi(2|1)/dP1 (soliton 1 starts on the left)')
    print('ALL CHECKS PASSED' if ok else 'FAILED'); assert ok

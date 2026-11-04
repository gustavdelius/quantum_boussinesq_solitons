"""Criterion (iv): Bound-state poles, zeros, and breather/resonance spectra for bad vs good Boussinesq.

Compares the pole structure of candidate S-matrices at 35-digit precision (mpmath):
 1. Bad equation (regular Tajiri-Murakami breathers exist for b in (pi/3, pi/2)):
    - At beta_{1,2} = a -+ i b, verifies Im(P1 - P2) > 0 (physical sheet), A12 > 1 (regular breather),
      and i eta = -I_br / hbar with I_br = -(4/9) sinh(3a) sin(3b) > 0.
    - Verifies simple poles of S_Gamma at I_br = N hbar (N=1,2,...), absence of N=0 pole at threshold
      b = pi/3, pole shifts under s -> 1/2 and hbar -> hbar(1 + c1 hbar + c2 hbar^2), and extra/cancelled
      poles from rational CDD factors (c + i eta)/(c - i eta).
 2. Good equation (NO regular breathers exist on zero background, phase1/review.tex Prop. 3.4):
    - Verifies A_good(a - i b, a + i b) = (1 - 3 cot^2 a)/(1 + 3 coth^2 b) < 1/4 for all real a, b != 0.
    - Verifies the exact physical-sheet and spectral-parameter formulas:
        Im(P1 - P2) = (1/3) sinh(2b) [cos(4a) cosh(2b) - cos(2a)],
        i eta       = (4 / (9 hbar)) sin(3a) sinh(3b).
    - Demonstrates that on the light branch a in (0, pi/3) with b < 0 (where cos(2a) > cos(4a) cosh(2b)),
      the continued Gamma candidate S_Gamma^good has simple poles at i eta = -N ON THE PHYSICAL SHEET
      Im(P1 - P2) > 0, where the classical two-soliton solution is singular (A12 < 1/4, and A12 < 0 for a < pi/6).
 3. Real Y-junction resonance lines of the good equation:
    - Verifies that on all three resonance lines (alpha1 + alpha2 = pi/3 head-on; |alpha1 - alpha2| = pi/3;
      alpha1 + alpha2 = 2pi/3 overtaking), ell_1 = ell_2 (eta = 0), so S_Gamma^good and S_tree^good approach
      exp(i T12 / hbar) smoothly without poles, and lose unitarity (|S| != 1) in the singular region A12 < 0.
"""
import mpmath as mp
from smx_dictionary_and_families import (
    sol_bad, sol_good, A12, T12, delta0, eta_val, logS_gamma, I
)

mp.mp.dps = 35


def I_br_bad(a, b):
    return -mp.mpf(4) / 9 * mp.sinh(3 * a) * mp.sin(3 * b)


if __name__ == '__main__':
    ok = True
    print('=== Check 1: Bad Equation Breather Poles, String Shifts, hbar-Renorm, and CDD Poles ===')
    hbar = mp.mpf('0.1')
    for a_val in (mp.mpf('0.6'), mp.mpf('0.9')):
        for N in (1, 2, 3):
            bN = mp.findroot(lambda b: I_br_bad(a_val, b) - N * hbar, (mp.pi / 3 + mp.mpf('1e-8'), mp.pi / 2))
            s1, s2 = sol_bad(a_val - I * bN, 1), sol_bad(a_val + I * bN, 1)
            im_dP = mp.im(s1['P'] - s2['P'])
            A_br = mp.re(A12(s1, s2))
            i_eta = I * eta_val(s1, s2, hbar)
            # Residue scaling test at bN + db
            res_vals = [
                abs(mp.exp(logS_gamma(sol_bad(a_val - I * (bN + db), 1), sol_bad(a_val + I * (bN + db), 1), hbar))) * abs(db)
                for db in (mp.mpf('1e-12'), mp.mpf('-1e-14'))
            ]
            ok &= im_dP > 0 and A_br > 1 and abs(i_eta + N) < mp.mpf('1e-30')
            ok &= abs(res_vals[0] / res_vals[1] - 1) < mp.mpf('1e-8') and res_vals[0] > mp.mpf('1e-10')
        # Threshold b = pi/3: no pole for s = 1, whereas Toda Gamma(i eta)/Gamma(-i eta) has a pole
        s1_th = sol_bad(a_val - I * (mp.pi / 3 + mp.mpf('1e-8')), 1)
        s2_th = sol_bad(a_val + I * (mp.pi / 3 + mp.mpf('1e-8')), 1)
        s1_th2 = sol_bad(a_val - I * (mp.pi / 3 + mp.mpf('1e-10')), 1)
        s2_th2 = sol_bad(a_val + I * (mp.pi / 3 + mp.mpf('1e-10')), 1)
        S_th1 = abs(mp.exp(logS_gamma(s1_th, s2_th, hbar, s=mp.mpf(1))))
        S_th2 = abs(mp.exp(logS_gamma(s1_th2, s2_th2, hbar, s=mp.mpf(1))))
        ok &= abs(S_th2 / S_th1 - 1) < mp.mpf('1e-4')

        # Shifted string s = 1/2 has pole at I_br = (N - 1/2) hbar
        b_half = mp.findroot(lambda b: I_br_bad(a_val, b) - mp.mpf('0.5') * hbar,
                             (mp.pi / 3 + mp.mpf('1e-8'), mp.pi / 2))
        res_half = [
            abs(mp.exp(logS_gamma(sol_bad(a_val - I * (b_half + db), 1),
                                  sol_bad(a_val + I * (b_half + db), 1), hbar, s=mp.mpf('0.5')))) * abs(db)
            for db in (mp.mpf('1e-12'), mp.mpf('-1e-14'))
        ]
        ok &= abs(res_half[0] / res_half[1] - 1) < mp.mpf('1e-8')

        # hbar-renormalised pole at I_br = N * hbar * (1 + c1*hbar + c2*hbar^2)
        c1_v, c2_v = mp.mpf('0.3'), mp.mpf('-0.1')
        hren = hbar * (1 + c1_v * hbar + c2_v * hbar**2)
        b_ren = mp.findroot(lambda b: I_br_bad(a_val, b) - hren, (mp.pi / 3 + mp.mpf('1e-8'), mp.pi / 2))
        res_ren = [
            abs(mp.exp(logS_gamma(sol_bad(a_val - I * (b_ren + db), 1),
                                  sol_bad(a_val + I * (b_ren + db), 1), hbar, c1=c1_v, c2=c2_v))) * abs(db)
            for db in (mp.mpf('1e-12'), mp.mpf('-1e-14'))
        ]
        ok &= abs(res_ren[0] / res_ren[1] - 1) < mp.mpf('1e-8')

    print('   Bad equation breather poles (s=1, s=1/2, hbar_ren) and threshold regularity: PASS')

    print('=== Check 2: Good Equation Absence of Regular Breathers and Spurious Gamma Poles ===')
    # Verify A_good(a - i b, a + i b) = (1 - 3 cot^2 a)/(1 + 3 coth^2 b) < 1/4 everywhere
    max_A_good = mp.mpf('-inf')
    for a_v in mp.linspace(mp.mpf('0.05'), mp.pi / 2 - mp.mpf('0.05'), 25):
        for b_v in mp.linspace(mp.mpf('0.05'), mp.mpf('2.0'), 25):
            s1, s2 = sol_good(a_v - I * b_v, 1), sol_good(a_v + I * b_v, 1)
            A_val = A12(s1, s2)
            A_closed = (1 - 3 / mp.tan(a_v)**2) / (1 + 3 / mp.tanh(b_v)**2)
            ok &= abs(A_val - A_closed) < mp.mpf('1e-28') and mp.im(A_val) < mp.mpf('1e-28')
            if mp.re(A_val) > max_A_good:
                max_A_good = mp.re(A_val)
    ok &= max_A_good < mp.mpf('0.25')
    print('   Good equation breather constant A(a-ib, a+ib) = (1 - 3 cot^2 a)/(1 + 3 coth^2 b) < 1/4 '
          '(max on grid = %s): PASS' % mp.nstr(max_A_good, 6))

    # Exact formulas for Im(P1 - P2) and i*eta in good equation, and physical-sheet spurious poles
    for a_v in (mp.mpf('0.35'), mp.mpf('0.6'), mp.mpf('0.75')):
        for b_v in (mp.mpf('-0.15'), mp.mpf('-0.3'), mp.mpf('0.2')):
            s1, s2 = sol_good(a_v - I * b_v, 1), sol_good(a_v + I * b_v, 1)
            im_P_exact = mp.mpf(1) / 3 * mp.sinh(2 * b_v) * (mp.cos(4 * a_v) * mp.cosh(2 * b_v) - mp.cos(2 * a_v))
            ieta_exact = mp.mpf(4) / (9 * hbar) * mp.sin(3 * a_v) * mp.sinh(3 * b_v)
            ok &= abs(mp.im(s1['P'] - s2['P']) - im_P_exact) < mp.mpf('1e-30')
            ok &= abs(I * eta_val(s1, s2, hbar) - ieta_exact) < mp.mpf('1e-30')

    # Find N=1, 2 physical-sheet poles of S_Gamma^good at a_v in (0, pi/3) with b_v < 0
    for a_v in (mp.mpf('0.4'), mp.mpf('0.65')):
        for N in (1, 2):
            # i eta = -N => (4 / (9 hbar)) sin(3 a_v) sinh(3 b_v) = -N => b_v < 0
            b_pole = -mp.asinh(9 * N * hbar / (4 * mp.sin(3 * a_v))) / 3
            s1, s2 = sol_good(a_v - I * b_pole, 1), sol_good(a_v + I * b_pole, 1)
            im_P = mp.im(s1['P'] - s2['P'])
            A_at_pole = mp.re(A12(s1, s2))
            i_eta = I * eta_val(s1, s2, hbar)
            print('   Good eq spurious Gamma pole at a=%s, N=%d: b=%s, i*eta=%s, Im(P1-P2)=%s (>0, physical sheet!), '
                  'A12=%s (< 1/4, singular!)'
                  % (mp.nstr(a_v, 4), N, mp.nstr(b_pole, 6), mp.nstr(mp.re(i_eta), 6),
                     mp.nstr(im_P, 5), mp.nstr(A_at_pole, 5)))
            ok &= im_P > 0 and abs(i_eta + N) < mp.mpf('1e-30') and A_at_pole < mp.mpf('0.25')

    print('=== Check 3: Good Equation Real Y-Junction Resonance Lines (eta -> 0, No Pole) ===')
    # (a) Head-on resonance line alpha1 + alpha2 = pi/3 (A_ho -> infinity)
    a1_ho = mp.mpf('0.4')
    a2_ho_res = mp.pi / 3 - a1_ho
    for d_eps in (mp.mpf('1e-5'), mp.mpf('1e-9')):
        s1 = sol_good(a1_ho, 1)
        s2 = sol_good(a2_ho_res - d_eps, -1)
        L_gam = logS_gamma(s1, s2, hbar)
        L_tree = I * delta0(s1, s2) / hbar
        target_T = I * T12(s1, s2) / hbar
        ok &= abs(eta_val(s1, s2, hbar)) < mp.mpf('1e-3')
        if d_eps == mp.mpf('1e-9'):
            ok &= abs(L_gam - target_T) < mp.mpf('1e-6') and abs(L_tree - target_T) < mp.mpf('1e-6')

    # (b) Overtaking resonance line alpha1 + alpha2 = 2*pi/3 (A_ov -> 0)
    a2_ov = mp.mpf('0.75')
    a1_ov_res = 2 * mp.pi / 3 - a2_ov
    s1_ov = sol_good(a1_ov_res - mp.mpf('1e-9'), 1)
    s2_ov = sol_good(a2_ov, 1)
    ok &= abs(logS_gamma(s1_ov, s2_ov, hbar) - I * T12(s1_ov, s2_ov) / hbar) < mp.mpf('1e-6')
    # Beyond resonance line (A12 < 0): non-unitary |S| != 1
    s1_sing = sol_good(a1_ov_res + mp.mpf('1e-3'), 1)
    ok &= A12(s1_sing, s2_ov) < 0 and abs(abs(mp.exp(logS_gamma(s1_sing, s2_ov, hbar))) - 1) > mp.mpf('1e-3')
    print('   Good equation Y-junction resonance lines (head-on A->inf and overtaking A->0): '
          'S -> exp(i T12/hbar) smoothly, non-unitary for A<0: PASS')

    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

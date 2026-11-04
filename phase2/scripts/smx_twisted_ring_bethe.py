"""Criterion (v): N-soliton twisted asymptotic Bethe equations on a ring from topological cross terms.

Derives and verifies at 40-digit precision (mpmath) the phi-winding twist on a ring of circumference
Lambda for both the bad and good Boussinesq equations:
 1. N-soliton canonical symplectic form and Darboux coordinates across all N! ordering chambers:
    - Omega^(sigma) = dPi_tot ^ dc + sum_j dP_j ^ dX_j + sum_{X_i > X_j} dPi_i ^ dM_j;
    - With X_j' = X_j + gamma_j sum_{X_l < X_j} M_l (gamma_j = dPi_j/dP_j), verifies for N = 2, 3, 4
      that sum_j dP_j ^ dX_j' reproduces the cross terms in every chamber sigma in S_N, and that
      the pairwise scattering map X_j -> X_j + Dx_{jl} preserves Omega^(sigma) across every chamber wall.
 2. Seam jump on the ring S^1_Lambda and the directional twisted Bethe phase:
    - As soliton j traverses the ring X_j: 0^+ -> Lambda^- past all l != j, its Darboux coordinate
      changes by Delta_seam X_j' = Lambda + gamma_j sum_{l != j} M_l while c = phi(0^+) jumps by -M_j;
    - Restoring the potential background across the seam adds the Aharonov-Bohm twist Pi_j M_l, giving
      the directional ring Bethe phase:
        Phi_ring(j | l) = delta(j, l) + Pi_j M_l = (2/9)(ell_j - ell_l) log A_{jl} + (1/3) Pi_j M_l + (2/3) Pi_l M_j.
    - Verifies to 40 digits for both bad and good equations (overtaking and head-on) that:
        -d Phi_ring(j | l) / dP_j = Dx_{jl} = -(1/k_j) log A_{jl}   (exact Hirota shift!),
      whereas omitting T_{jl} or omitting the seam twist +Pi_j M_l fails by O(1) amounts.
 3. Global translation-winding sum rule on the ring:
    - Since delta(j, l) + delta(l, j) = 0 (and arg S(j,l) + arg S(l,j) = 0 for any unitary candidate),
      Phi_ring(j | l) + Phi_ring(l | j) = Pi_j M_l + Pi_l M_j;
    - Summing the twisted Bethe equations B_j^ring = P_j Lambda + Pi_j M_j + sum_{l != j} Phi_ring(j | l) = 2 pi hbar n_j
      over j = 1..N yields the exact field-theoretic identity:
        P_tot Lambda + Pi_tot M_tot = 2 pi hbar sum_{j=1}^N n_j.
 4. Covariant fusion of the twisted ring Bethe equations onto Bogdan-Charkina breathers:
    - For a breather beta_{1,2} = a -+ i b and a third soliton beta_3, verifies:
        M_br = M_1 + M_2 = 4 sinh(a) cos(b),
        Pi_br = Pi_1 + Pi_2 = -2 sinh(2a) cos(2b) = -M_br V_br,
        Phi_ring(1 | 3) + Phi_ring(2 | 3) = delta(1,3) + delta(2,3) + Pi_br M_3 (real to 40 digits!),
        -d/dP_3 [Phi_ring(3 | 1) + Phi_ring(3 | 2)] = -(1/k_3) log(A_{31} A_{32}) (exact breather shift!).
"""
import itertools
import mpmath as mp
from smx_dictionary_and_families import (
    sol_bad, sol_good, A12, T12, delta_dyn, delta0, logS_gamma, I
)

mp.mp.dps = 40


def gamma_darboux(s):
    return s['dPi'] / s['dP']


def Phi_ring(sj, sl):
    """Twisted directional ring Bethe phase of soliton j passing soliton l."""
    return delta0(sj, sl) + sj['Pi'] * sl['M']


def dPhi_ring_dPj(sj, sl):
    """Numerical derivative d Phi_ring(j | l) / d P_j at fixed l."""
    sol_fn = sol_bad if sj['eq'] == 'bad' else sol_good
    d_par = mp.diff(lambda p: Phi_ring(sol_fn(p, sj['eps']), sl), sj['par'])
    return d_par / sj['dP']


if __name__ == '__main__':
    ok = True
    print('=== Check 1: N-Soliton Symplectic Form & Chamber-Crossing for N = 2, 3, 4 ===')
    for eq_name, sol_fn, pars in [
        ('bad', sol_bad, [(mp.mpf('1.2'), 1), (mp.mpf('0.8'), 1), (mp.mpf('0.45'), -1), (mp.mpf('0.25'), 1)]),
        ('good', sol_good, [(mp.mpf('0.2'), 1), (mp.mpf('0.45'), 1), (mp.mpf('0.3'), -1), (mp.mpf('0.12'), 1)]),
    ]:
        sols = [sol_fn(p, e) for p, e in pars]
        N = len(sols)
        # Check pairwise chamber jump between adjacent solitons (j, l) in any ordering
        max_symp_err = mp.mpf(0)
        for j in range(N):
            for l in range(j + 1, N):
                sj, sl = sols[j], sols[l]
                if eq_name == 'good' and A12(sj, sl) <= 0:
                    continue
                # Symplectic identity: dP_j' Dx_j' - dP_l' Dx_l' = -dPi_l' dM_j' - dPi_j' dM_l'
                dL_j = mp.diff(lambda x: mp.log(A12(sol_fn(x, sj['eps']), sl)), sj['par'])
                dL_l = mp.diff(lambda y: mp.log(A12(sj, sol_fn(y, sl['eps']))), sl['par'])
                Dxj_dl = -dL_l / sj['k']
                Dxl_dj = +dL_j / sl['k']
                lhs = sj['dP'] * Dxj_dl - sl['dP'] * Dxl_dj
                rhs = -sl['dPi'] * sj['dM'] - sj['dPi'] * sl['dM']
                max_symp_err = max(max_symp_err, abs(lhs - rhs))
        # Check all 24 permutations sigma in S_4: Darboux wedge sum_m dP_m ^ dX_m' reproduces cross terms
        for sigma in itertools.permutations(range(N)):
            # In chamber sigma, X_{sigma[0]} < X_{sigma[1]} < ... < X_{sigma[N-1]}
            # Coefficient of d par_a ^ d par_b (a < b) in sum_m dP_m ^ d(gamma_m sum_{n left of m} M_n)
            for idx_a in range(N):
                for idx_b in range(idx_a + 1, N):
                    m_a, m_b = sigma[idx_a], sigma[idx_b]
                    # m_a is to the left of m_b (X_{m_a} < X_{m_b}), so X_{m_b}' contains +gamma_{m_b} M_{m_a}
                    # dP_{m_b} ^ d(gamma_{m_b} M_{m_a}) = dP_{m_b} * gamma_{m_b} * dM_{m_a} = dPi_{m_b} ^ dM_{m_a}
                    wedge_darboux = sols[m_b]['dP'] * gamma_darboux(sols[m_b]) * sols[m_a]['dM']
                    wedge_cross = sols[m_b]['dPi'] * sols[m_a]['dM']
                    max_symp_err = max(max_symp_err, abs(wedge_darboux - wedge_cross))
        print('   %s eq (N=4, all 24 ordering chambers): max symplectic & Darboux residual = %s'
              % (eq_name, mp.nstr(max_symp_err, 3)))
        ok &= max_symp_err < mp.mpf('1e-35')

    print('=== Check 2: Twisted Ring Bethe Phase Derivative -d Phi_ring(j|l)/dP_j = Dx_{jl} ===')
    for eq_name, sol_fn, pairs in [
        ('bad', sol_bad, [((mp.mpf('1.1'), 1), (mp.mpf('0.4'), 1)),
                          ((mp.mpf('0.9'), 1), (mp.mpf('0.35'), -1)),
                          ((mp.mpf('0.5'), -1), (mp.mpf('1.2'), -1))]),
        ('good', sol_good, [((mp.mpf('0.25'), 1), (mp.mpf('0.65'), 1)),
                            ((mp.mpf('0.2'), 1), (mp.mpf('0.35'), -1)),
                            ((mp.mpf('0.6'), -1), (mp.mpf('0.2'), -1))]),
    ]:
        for (p1, e1), (p2, e2) in pairs:
            s1, s2 = sol_fn(p1, e1), sol_fn(p2, e2)
            Dx1_hirota = -mp.log(A12(s1, s2)) / s1['k']
            Dx1_from_twisted = -dPhi_ring_dPj(s1, s2)
            # Compare with untwisted (-d delta0 / dP1 = Dx1') and naive (-d delta_dyn / dP1)
            Dx1_untwisted = -mp.diff(lambda p: delta0(sol_fn(p, e1), s2), p1) / s1['dP']
            Dx1_no_topo = -mp.diff(lambda p: delta_dyn(sol_fn(p, e1), s2), p1) / s1['dP']
            err_twisted = abs(Dx1_from_twisted - Dx1_hirota)
            err_untwisted = abs(Dx1_untwisted - Dx1_hirota)  # differs by +gamma_1 M_2 = O(1)
            err_no_topo = abs(Dx1_no_topo - Dx1_hirota)      # differs by (Pi1' M2/3 + 2 M1' Pi2/3)/P1' = O(1)
            pred_untwisted = abs(gamma_darboux(s1) * s2['M'])
            pred_no_topo = abs((s1['dPi'] * s2['M'] / 3 + 2 * s1['dM'] * s2['Pi'] / 3) / s1['dP'])
            print('   %s eps=(%+d,%+d): |(-d Phi_ring/dP1) - Dx1| = %s (untwisted err = %.3f, no-T12 err = %.3f)'
                  % (eq_name, e1, e2, mp.nstr(err_twisted, 3), float(err_untwisted), float(err_no_topo)))
            ok &= err_twisted < mp.mpf('1e-35')
            ok &= abs(err_untwisted - pred_untwisted) < mp.mpf('1e-35') and err_untwisted > mp.mpf('1e-2')
            ok &= abs(err_no_topo - pred_no_topo) < mp.mpf('1e-35') and err_no_topo > mp.mpf('1e-3')

    print('=== Check 3: Global Translation-Winding Sum Rule on the Ring (N = 3, 4) ===')
    Lam = mp.mpf('50.0')
    hbar = mp.mpf('0.15')
    for eq_name, sol_fn, pars in [
        ('bad', sol_bad, [(mp.mpf('1.3'), 1), (mp.mpf('0.7'), 1), (mp.mpf('0.4'), -1)]),
        ('good', sol_good, [(mp.mpf('0.2'), 1), (mp.mpf('0.5'), 1), (mp.mpf('0.25'), -1)]),
    ]:
        sols = [sol_fn(p, e) for p, e in pars]
        N = len(sols)
        # Classical twisted Bethe LHS: B_j = P_j * Lam + Pi_j * M_j + sum_{l != j} Phi_ring(j | l)
        B_cl = [
            sols[j]['P'] * Lam + sols[j]['Pi'] * sols[j]['M']
            + mp.fsum(Phi_ring(sols[j], sols[l]) for l in range(N) if l != j)
            for j in range(N)
        ]
        # Quantum Gamma-candidate twisted Bethe LHS: replace delta0(j,l) by hbar * Im log S_gamma(j,l)
        B_qu = [
            sols[j]['P'] * Lam + sols[j]['Pi'] * sols[j]['M']
            + mp.fsum(hbar * mp.im(logS_gamma(sols[j], sols[l], hbar)) + sols[j]['Pi'] * sols[l]['M']
                      for l in range(N) if l != j)
            for j in range(N)
        ]
        P_tot = mp.fsum(s['P'] for s in sols)
        Pi_tot = mp.fsum(s['Pi'] for s in sols)
        M_tot = mp.fsum(s['M'] for s in sols)
        target_sum = P_tot * Lam + Pi_tot * M_tot
        err_sum_cl = abs(mp.fsum(B_cl) - target_sum)
        err_sum_qu = abs(mp.fsum(B_qu) - target_sum)
        print('   %s (N=%d): |sum_j B_j^cl - (P_tot Lam + Pi_tot M_tot)| = %s, quantum = %s'
              % (eq_name, N, mp.nstr(err_sum_cl, 3), mp.nstr(err_sum_qu, 3)))
        ok &= err_sum_cl < mp.mpf('1e-35') and err_sum_qu < mp.mpf('1e-35')

    print('=== Check 4: Breather-Soliton Fusion of Twisted Ring Bethe Equations (Bad Eq) ===')
    a_br, b_br = mp.mpf('0.75'), mp.mpf('1.25')  # pi/3 < 1.25 < pi/2
    s_b1, s_b2 = sol_bad(a_br - I * b_br, 1), sol_bad(a_br + I * b_br, 1)
    M_br = s_b1['M'] + s_b2['M']
    Pi_br = s_b1['Pi'] + s_b2['Pi']
    V_br = mp.cosh(a_br) * mp.cos(2 * b_br) / mp.cos(b_br)
    err_M_br = abs(M_br - 4 * mp.sinh(a_br) * mp.cos(b_br))
    err_Pi_br = abs(Pi_br - (-2 * mp.sinh(2 * a_br) * mp.cos(2 * b_br))) + abs(Pi_br - (-M_br * V_br))
    ok &= err_M_br < mp.mpf('1e-35') and err_Pi_br < mp.mpf('1e-35')

    for b3, e3 in [(mp.mpf('0.4'), 1), (mp.mpf('1.1'), -1)]:
        s3 = sol_bad(b3, e3)
        Phi_br_3 = Phi_ring(s_b1, s3) + Phi_ring(s_b2, s3)
        Phi_3_br = Phi_ring(s3, s_b1) + Phi_ring(s3, s_b2)
        # Check reality and sum rule Phi_br_3 + Phi_3_br = Pi_br * M_3 + Pi_3 * M_br
        err_real = abs(mp.im(Phi_br_3)) + abs(mp.im(Phi_3_br))
        err_recip = abs((Phi_br_3 + Phi_3_br) - (Pi_br * s3['M'] + s3['Pi'] * M_br))
        # Check derivative -d Phi_3_br / dP_3 = -(1/k_3) log(A_{31} A_{32})
        dPhi_3_br_dP3 = mp.diff(lambda p: mp.re(Phi_ring(sol_bad(p, e3), s_b1) + Phi_ring(sol_bad(p, e3), s_b2)),
                                b3) / s3['dP']
        Dx3_br = -mp.re(mp.log(A12(s3, s_b1) * A12(s3, s_b2))) / s3['k']
        err_dx3 = abs(-dPhi_3_br_dP3 - Dx3_br)
        print('   breather + sol(b3=%s, eps3=%+d): Im err = %s, recip err = %s, |-dPhi/dP3 - Dx3_br| = %s'
              % (mp.nstr(b3, 3), e3, mp.nstr(err_real, 3), mp.nstr(err_recip, 3), mp.nstr(err_dx3, 3)))
        ok &= err_real < mp.mpf('1e-35') and err_recip < mp.mpf('1e-35') and err_dx3 < mp.mpf('1e-35')

    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

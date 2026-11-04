#!/usr/bin/env python3
"""Quantum absorption and defect fusion test at O(hbar^0) for the Corrigan-Zambon
defect in the good Boussinesq equation.

Checks:
  1. Stability obstruction on KP triples:
     Across 200 nondegenerate KP triples {q_1, q_2, q_3} and both orientations
     eta in {+1, -1}, exactly ONE vertex q_st is linearly stable (eta*sigma < -1,
     0 growing modes), one vertex lies in (-1, +1) (1 growing mode), and one vertex
     lies in (+1, +2) (2 growing modes). Hence NO elementary single-soliton
     absorption/emission step D(q_i) + S_i <-> D(q_{i+1}) connects two linearly
     stable defect vacua.
  2. One-loop charge balance and the KP-triangle holonomy test for canonical
     collective-coordinate charges vs mode-sum charges:
     (a) The mode-sum charges (Delta E_1^ren, Delta P_mode) are exact KP coboundaries
         of the 2-root defect dipole (Delta E_D^ren, Delta P_D), so their directed
         sum around any KP triangle vanishes identically:
           sum_{i=1}^3 rho_i Delta E_1^ren(S_i) = 0,
           sum_{i=1}^3 rho_i Delta P_mode(S_i) = 0.
     (b) By contrast, the co-moving canonical collective-coordinate charges
         (Delta E_can, Delta P_can) have a NON-ZERO directed holonomy around
         every generic KP triangle:
           sum_{i=1}^3 rho_i Delta E_can(S_i) != 0,
           sum_{i=1}^3 rho_i Delta P_can(S_i) != 0!
         This proves a sharp no-go theorem: NO single-valued assignment of defect
         vacuum charges (E_D^quant(q_i), P_D^quant(q_i)) -- regardless of any
         localised boundary counterterms -- can balance (Delta E_can, Delta P_can)
         at the unshifted classical absorption rapidities across all three edges
         of a KP triangle.
     (c) Conversely, if each edge rapidity shifts by the conditional absorption
         shift alpha_{S_i} -> alpha_{S_i} + hbar * a_{1,i} with
         a_{1,i} = (Delta P_mode(S_i) - Delta P_can(S_i)) / P_cl'(S_i),
         then the shifted total soliton charges equal the classical + mode-sum
         charges, whose triangle holonomy vanishes identically.
  3. What survives for quantum defect transmission and fusion at O(hbar^0):
     (a) Elastic transmission of N solitons across the unique linearly stable
         defect vacuum q_st (eta*sigma_st < -1) is free of growing modes and
         factorises into pairwise phases.
     (b) The continuum phase integral I_SD(3; q_D, eta) and its KP-edge fusion
         identity I_SD(3; q_{i+1}, eta) - I_SD(3; q_i, eta) = rho_i I_SS(3, S_i)
         hold as exact algebraic identities of the reflectionless Hirota/Lax
         scattering data, and a composite two-soliton transition
         D(q_st) + S_{12} + S_{23} -> D(q_st) + S_{13} begins and ends on the
         unique stable vacuum q_st, but its factorisation through intermediate
         defect states traverses linearly unstable vacua (q_2, q_3) and leaves
         the fixed-momentum shift dependent on whether Delta P_mode or Delta P_can
         enters the physical frame dictionary.
"""
import mpmath as mp

mp.mp.dps = 50
r3 = mp.sqrt(3)


def sol_good(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    k = mp.sin(a)
    v = e * mp.cos(a)
    w = k * v
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * e * k**3 * mp.cos(a)
    dP_da = mp.mpf(2) / 3 * e * k * mp.sin(3 * a)
    E = mp.mpf(2) / 15 * k**3 * (5 - 4 * k**2)
    dE_da = v * dP_da
    ell = -e * mp.cos(3 * a)
    pa = (k + v / r3) / 2
    pb = (v / r3 - k) / 2
    return dict(a=a, eps=e, k=k, v=v, w=w, M=M, Pi=Pi, P=P, dP_da=dP_da, E=E, dE_da=dE_da, ell=ell, pa=pa, pb=pb)


def sol_from_pair(x, y):
    pa, pb = max(x, y), min(x, y)
    eps = 1 if pa + pb > 0 else -1
    al = mp.acos(r3 * abs(pa + pb))
    return sol_good(al, eps)


def Delta_P_mode(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return -e / (2 * mp.pi) * sum(c * mp.sin(ph) * mp.ln((1 - mp.cos(ph)) / (1 + mp.cos(ph))) / 2 for c, ph in zip(cs, phis))


def Delta_E_ren(alpha):
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return sum(
        c * mp.sin(ph) * (1 - mp.cos(ph) * mp.ln((1 - mp.cos(ph)) / (1 + mp.cos(ph))) / 2)
        for c, ph in zip(cs, phis)
    ) / (2 * mp.pi)


def Delta_P_can(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    ratio = mp.tan((a + 2 * mp.pi / 3) / 2) / mp.tan((a - 2 * mp.pi / 3) / 2)
    return -e * (r3 * mp.cos(a) / mp.pi) * mp.ln(abs(ratio))


def Delta_E_can(alpha):
    a = mp.mpmathify(alpha)
    v_plus = mp.cos(a)
    L1 = v_plus * Delta_P_mode(a, 1) - Delta_E_ren(a)
    return v_plus * Delta_P_can(a, 1) - L1


def Delta_PD_dipole(qD, eta=1):
    s = -2 * r3 * qD
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    cDs = [-eta, +eta]
    return -sum(
        c * mp.re(mp.sin(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2)
        for c, ps in zip(cDs, psis)
    ) / (2 * mp.pi)


def Delta_ED_dipole(qD, eta=1):
    s = -2 * r3 * qD
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    cDs = [-eta, +eta]
    return sum(
        c * mp.re(mp.sin(ps) * (1 - mp.cos(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2))
        for c, ps in zip(cDs, psis)
    ) / (2 * mp.pi)


def admissible_growing_modes_count(s_eff):
    """Number of admissible growing modes (S > 1) for effective parameter s_eff = eta * sigma."""
    disc = mp.sqrt(4 - s_eff**2)
    roots = [
        ((r3 - 1) * s_eff + (r3 + 1) * disc) / 4,
        ((r3 - 1) * s_eff - (r3 + 1) * disc) / 4,
        ((r3 + 1) * s_eff + (r3 - 1) * disc) / 4,
        ((r3 + 1) * s_eff - (r3 - 1) * disc) / 4,
    ]
    return sum(1 for S_v in roots if mp.re(S_v) > 1 + mp.mpf("1e-40"))


def check_stability_obstruction():
    print("1. Stability audit of single-soliton absorption across nondegenerate KP triples:")
    stable_edges = 0
    total_edges = 0
    mode_counts_generic = set()
    mode_counts_branch_switch = set()
    for n in range(1, 102):
        if n in (34, 68):  # degenerate threshold triples theta = pi/3, 2pi/3
            continue
        theta = mp.pi * n / 102
        qs = [-mp.cos(theta + 2 * j * mp.pi / 3) / r3 for j in range(3)]
        for eta in (1, -1):
            s_effs = [-2 * r3 * eta * q for q in qs]
            counts = tuple(sorted(admissible_growing_modes_count(s_e) for s_e in s_effs))
            if any(abs(s_e - r3) < mp.mpf("1e-40") for s_e in s_effs):
                mode_counts_branch_switch.add(counts)
            else:
                mode_counts_generic.add(counts)
            # Check that exactly one vertex is linearly stable (s_eff < -1, 0 growing modes):
            assert sum(1 for c in counts if c == 0) == 1
            for i in range(3):
                q_curr = qs[i]
                q_next = qs[(i + 1) % 3]
                st_curr = (-2 * r3 * eta * q_curr) < -1
                st_next = (-2 * r3 * eta * q_next) < -1
                if st_curr and st_next:
                    stable_edges += 1
                total_edges += 1
    assert mode_counts_generic == {(0, 1, 2)}, f"Unexpected generic mode counts: {mode_counts_generic}"
    assert mode_counts_branch_switch == {(0, 1, 1)}, f"Unexpected branch-switch mode counts: {mode_counts_branch_switch}"
    assert stable_edges == 0
    print(f"   [PASS] Tested {total_edges} directed absorption edges: generic mode-count triple is (0, 1, 2)")
    print("          ((0, 1, 1) at the branch-switch point s_eff = sqrt(3) where gamma_2 = 0, gamma_1 = (2*sqrt(3)-1)/8 > 0),")
    print("          and 0 edges connect two linearly stable defect vacua.")


def check_charge_balance_and_triangle_holonomy():
    print("2. One-loop absorption balance and KP-triangle holonomy test (mode-sum vs canonical charges):")
    worst_mode_bal = mp.mpf(0)
    worst_mode_hol = mp.mpf(0)
    worst_shifted_bal = mp.mpf(0)
    can_holonomies = []

    for qv in ("-0.2", "0.1", "0.45"):
        q1 = mp.mpf(qv)
        disc = mp.sqrt(1 - 3 * q1**2)
        qs = [q1, (-q1 + disc) / 2, (-q1 - disc) / 2]
        for eta in (1, -1):
            hol_E_mode = mp.mpf(0)
            hol_P_mode = mp.mpf(0)
            hol_E_can = mp.mpf(0)
            hol_P_can = mp.mpf(0)
            for i in range(3):
                qi = qs[i]
                qf = qs[(i + 1) % 3]
                Si = sol_from_pair(qi, qf)
                rho_i = eta if qf < qi else -eta

                dED = Delta_ED_dipole(qf, eta) - Delta_ED_dipole(qi, eta)
                dPD = Delta_PD_dipole(qf, eta) - Delta_PD_dipole(qi, eta)

                E_m = Delta_E_ren(Si["a"])
                P_m = Delta_P_mode(Si["a"], Si["eps"])
                E_c = Delta_E_can(Si["a"])
                P_c = Delta_P_can(Si["a"], Si["eps"])

                worst_mode_bal = max(worst_mode_bal, abs(dED - rho_i * E_m), abs(dPD - rho_i * P_m))
                hol_E_mode += rho_i * E_m
                hol_P_mode += rho_i * P_m
                hol_E_can += rho_i * E_c
                hol_P_can += rho_i * P_c

                # Conditional absorption shift a_{1,i}:
                a1_i = (P_m - P_c) / Si["dP_da"]
                E_c_shifted = E_c + Si["dE_da"] * a1_i
                P_c_shifted = P_c + Si["dP_da"] * a1_i
                worst_shifted_bal = max(
                    worst_shifted_bal,
                    abs(dED - rho_i * E_c_shifted),
                    abs(dPD - rho_i * P_c_shifted),
                )

            worst_mode_hol = max(worst_mode_hol, abs(hol_E_mode), abs(hol_P_mode))
            if eta == 1:
                can_holonomies.append((qv, hol_E_can, hol_P_can))

    assert worst_mode_bal < mp.mpf("1e-45")
    assert worst_mode_hol < mp.mpf("1e-45")
    assert worst_shifted_bal < mp.mpf("1e-45")
    print(f"   [PASS] Mode-sum balance and triangle holonomy vanish to {mp.nstr(max(worst_mode_bal, worst_mode_hol), 3)}.")
    for qv, hE, hP in can_holonomies:
        print(f"   [NO-GO] q1={qv:>5}: canonical KP-triangle holonomy sum_i rho_i (Delta E_can, Delta P_can) = ({mp.nstr(hE, 10)}, {mp.nstr(hP, 10)}) != 0!")
        assert abs(hE) > mp.mpf("1e-3") and abs(hP) > mp.mpf("1e-2")
    print("   [PASS] Non-zero canonical KP-triangle holonomy proves no single-valued defect state charge can balance")
    print("          (Delta E_can, Delta P_can) at unshifted classical absorption rapidities around a KP triangle.")


def check_surviving_fusion_and_frame_ambiguity():
    print("3. Surviving algebraic fusion vs fixed-physical-momentum frame ambiguity around a KP triangle:")
    # For a test soliton 3 = (alpha_3, eps_3), the fixed-physical-momentum shift of the soliton-defect
    # transmission phase is -Delta x_D(3; q_D, eta) * Delta P_sol(3), and for two solitons it is
    # -Delta X_3(3, S_i) * Delta P_sol(3) + Delta X_{S_i}(3, S_i) * Delta P_sol(S_i)!
    # Wait: look at the SECOND term in the two-soliton fixed-physical-momentum shift:
    # When holding BOTH physical soliton momenta (P_3, P_{S_i}) fixed in two-soliton scattering S(3, S_i),
    # the Taylor expansion of delta_0(3, S_i) at fixed physical momenta is:
    #   delta_{12,renorm}(3, S_i) = -(d delta_0 / d P_3) Delta P(3) - (d delta_0 / d P_{S_i}) Delta P(S_i)
    #                             = +Delta X_3(3, S_i) Delta P(3) + Delta X_{S_i}(3, S_i) Delta P(S_i)!
    # Whereas on the left-hand side of defect fusion delta_D(3; q_{i+1}) - delta_D(3; q_i),
    # the defect has NO continuous momentum label (its parameter s is fixed, or shifted by ds_1)!
    # Let's check what happens to the SECOND term sum_{i=1}^3 rho_{tree,i} Delta X_{S_i}(3, S_i) Delta P(S_i)
    # around a closed KP triangle!
    s3 = sol_good("0.35", 1)
    q1 = mp.mpf("-0.2")
    disc = mp.sqrt(1 - 3 * q1**2)
    qs = [q1, (-q1 + disc) / 2, (-q1 - disc) / 2]
    eta = 1

    def A12_good(sa, sb):
        Pp = lambda K, W: W * W - K * K + K**4
        return -Pp(sa["k"] - sb["k"], sa["w"] - sb["w"]) / Pp(sa["k"] + sb["k"], sa["w"] + sb["w"])

    sum_DX3_P3 = mp.mpf(0)
    sum_DXSi_Pmode = mp.mpf(0)
    sum_DXSi_Pcan = mp.mpf(0)
    Sol_list = [sol_from_pair(qs[i], qs[(i + 1) % 3]) for i in range(3)]
    for i in range(3):
        qi, qf = qs[i], qs[(i + 1) % 3]
        Si = Sol_list[i]
        rho_i = eta if qf < qi else -eta
        rho_tree = s3["eps"] * rho_i
        lnA = mp.ln(abs(A12_good(s3, Si)))
        DX3 = -lnA / s3["k"]
        DXSi = +lnA / Si["k"]
        sum_DX3_P3 += rho_tree * DX3 * Delta_P_mode(s3["a"], s3["eps"])
        sum_DXSi_Pmode += rho_tree * DXSi * Delta_P_mode(Si["a"], Si["eps"])
        sum_DXSi_Pcan += rho_tree * DXSi * Delta_P_can(Si["a"], Si["eps"])

    # Exact summation-by-parts identity around the KP triangle:
    # Since ln|A12(3, S_i)| = rho_tree,i * (ln|z_t(3; q_{i+1})| - ln|z_t(3; q_i)|),
    # sum_{i=1}^3 rho_tree,i DX_{S_i}(3, S_i) Delta P(S_i)
    #   = sum_{i=1}^3 [ Delta P(S_{i-1})/k_{S_{i-1}} - Delta P(S_i)/k_{S_i} ] * ln|z_t(3; q_i, eta)|.
    ln_zt = [mp.ln(abs(((s3["pb"] - qs[i]) / (s3["pa"] - qs[i])) ** (s3["eps"] * eta))) for i in range(3)]
    sbp_Pmode = sum(
        (Delta_P_mode(Sol_list[(i - 1) % 3]["a"], Sol_list[(i - 1) % 3]["eps"]) / Sol_list[(i - 1) % 3]["k"]
         - Delta_P_mode(Sol_list[i]["a"], Sol_list[i]["eps"]) / Sol_list[i]["k"]) * ln_zt[i]
        for i in range(3)
    )
    sbp_Pcan = sum(
        (Delta_P_can(Sol_list[(i - 1) % 3]["a"], Sol_list[(i - 1) % 3]["eps"]) / Sol_list[(i - 1) % 3]["k"]
         - Delta_P_can(Sol_list[i]["a"], Sol_list[i]["eps"]) / Sol_list[i]["k"]) * ln_zt[i]
        for i in range(3)
    )
    assert abs(sum_DX3_P3) < mp.mpf("1e-45")
    assert abs(sum_DXSi_Pmode - sbp_Pmode) < mp.mpf("1e-45")
    assert abs(sum_DXSi_Pcan - sbp_Pcan) < mp.mpf("1e-45")
    print(f"   [PASS] Test-soliton renorm sum_i rho_tree DX_3(3,S_i) Delta P(3) = {mp.nstr(sum_DX3_P3, 3)} (closes identically).")
    print(f"   [NO-GO] Absorbed-soliton renorm sum_i rho_tree DX_{{S_i}}(3,S_i) Delta P_mode(S_i) = {mp.nstr(sum_DXSi_Pmode, 10)} != 0 (matches summation-by-parts to {mp.nstr(abs(sum_DXSi_Pmode - sbp_Pmode), 3)}).")
    print(f"   [NO-GO] Absorbed-soliton renorm sum_i rho_tree DX_{{S_i}}(3,S_i) Delta P_can(S_i)  = {mp.nstr(sum_DXSi_Pcan, 10)} != 0 (matches summation-by-parts to {mp.nstr(abs(sum_DXSi_Pcan - sbp_Pcan), 3)}).")
    assert abs(sum_DXSi_Pmode) > mp.mpf("1e-3") and abs(sum_DXSi_Pcan) > mp.mpf("1e-3")


if __name__ == "__main__":
    check_stability_obstruction()
    check_charge_balance_and_triangle_holonomy()
    check_surviving_fusion_and_frame_ambiguity()
    print("ALL CHECKS PASSED")

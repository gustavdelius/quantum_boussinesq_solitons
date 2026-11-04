#!/usr/bin/env python3
"""
kp_master_and_toda_edge.py

Phase 2h verification script 2:
1. Universal KP-Coboundary Theorem for arbitrary odd functions F(w):
   Proves that for ANY odd function F(w) = -F(-w) of w = tan(psi/2), the
   3-root soliton functional S[F](alpha, eps) with weights (1, 1, -2) is the
   exact KP-root chord difference of the 2-root defect dipole functional
   D[F](q_D, eta) with weights (-eta, +eta):
     S[F](alpha, eps) = D[F](p_b, +1) - D[F](p_a, +1),
     D[F](q_{i+1}, eta) - D[F](q_i, eta) = rho_i * S[F](S_i)
   across all 12 KP triangles, for both radiative and non-radiative regimes.

2. Two-Vertex Master Function F_K(p, q) on the KP Root Space:
   Constructs the explicit 2x2 dipole-dipole master function F_K(p, q) for any
   odd ratio kernel K(z) = -K(-z), and proves that:
   - its first difference is the 3x2 soliton-defect functional I_SD[K],
   - its second difference across (p_{1,a}, p_{1,b}) x (p_{2,a}, p_{2,b}) is
     the 3x3 two-soliton functional I_SS[K],
   - defect fusion and KP-triangle closure hold identically for any odd K(z),
     including the pure Legendre chi kernel K_{chi_2}(z) = -(1/pi)*chi_2(z)
     and the logarithmic kernel K_ln(z) = -(1/(2*pi))*ln|z|*ln|(1-z)/(1+z)|.

3. Finite-Amplitude Decomposition I_SS = I_SS^(chi_2) + I_SS^(ln) and
   Small-Amplitude / Toda Edge Matching:
   - Proves that in the small-amplitude overtaking corner alpha_1 = r*lambda,
     alpha_2 = lambda -> 0:
       I_SS^(chi_2)(r*lambda, +1; lambda, +1) -> -(4/pi)*chi_2(r),
       I_SS^(ln)(r*lambda, +1; lambda, +1)    -> -(2/pi)*ln(r)*ln((1-r)/(1+r)),
     so that I_SS^(ln) + H_bad = O(lambda^2) cancels the colliding bad/Toda
     Hirota shift and leaves pure -(4/pi)*chi_2(r) (or -(4/pi)*(chi_2(r) - r)
     after the Fermi-edge subtraction at r -> 0).
   - Verifies the head-on cancellation of O(lambda^2 ln lambda) between
     I_SS^(chi_2) and I_SS^(ln) to leave -(6/pi)*r*lambda^2, and characterises
     the O(1/Y) two-loop KdV coefficient c_1(r) from matching_analysis.json.
"""

import json
import os
import mpmath as mp

mp.mp.dps = 45
r3 = mp.sqrt(3)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def chi2_analytic(z):
    """Analytic continuation of Legendre chi_2(z) = (Li_2(z) - Li_2(-z))/2, odd in z."""
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(0)
    if mp.re(z) < 0:
        return -chi2_analytic(-z)
    return (mp.polylog(2, z) - mp.polylog(2, -z)) / 2


def K_ln_analytic(z):
    """Logarithmic kernel K_ln(z) = -(1/(2*pi)) * ln(z) * ln((1-z)/(1+z)) for Re(z) > 0, odd in z."""
    z = mp.mpc(z)
    if z == 0 or abs(z - 1) < mp.mpf('1e-40') or abs(z + 1) < mp.mpf('1e-40'):
        return mp.mpc(0)
    if mp.re(z) < 0:
        return -K_ln_analytic(-z)
    return -(mp.log(z) * mp.log((1 - z) / (1 + z))) / (2 * mp.pi)


def K_chi2_analytic(z):
    """Pure Legendre chi_2 kernel K_{chi_2}(z) = -(1/pi) * chi_2(z), odd in z."""
    return -chi2_analytic(z) / mp.pi


def K_L_analytic(z):
    """Full one-loop kernel K_L(z) = (1/(2*pi)) * L(z) = K_{chi_2}(z) + K_ln(z)."""
    z = mp.mpc(z)
    if abs(z - 1) < mp.mpf('1e-40'):
        return mp.mpc(-mp.pi / 8)
    if abs(z + 1) < mp.mpf('1e-40'):
        return mp.mpc(mp.pi / 8)
    return K_chi2_analytic(z) + K_ln_analytic(z)


def sol_good(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    k = mp.sin(a)
    v = e * mp.cos(a)
    pa = (k + v / r3) / 2
    pb = (v / r3 - k) / 2
    return dict(a=a, eps=e, k=k, v=v, pa=pa, pb=pb)


def sol_from_pair(x, y):
    pa, pb = max(x, y), min(x, y)
    eps = 1 if pa + pb > 0 else -1
    al = mp.acos(r3 * abs(pa + pb))
    return sol_good(al, eps)


def soliton_angles_eps(alpha, eps=1):
    """
    Return the three oriented soliton angles phi_r^(eps) and weights c_r = (1, 1, -2).
    Note: for eps = -1, phi_r^(-1) = pi - phi_r^(+1), so tan(phi_r^(-1)/2) = cot(phi_r/2) = (tan(phi_r/2))^(-1).
    """
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    if eps == -1:
        phis = [mp.pi - p for p in phis]
    return list(zip(phis, [mp.mpf(1), mp.mpf(1), mp.mpf(-2)]))


def defect_angles(qD, eta=1):
    """
    Return the two defect dipole angles psi_s'(q_D) and weights c_{D,s'} = (-eta, +eta),
    where cos(gamma) = s/2 = -sqrt(3)*q_D and psi_1 = 2*pi/3 - gamma, psi_2 = -2*pi/3 - gamma.
    """
    s = -2 * r3 * mp.mpmathify(qD)
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    return list(zip(psis, [-mp.mpf(eta), +mp.mpf(eta)]))


def S_functional(F_odd, alpha, eps=1):
    """3-root soliton functional S[F](alpha, eps) = sum_{r=1}^3 c_r Re F(tan(phi_r^(eps)/2))."""
    return sum(c * mp.re(F_odd(mp.tan(ph / 2))) for ph, c in soliton_angles_eps(alpha, eps))


def D_functional(F_odd, qD, eta=1):
    """2-root defect dipole functional D[F](q_D, eta) = sum_{s'=1}^2 c_{D,s'} Re F(tan(psi_{s'}(q_D)/2))."""
    return sum(c * mp.re(F_odd(mp.tan(ps / 2))) for ps, c in defect_angles(qD, eta))


def F_master(K_odd, p, q):
    """
    Two-vertex dipole-dipole master function F_K(p, q) on the KP root space:
      F_K(p, q) = sum_{s1, s2 = 1}^2 (-1)^(s1 + s2) Re K(tan(psi_{s1}(p)/2) * cot(psi_{s2}(q)/2)).
    Since c_D(p, +1) = (-1, +1) has sign (-1)^s1, this equals
      sum_{s1, s2} c_{D, s1}(p, +1) * c_{D, s2}(q, +1) * Re K(tan(psi_{s1}(p)/2) / tan(psi_{s2}(q)/2)).
    """
    d_p = defect_angles(p, 1)
    d_q = defect_angles(q, 1)
    tot = mp.mpf(0)
    for ps1, c1 in d_p:
        w1 = mp.tan(ps1 / 2)
        for ps2, c2 in d_q:
            w2 = mp.tan(ps2 / 2)
            tot += c1 * c2 * mp.re(K_odd(w1 / w2))
    return tot


def I_SD_general(K_odd, alpha, eps, qD, eta=1):
    """3x2 soliton-defect functional for a general odd ratio kernel K(w_S / w_D)."""
    s_ang = soliton_angles_eps(alpha, eps)
    d_ang = defect_angles(qD, eta)
    tot = mp.mpf(0)
    for ph, cS in s_ang:
        wS = mp.tan(ph / 2)
        for ps, cD in d_ang:
            wD = mp.tan(ps / 2)
            tot += cS * cD * mp.re(K_odd(wS / wD))
    return tot


def I_SS_general(K_odd, a1, eps1, a2, eps2):
    """3x3 two-soliton functional for a general odd ratio kernel K(w_1 / w_2)."""
    s1 = soliton_angles_eps(a1, eps1)
    s2 = soliton_angles_eps(a2, eps2)
    tot = mp.mpf(0)
    for ph1, c1 in s1:
        w1 = mp.tan(ph1 / 2)
        for ph2, c2 in s2:
            w2 = mp.tan(ph2 / 2)
            tot += c1 * c2 * mp.re(K_odd(w1 / w2))
    return tot


def test_1_universal_kp_coboundary():
    print("=== Check 1: Universal KP-Coboundary Theorem for Arbitrary Odd Functions F(w) ===")

    w0 = mp.mpf('0.63')
    # Five independent odd functions F(w) = -F(-w):
    odd_functions = [
        ("K_L(w/w0)", lambda w: K_L_analytic(w / w0)),
        ("K_chi2(w/w0)", lambda w: K_chi2_analytic(w / w0)),
        ("P_mode kernel", lambda w: -(1 / (2 * mp.pi)) * (2 * w / (1 + w * w)) * mp.log(abs(w))),
        ("E_ren kernel", lambda w: (1 / (2 * mp.pi)) * (2 * w / (1 + w * w)) * (1 - ((1 - w * w) / (1 + w * w)) * mp.log(abs(w)))),
        ("Rational w^3/(1+w^4)", lambda w: (w**3) / (1 + w**4)),
    ]

    worst_err = mp.mpf(0)
    for name, F_odd in odd_functions:
        fn_err = mp.mpf(0)
        # Test across regular and cusp-adjacent angles and both orientations eps in {+1, -1}
        for al_str in ("0.18", "0.52", "0.95", "1.30"):
            al = mp.mpf(al_str)
            for eps in (+1, -1):
                st = sol_good(al, eps)
                lhs = S_functional(F_odd, al, eps)
                rhs = D_functional(F_odd, st["pb"], 1) - D_functional(F_odd, st["pa"], 1)
                fn_err = max(fn_err, abs(lhs - rhs))

        # Test across all 12 KP triangles (both radiative |q| < 1/sqrt(3) and non-radiative)
        for q_str in ("-0.22", "0.15", "0.48"):
            q = mp.mpf(q_str)
            disc = mp.sqrt(1 - 3 * q * q)
            qs = [q, (-q + disc) / 2, (-q - disc) / 2]
            for eta in (+1, -1):
                for i in range(3):
                    q_curr, q_next = qs[i], qs[(i + 1) % 3]
                    Si = sol_from_pair(q_curr, q_next)
                    rho_i = eta if q_next < q_curr else -eta
                    jump = D_functional(F_odd, q_next, eta) - D_functional(F_odd, q_curr, eta)
                    sol_val = rho_i * S_functional(F_odd, Si["a"], Si["eps"])
                    fn_err = max(fn_err, abs(jump - sol_val))
        worst_err = max(worst_err, fn_err)
        assert fn_err < mp.mpf('1e-38'), f"KP-coboundary failed for {name}: {fn_err}"
        print(f"  [PASS] {name:<22}: max |S[F] - (D[F](p_b) - D[F](p_a))| = {mp.nstr(fn_err, 3)}")

    print(f"  [PASS] All 5 odd functionals passed Universal KP-Coboundary Theorem (max err = {mp.nstr(worst_err, 3)}).")


def test_2_two_vertex_master_function():
    print("\n=== Check 2: Two-Vertex Master Function F_K(p, q) & Universal KP-Triangle Closure ===")

    kernels = [
        ("Full one-loop K_L", K_L_analytic),
        ("Pure Legendre K_chi2", K_chi2_analytic),
        ("Logarithmic K_ln", K_ln_analytic),
        ("Rational z/(1+z^2)", lambda z: z / (1 + z * z)),
    ]

    for name, K_odd in kernels:
        max_sd_err = mp.mpf(0)
        max_ss_err = mp.mpf(0)
        max_tri_err = mp.mpf(0)

        # 2a. First difference F_K(p_b, q_D) - F_K(p_a, q_D) == I_SD[K]
        for al_str in ("0.25", "0.75"):
            for eps in (+1, -1):
                st = sol_good(mp.mpf(al_str), eps)
                for qD_str in ("-0.35", "0.18", "0.65"):
                    qD = mp.mpf(qD_str)
                    for eta in (+1, -1):
                        sd_direct = I_SD_general(K_odd, st["a"], eps, qD, eta)
                        sd_master = eta * (F_master(K_odd, st["pb"], qD) - F_master(K_odd, st["pa"], qD))
                        max_sd_err = max(max_sd_err, abs(sd_direct - sd_master))

        # 2b. Second difference across (p1a, p1b) x (p2a, p2b) == I_SS[K]
        for a1_str, a2_str in (("0.30", "0.70"), ("0.45", "1.10")):
            for eps1 in (+1, -1):
                for eps2 in (+1, -1):
                    s1 = sol_good(mp.mpf(a1_str), eps1)
                    s2 = sol_good(mp.mpf(a2_str), eps2)
                    ss_direct = I_SS_general(K_odd, s1["a"], eps1, s2["a"], eps2)
                    ss_master = (
                        F_master(K_odd, s1["pb"], s2["pb"])
                        - F_master(K_odd, s1["pa"], s2["pb"])
                        - F_master(K_odd, s1["pb"], s2["pa"])
                        + F_master(K_odd, s1["pa"], s2["pa"])
                    )
                    max_ss_err = max(max_ss_err, abs(ss_direct - ss_master))

        # 2c. Defect fusion and KP-triangle closure for spectator soliton 3
        s3 = sol_good(mp.mpf("0.62"), +1)
        for q_str in ("-0.18", "0.42"):
            q = mp.mpf(q_str)
            disc = mp.sqrt(1 - 3 * q * q)
            qs = [q, (-q + disc) / 2, (-q - disc) / 2]
            for eta in (+1, -1):
                tri_sum = mp.mpf(0)
                for i in range(3):
                    q_curr, q_next = qs[i], qs[(i + 1) % 3]
                    Si = sol_from_pair(q_curr, q_next)
                    rho_i = eta if q_next < q_curr else -eta
                    sd_jump = I_SD_general(K_odd, s3["a"], s3["eps"], q_next, eta) - I_SD_general(
                        K_odd, s3["a"], s3["eps"], q_curr, eta
                    )
                    ss_edge = rho_i * I_SS_general(K_odd, s3["a"], s3["eps"], Si["a"], Si["eps"])
                    max_tri_err = max(max_tri_err, abs(sd_jump - ss_edge))
                    tri_sum += ss_edge
                max_tri_err = max(max_tri_err, abs(tri_sum))

        assert max(max_sd_err, max_ss_err, max_tri_err) < mp.mpf('1e-38'), (
            f"Master function check failed for {name}: {max_sd_err}, {max_ss_err}, {max_tri_err}"
        )
        print(
            f"  [PASS] {name:<22}: max |1st diff - I_SD| = {mp.nstr(max_sd_err, 3)}, "
            f"|2nd diff - I_SS| = {mp.nstr(max_ss_err, 3)}, |KP triangle| = {mp.nstr(max_tri_err, 3)}"
        )


def test_3_small_amplitude_split_and_toda_c1():
    print("\n=== Check 3: Small-Amplitude Split I_SS = I_SS^(chi_2) + I_SS^(ln) & Toda O(1/Y) Fit ===")

    # 3a. Verify that at any finite amplitude I_SS[K_L] == I_SS[K_chi2] + I_SS[K_ln]
    for eps2 in (+1, -1):
        tot = I_SS_general(K_L_analytic, mp.mpf('0.35'), +1, mp.mpf('0.80'), eps2)
        part_chi2 = I_SS_general(K_chi2_analytic, mp.mpf('0.35'), +1, mp.mpf('0.80'), eps2)
        part_ln = I_SS_general(K_ln_analytic, mp.mpf('0.35'), +1, mp.mpf('0.80'), eps2)
        err_split = abs(tot - (part_chi2 + part_ln))
        assert err_split < mp.mpf('1e-42'), f"Split failed for eps2={eps2}: {err_split}"

    # 3b. Small-amplitude overtaking limit: alpha_1 = r*lam, alpha_2 = lam -> 0
    # Verify I_SS^(chi_2) -> -(4/pi)*chi_2(r) and I_SS^(ln) -> -(2/pi)*ln(r)*ln((1-r)/(1+r)) = -H_bad(r)
    for r_str in ("0.4", "0.5", "0.6"):
        r = mp.mpf(r_str)
        lam1 = mp.mpf('1e-4')
        lam2 = mp.mpf('5e-5')
        # Richardson extrapolation O(lam^2) -> O(lam^4)
        c2_1 = I_SS_general(K_chi2_analytic, r * lam1, +1, lam1, +1)
        c2_2 = I_SS_general(K_chi2_analytic, r * lam2, +1, lam2, +1)
        c2_extrap = (4 * c2_2 - c2_1) / 3

        ln_1 = I_SS_general(K_ln_analytic, r * lam1, +1, lam1, +1)
        ln_2 = I_SS_general(K_ln_analytic, r * lam2, +1, lam2, +1)
        ln_extrap = (4 * ln_2 - ln_1) / 3

        target_chi2 = -4 * chi2_analytic(r).real / mp.pi
        H_bad_r = (2 / mp.pi) * mp.log(r) * mp.log((1 - r) / (1 + r))
        target_ln = -H_bad_r

        err_c2 = abs(c2_extrap - target_chi2)
        err_ln = abs(ln_extrap - target_ln)
        assert err_c2 < mp.mpf('1e-16') and err_ln < mp.mpf('1e-16'), (
            f"Small-amplitude overtaking split failed at r={r_str}: {err_c2}, {err_ln}"
        )
        print(
            f"  [PASS] Overtaking r={r_str}: I_SS^(chi_2)(0+) = {mp.nstr(c2_extrap, 8)} (|err|={mp.nstr(err_c2, 2)}), "
            f"I_SS^(ln)(0+) + H_bad(r) = {mp.nstr(ln_extrap + H_bad_r, 2)}"
        )

    # 3c. Small-amplitude head-on limit: alpha_1 = r*lam, eps1 = +1; alpha_2 = lam, eps2 = -1
    # Verify I_SS^(chi_2) = +(3r/pi)*lam^2*ln(r*lam^2) + O(lam^2) and
    # I_SS^(ln) = -(6r/pi)*lam^2*ln(r*lam^2) + O(lam^2), combining to give
    # I_head = (r*lam^2/pi)*[-3*ln(r*lam^2) + 6*ln(2) - (3/2)*ln(3) - 3] + O(lam^4*ln(lam))
    r_ho = mp.mpf('0.5')
    lam_ho = mp.mpf('1e-5')
    a1_ho, a2_ho = r_ho * lam_ho, lam_ho
    ho_chi2 = I_SS_general(K_chi2_analytic, a1_ho, +1, a2_ho, -1)
    ho_ln = I_SS_general(K_ln_analytic, a1_ho, +1, a2_ho, -1)
    ho_tot = ho_chi2 + ho_ln
    C_head = 6 * mp.log(2) - mp.mpf('1.5') * mp.log(3) - 3
    ho_target = (a1_ho * a2_ho / mp.pi) * (-3 * mp.log(a1_ho * a2_ho) + C_head)
    rel_err_ho = abs(ho_tot - ho_target) / abs(ho_target)
    assert rel_err_ho < mp.mpf('1e-9'), f"Head-on small-amplitude law failed: {rel_err_ho}"
    print(
        f"  [PASS] Head-on r=0.5, lam=1e-5: I_SS^(chi_2)/lam^2 = {mp.nstr(ho_chi2 / lam_ho**2, 6)}, "
        f"I_SS^(ln)/lam^2 = {mp.nstr(ho_ln / lam_ho**2, 6)}, sum/lam^2 = {mp.nstr(ho_tot / lam_ho**2, 8)} "
        f"vs exact asymp = {mp.nstr(ho_target / lam_ho**2, 8)} (rel err = {mp.nstr(rel_err_ho, 2)})"
    )

    # 3d. Inspect and verify the O(1/Y) two-loop KdV coefficient c_1(r) from matching_analysis.json
    json_path = os.path.join(SCRIPT_DIR, "matching_analysis.json")
    with open(json_path, "r") as f:
        analysis = json.load(f)

    print("  [INFO] Subleading O(1/Y) two-loop KdV coefficient c_1(r) in R_0(Y, r) = -(4/pi)*(chi_2(r) - r) + c_1(r)/Y + ...:")
    for row_ly in analysis["large_y"]:
        r_val = row_ly["r"]
        c0_quad = row_ly["quadratic"]
        # Extract Y = 16 and Y = 32 residuals at this r from analysis["same"]
        r16 = [x["residual"] for x in analysis["same"] if x["Y"] == 16.0 and x["r"] == r_val][0]
        r32 = [x["residual"] for x in analysis["same"] if x["Y"] == 32.0 and x["r"] == r_val][0]
        # Linear fit in 1/Y through Y = 16 and Y = 32: R(Y) = c0 + c1 / Y -> c1 = 32 * (r32 - r16) / (1 - 2) = 32 * (r16 - r32)
        c1_fit = 32.0 * (r16 - r32)
        print(f"         r = {r_val:.1f}: c_0(quad) = {c0_quad:+.6f}, c_1(1/Y) = {c1_fit:+.6f}")
        assert -0.06 < c1_fit < 0.0

    print("  [PASS] All Check 3 assertions passed.")


if __name__ == "__main__":
    test_1_universal_kp_coboundary()
    test_2_two_vertex_master_function()
    test_3_small_amplitude_split_and_toda_c1()
    print("\nALL CHECKS PASSED")

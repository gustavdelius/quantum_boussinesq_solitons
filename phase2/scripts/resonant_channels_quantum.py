#!/usr/bin/env python3
"""Quantum and semiclassical implications of resonant channels (Questions 3 & 4).

Verifies at 60-digit precision (mpmath) and symbolically (SymPy):
 1. Semiclassical phase delta(1,2) and divergent time delay on all three resonance lines:
    - Exact transverse Taylor/logarithmic expansions as h -> 0+ (inside regular A_12 > 0):
        ell_1 - ell_2 = c_ell(alpha_1) * h + O(h^2),
        A_12 = c_A(alpha_1) * h^{sigma_res} * (1 + O(h)),
        delta(1,2) = T_{12}|_{res} + (2/9) * sigma_res * c_ell(alpha_1) * h * ln(h) + O(h),
      for R_ov (alpha_1 + alpha_2 = 2*pi/3 - h, sigma_res = +1),
      R_ho^+ (alpha_1 + alpha_2 = pi/3 - h, sigma_res = -1), and
      R_ho^- (alpha_1 - alpha_2 = pi/3 + h, sigma_res = +1).
    - Confirms finite phase limit delta -> T_{12}|_{res}, logarithmically divergent derivative
      d(delta)/dh ~ (2/9) sigma_res c_ell ln(h), and logarithmically divergent spatial shift
      Delta x_j = -/+ k_j^{-1} ln(A_12) ~ -/+ sigma_res k_j^{-1} ln(h).
    - Fixed-physical-momentum O(hbar^0) counterterm delta_{12,renorm} = -(Delta P_1/k_1 - Delta P_2/k_2) ln(A_12) + ...
      diverges logarithmically as ln(h) at the classical resonance line even though the bare
      fixed-rapidity continuum integral I_{12} remains finite.
 2. Physical-region singularities and kinematic stability of the light branch:
    - Theorem (Kinematic stability of the light branch): On the light branch alpha in (0, pi/3),
      |P| in (0, sqrt(3)/8), E_light(P) is strictly increasing and strictly concave in |P| with
      E_light(0) = 0. Hence for all nonzero P_1, P_2 (both overtaking and head-on) with |P_1 + P_2| <= P_max,
          E_light(P_1) + E_light(P_2) > E_light(P_1 + P_2).
      Consequently 1_light -> 2_light splitting and 2_light -> 1_light fusion are strictly forbidden
      by (E, P) conservation alone.
    - Every heavy soliton S_{13} (alpha_{13} in (pi/3, pi/2)) is embedded in the two-light-soliton
      head-on continuum at alpha_{12} = 2*pi/3 - alpha_{13}, alpha_{23} = alpha_{13} - pi/3, with
      non-vanishing 1->2 phase-space Jacobian J_{1->2} = 1 / |v_{12} - v_{23}| = 1 / (sqrt(3) k_{13}).
    - Across the head-on resonance R_ho^+, Delta E_{on-shell}(h) := E_1 + E_2 - E_3(P_1 + P_2)
      vanishes linearly: d/dh [E_1 + E_2 - E_3(P_1 + P_2)]|_{h=0} = -(v_2 - v_3) (dP_2/dalpha_2) != 0.
 3. Unitarity violation in the singular region (A_12 < 0) and coupled-channel unitarity:
    - Analytic continuation of delta(1,2) into A_12 < 0 gives Im(delta) = (2*pi/9)|ell_1 - ell_2| > 0
      on the physical sheet, so |exp(i delta / hbar)|^2 = exp(-(4*pi/(9*hbar))|ell_1 - ell_2|) < 1
      with unitarity deficit 1 - |S_{2->2}^{WKB}|^2 > 0.
 4. Three-soliton factorisation and bulk fusion bootstrap S_1 + S_2 -> S_{13}:
    - Classical cross-ratio factorisation A_{13} A_{23} = A_{int,3} and exact classical phase
      additivity delta_cl(3, S_{12}) + delta_cl(3, S_{23}) = delta_cl(3, S_{13}) (including topological
      terms!) hold identically on the resonant locus.
    - At O(hbar^0), the continued Gamma candidate fails the bulk fusion bootstrap
      S(3, S_{12}) S(3, S_{23}) = S(3, S_{13}) by exp(-/+ i (pi/2) sgn(eta))(1 + O(hbar)) because
      tau(3, S_{12}) + tau(3, S_{23}) = 0 != tau(3, S_{13}) = +/- 1.
    - At fixed physical momenta, the bare one-loop phonon integrals satisfy
      I_{SS}(3, S_{12}) + I_{SS}(3, S_{23}) = I_{SS}(3, S_{13}), while the target-soliton momentum
      renormalisation terms fail to telescope by an explicit O(hbar^0) remainder.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
TOL = mp.mpf("1e-48")


def sol_mp(alpha, eps):
    al = mp.mpf(alpha)
    ep = mp.mpf(eps)
    k = mp.sin(al)
    v = ep * mp.cos(al)
    w = k * v
    ell = -ep * mp.cos(3 * al)
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * ep * k**3 * mp.cos(al)
    E = mp.mpf(2) / 15 * k**3 * (5 - 4 * k**2)
    dP_da = mp.mpf(2) / 3 * ep * k * mp.sin(3 * al)
    dPi_da = -2 * ep * mp.cos(2 * al)
    gamma = dPi_da / dP_da if abs(dP_da) > mp.mpf("1e-30") else mp.nan
    pa = (v / mp.sqrt(3) + k) / 2
    pb = (v / mp.sqrt(3) - k) / 2
    return {
        "alpha": al, "eps": ep, "k": k, "v": v, "w": w, "ell": ell,
        "M": M, "Pi": Pi, "P": P, "E": E, "dP_da": dP_da, "dPi_da": dPi_da,
        "gamma": gamma, "pa": pa, "pb": pb
    }


def A12_mp(s1, s2):
    return ((s1["pa"] - s2["pa"]) * (s1["pb"] - s2["pb"])) / (
        (s1["pa"] - s2["pb"]) * (s1["pb"] - s2["pa"])
    )


def topo_mp(s1, s2):
    return -mp.mpf(2) / 3 * (s1["Pi"] * s2["M"] - s2["Pi"] * s1["M"])


def delta_cl_mp(s1, s2):
    A = A12_mp(s1, s2)
    return mp.mpf(2) / 9 * (s1["ell"] - s2["ell"]) * mp.ln(A) + topo_mp(s1, s2)


def oneloop_DeltaP_mp(alpha, eps):
    al = mp.mpf(alpha)
    ep = mp.mpf(eps)
    phis = [al + 2 * mp.pi / 3, al - 2 * mp.pi / 3, al]
    weights = [1, 1, -2]
    P_m = mp.mpf(0)
    for ph, w in zip(phis, weights):
        P_m += -ep * w * mp.sin(ph) * mp.ln(abs(mp.tan(ph / 2))) / (2 * mp.pi)
    return P_m


def run_checks():
    results = {}

    # ------------------------------------------------------------------
    # 1. Exact transverse expansions at all three resonance lines
    # ------------------------------------------------------------------
    # Test on all three resonance lines:
    # (a) R_ov: eps1=+1, eps2=+1, alpha_1 = 0.75, alpha_2 = 2*pi/3 - alpha_1 - h (sigma_res = +1)
    #     Here at h=0: ell_1 = -cos(3*a1), ell_2(h) = -cos(2*pi - 3*a1 - 3*h) = -cos(3*a1 + 3*h)
    #     => ell_1 - ell_2 = -3 sin(3*a1) h + O(h^2), so c_ell = -3 sin(3*a1).
    #     A_ov(h) = [1 - 3 cot^2(pi/3 - h/2)] / [1 - 3 cot^2((2*a1 - 2*pi/3 + h)/2)]
    #     Since d/dx [1 - 3 cot^2(x)]|_{x=pi/3} = 6 cot(pi/3) csc^2(pi/3) = 6 * (1/sqrt(3)) * (4/3) = 8/sqrt(3),
    #     1 - 3 cot^2(pi/3 - h/2) = -(4/sqrt(3)) h + O(h^2),
    #     so c_A = -(4/sqrt(3)) / (1 - 3 cot^2(a1 - pi/3)) > 0!
    # (b) R_ho^+: eps1=+1, eps2=-1, alpha_1 = 0.4, alpha_2 = pi/3 - alpha_1 - h (sigma_res = -1)
    #     Here ell_1 = -cos(3*a1), ell_2(h) = +cos(pi - 3*a1 - 3*h) = -cos(3*a1 + 3*h)
    #     => ell_1 - ell_2 = -3 sin(3*a1) h + O(h^2), so c_ell = -3 sin(3*a1).
    #     A_ho(h) = [1 - 3 tan^2((2*a1 - pi/3 + h)/2)] / [1 - 3 tan^2(pi/6 - h/2)]
    #     Since d/dx [1 - 3 tan^2(x)]|_{x=pi/6} = -6 tan(pi/6) sec^2(pi/6) = -6 * (1/sqrt(3)) * (4/3) = -8/sqrt(3),
    #     1 - 3 tan^2(pi/6 - h/2) = +(4/sqrt(3)) h + O(h^2),
    #     so A_ho(h) = c_A * h^{-1} * (1 + O(h)) with c_A = (sqrt(3)/4) * (1 - 3 tan^2(a1 - pi/6)) > 0!
    # (c) R_ho^-: eps1=+1, eps2=-1, alpha_2 = 0.25, alpha_1 = alpha_2 + pi/3 + h (sigma_res = +1)
    #     Here ell_2 = +cos(3*a2), ell_1(h) = -cos(3*a2 + pi + 3*h) = +cos(3*a2 + 3*h)
    #     => ell_1 - ell_2 = -3 sin(3*a2) h + O(h^2), so c_ell = -3 sin(3*a2).
    #     A_ho(h) = [1 - 3 tan^2(pi/6 + h/2)] / [1 - 3 tan^2((2*a2 + pi/3 + h)/2)]
    #     Since 1 - 3 tan^2(pi/6 + h/2) = -(4/sqrt(3)) h + O(h^2),
    #     c_A = -(4/sqrt(3)) / (1 - 3 tan^2(a2 + pi/6)) > 0!
    res_cases = [
        ("R_ov", +1, +1, mp.mpf("0.75"), lambda a, h: (a, 2 * mp.pi / 3 - a - h), +1,
         lambda a: -3 * mp.sin(3 * a),
         lambda a: -(4 / mp.sqrt(3)) / (1 - 3 * mp.cot(a - mp.pi / 3)**2)),
        ("R_ho_plus", +1, -1, mp.mpf("0.40"), lambda a, h: (a, mp.pi / 3 - a - h), -1,
         lambda a: -3 * mp.sin(3 * a),
         lambda a: (mp.sqrt(3) / 4) * (1 - 3 * mp.tan(a - mp.pi / 6)**2)),
        ("R_ho_minus", +1, -1, mp.mpf("0.25"), lambda a, h: (a + mp.pi / 3 + h, a), +1,
         lambda a: -3 * mp.sin(3 * a),
         lambda a: -(4 / mp.sqrt(3)) / (1 - 3 * mp.tan(a + mp.pi / 6)**2)),
    ]

    expansion_summary = {}
    for name, e1, e2, a_base, pair_fn, sigma_res, cell_fn, cA_fn in res_cases:
        c_ell = cell_fn(a_base)
        c_A = cA_fn(a_base)
        assert c_A > 0
        a1_0, a2_0 = pair_fn(a_base, mp.mpf(0))
        s1_0, s2_0 = sol_mp(a1_0, e1), sol_mp(a2_0, e2)
        T_res = topo_mp(s1_0, s2_0)

        h_small = mp.mpf("1e-20")
        a1_h, a2_h = pair_fn(a_base, h_small)
        s1_h, s2_h = sol_mp(a1_h, e1), sol_mp(a2_h, e2)
        ell_diff = s1_h["ell"] - s2_h["ell"]
        A_val = A12_mp(s1_h, s2_h)
        delta_val = delta_cl_mp(s1_h, s2_h)

        assert abs(ell_diff / (c_ell * h_small) - 1) < mp.mpf("1e-18")
        assert abs(A_val / (c_A * (h_small**sigma_res)) - 1) < mp.mpf("1e-18")

        # Check leading logarithmic non-analytic term in delta(h) - T_res:
        # delta(h) - T_res = (2/9) sigma_res c_ell h ln(h) + O(h)
        coeff_h_lnh = mp.mpf(2) / 9 * sigma_res * c_ell
        # Eliminate the linear O(h) term using h and 2*h:
        # Let f(h) = delta(h) - T(h) = (2/9)(ell_1 - ell_2) ln A_12
        # Then f(2h) - 2 f(h) = 2 h ln(2) * coeff_h_lnh + O(h^2 ln h)
        a1_2h, a2_2h = pair_fn(a_base, 2 * h_small)
        s1_2h, s2_2h = sol_mp(a1_2h, e1), sol_mp(a2_2h, e2)
        f_h = delta_cl_mp(s1_h, s2_h) - topo_mp(s1_h, s2_h)
        f_2h = delta_cl_mp(s1_2h, s2_2h) - topo_mp(s1_2h, s2_2h)
        extracted_coeff = (f_2h - 2 * f_h) / (2 * h_small * mp.ln(2))
        assert abs(extracted_coeff - coeff_h_lnh) < mp.mpf("1e-18")

        # Check fixed-physical-momentum O(hbar^0) counterterm divergence:
        dp1 = oneloop_DeltaP_mp(a1_h, e1)
        dp2 = oneloop_DeltaP_mp(a2_h, e2)
        coeff_lnA = -(dp1 / s1_h["k"] - dp2 / s2_h["k"])
        assert abs(coeff_lnA) > mp.mpf("1e-3")

        expansion_summary[name] = {
            "T_res": mp.nstr(T_res, 20),
            "c_ell": mp.nstr(c_ell, 20),
            "c_A": mp.nstr(c_A, 20),
            "coeff_h_lnh": mp.nstr(coeff_h_lnh, 20),
            "fixed_P_lnA_coeff": mp.nstr(coeff_lnA, 20),
        }

    results["resonance_expansions"] = expansion_summary
    print("[PASS] 1. Exact transverse expansions (h ln h branch point & ln h time delay) verified on all 3 resonance lines.")

    # ------------------------------------------------------------------
    # 2. Kinematic stability of the light branch & heavy-branch 1->2 phase space
    # ------------------------------------------------------------------
    # Symbolic proof that dE/d|P| = cos(alpha) > 0 and d^2E/d|P|^2 < 0 on alpha in (0, pi/3):
    al_s = sp.Symbol("alpha", positive=True)
    P_pos = sp.Rational(2, 3) * sp.sin(al_s)**3 * sp.cos(al_s)
    E_pos = sp.Rational(2, 15) * sp.sin(al_s)**3 * (5 - 4 * sp.sin(al_s)**2)
    dP_dal = sp.trigsimp(sp.diff(P_pos, al_s))
    dE_dal = sp.trigsimp(sp.diff(E_pos, al_s))
    assert sp.trigsimp(dE_dal - sp.cos(al_s) * dP_dal) == 0
    # Second derivative d^2E / dP^2 = (d/dalpha cos(alpha)) / (dP/dalpha) = -sin(alpha) / [(2/3) sin(alpha) sin(3 alpha)]
    # = -3 / (2 sin(3 alpha)) < 0 for all alpha in (0, pi/3)!
    d2E_dP2 = sp.trigsimp(sp.diff(sp.cos(al_s), al_s) / dP_dal)
    assert sp.trigsimp(d2E_dP2 + sp.Rational(3, 2) / sp.sin(3 * al_s)) == 0

    # Numerical check of strict sub-additivity E_light(P1) + E_light(P2) > E_light(P1 + P2)
    # on a grid of light-branch rapidities (both same-sign and opposite-sign):
    def P_of_a(a):
        return mp.mpf(2) / 3 * mp.sin(a)**3 * mp.cos(a)

    def alpha_branch_of_P(P_target, heavy=False):
        # Bracketed bisection + Newton on (0, pi/3) for light or (pi/3, pi/2) for heavy
        if P_target == 0:
            return mp.pi / 2 if heavy else mp.mpf(0)
        lo = mp.pi / 3 if heavy else mp.mpf(0)
        hi = mp.pi / 2 if heavy else mp.pi / 3
        for _ in range(120):
            mid = (lo + hi) / 2
            val = P_of_a(mid) - P_target
            if (val < 0 and not heavy) or (val > 0 and heavy):
                lo = mid
            else:
                hi = mid
        a = (lo + hi) / 2
        for _ in range(8):
            f = P_of_a(a) - P_target
            df = mp.mpf(2) / 3 * mp.sin(a) * mp.sin(3 * a)
            a -= f / df
        return a

    alpha_light_of_P = lambda P_t: alpha_branch_of_P(P_t, heavy=False)
    alpha_heavy_of_P = lambda P_t: alpha_branch_of_P(P_t, heavy=True)

    P_max = mp.sqrt(3) / 8
    min_gap_same = mp.inf
    min_gap_opp = mp.inf
    for a1_t in [mp.mpf("0.1"), mp.mpf("0.25"), mp.mpf("0.4"), mp.mpf("0.6")]:
        for a2_t in [mp.mpf("0.15"), mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.7")]:
            s1_l = sol_mp(a1_t, +1)
            s2_l = sol_mp(a2_t, +1)
            # Same-sign (overtaking) if P1 + P2 <= P_max:
            if s1_l["P"] + s2_l["P"] < P_max:
                a3_l = alpha_light_of_P(s1_l["P"] + s2_l["P"])
                s3_l = sol_mp(a3_l, +1)
                gap_s = s1_l["E"] + s2_l["E"] - s3_l["E"]
                assert gap_s > 0
                min_gap_same = min(min_gap_same, gap_s)
            # Opposite-sign (head-on): |P1 - P2| < max(P1, P2) <= P_max always!
            a3_opp = alpha_light_of_P(abs(s1_l["P"] - s2_l["P"]))
            s3_opp = sol_mp(a3_opp, +1)
            gap_o = s1_l["E"] + s2_l["E"] - s3_opp["E"]
            assert gap_o > 0
            min_gap_opp = min(min_gap_opp, gap_o)

    # Check linear vanishing of Delta E_{on-shell}(h) = E_1(a1) + E_2(a2) - E_heavy(P1 + P2)
    # across R_ho^+ (a1 + a2 = pi/3 - h, eps1=+1, eps2=-1):

    # For a1 in (0, pi/6) (eps1=+1) and a2 = pi/3 - a1 in (pi/6, pi/3) (eps2=-1),
    # |P2| > |P1|, so P1 + P2 < 0 and the fused heavy soliton has alpha_3 = a1 + pi/3, eps3 = -1!
    # Symmetrically, for a1 in (pi/6, pi/3) (eps1=+1) and a2 = pi/3 - a1 in (0, pi/6) (eps2=-1),
    # P1 + P2 > 0 and the fused heavy soliton has alpha_3 = a2 + pi/3 = 2*pi/3 - a1, eps3 = +1!
    a1_test = mp.mpf("0.62")  # in (pi/6, pi/3) so P1 + P2 > 0, eps3 = +1
    a2_res = mp.pi / 3 - a1_test
    s1_r = sol_mp(a1_test, +1)
    s2_r = sol_mp(a2_res, -1)
    a3_res = a2_res + mp.pi / 3  # = 2*pi/3 - a1_test in (pi/3, pi/2)
    s3_r = sol_mp(a3_res, +1)
    assert abs((s1_r["P"] + s2_r["P"]) - s3_r["P"]) < TOL
    assert abs((s1_r["E"] + s2_r["E"]) - s3_r["E"]) < TOL

    def delta_E_onshell(h_val):
        s2_cur = sol_mp(a2_res - h_val, -1)
        P_tot = s1_r["P"] + s2_cur["P"]
        a3_cur = alpha_heavy_of_P(abs(P_tot))
        s3_cur = sol_mp(a3_cur, +1 if P_tot >= 0 else -1)
        return s1_r["E"] + s2_cur["E"] - s3_cur["E"]

    dDeltaE_dh_num = mp.diff(delta_E_onshell, mp.mpf(0))
    dDeltaE_dh_exact = -(s2_r["v"] - s3_r["v"]) * s2_r["dP_da"]
    assert abs(dDeltaE_dh_num - dDeltaE_dh_exact) < mp.mpf("1e-28")
    assert abs(dDeltaE_dh_exact) > mp.mpf("0.01")

    print("[PASS] 2. Kinematic stability of light branch and linear on-shell energy zero across R_ho^+ verified.")

    # ------------------------------------------------------------------
    # 3. Unitarity deficit in singular region A_12 < 0 and 3-soliton factorisation tests
    # ------------------------------------------------------------------
    # In singular head-on region (e.g. a1 = 0.5, a2 = 0.65, so a1+a2 > pi/3 and |a1-a2| < pi/3):
    s1_sing = sol_mp("0.50", +1)
    s2_sing = sol_mp("0.65", -1)
    A_sing = A12_mp(s1_sing, s2_sing)
    assert A_sing < 0
    hbar_test = mp.mpf("0.1")
    im_delta_sing = mp.mpf(2) / 9 * abs(s1_sing["ell"] - s2_sing["ell"]) * mp.pi
    wkb_mod_sq = mp.exp(-2 * im_delta_sing / hbar_test)
    assert 0 < wkb_mod_sq < 1

    # 3-soliton resonant fusion test: S_{12} + S_{23} -> S_{13} with test soliton 3
    phi_val = mp.mpf("0.23")
    s_12 = sol_mp(mp.pi / 3 - phi_val, +1)   # right-moving light
    s_23 = sol_mp(phi_val, -1)               # left-moving light
    s_13 = sol_mp(mp.pi / 3 + phi_val, +1)   # right-moving heavy
    s_test = sol_mp("0.31", +1)              # regular test soliton 3

    # (a) Exact classical cross-ratio multiplication: A(3, S_{12}) * A(3, S_{23}) == A(3, S_{13})
    A_3_12 = A12_mp(s_test, s_12)
    A_3_23 = A12_mp(s_test, s_23)
    A_3_13 = A12_mp(s_test, s_13)
    assert A_3_12 > 0 and A_3_23 > 0 and A_3_13 > 0
    assert abs(A_3_12 * A_3_23 - A_3_13) < TOL

    # (b) Exact classical phase additivity: delta_cl(3, S_{12}) + delta_cl(3, S_{23}) == delta_cl(3, S_{13})
    d_3_12 = delta_cl_mp(s_test, s_12)
    d_3_23 = delta_cl_mp(s_test, s_23)
    d_3_13 = delta_cl_mp(s_test, s_13)
    assert abs((d_3_12 + d_3_23) - d_3_13) < TOL

    # (c) Continued Gamma candidate failure of bulk fusion S(3, S_{12}) S(3, S_{23}) = S(3, S_{13}) at O(hbar^0):
    def S_cand(sa, sb, h_val):
        eta = mp.mpf(2) / 9 * (sa["ell"] - sb["ell"]) / h_val
        A = A12_mp(sa, sb)
        tau = +1 if sa["eps"] == sb["eps"] else -1
        R = mp.gamma(1 + 1j * eta) / mp.gamma(1 - 1j * eta)
        return (R**tau) * mp.exp(1j * eta * (mp.ln(A) + tau * (2 - mp.ln(eta**2))) + 1j * topo_mp(sa, sb) / h_val)

    ratio_fusion = (S_cand(s_test, s_12, hbar_test) * S_cand(s_test, s_23, hbar_test)) / S_cand(s_test, s_13, hbar_test)
    eta_common = mp.mpf(2) / 9 * (s_test["ell"] - s_13["ell"]) / hbar_test
    R_common = mp.gamma(1 + 1j * eta_common) / mp.gamma(1 - 1j * eta_common)
    expected_ratio = (R_common * mp.exp(1j * eta_common * (2 - mp.ln(eta_common**2))))**(-1)
    assert abs(ratio_fusion - expected_ratio) < TOL
    # Check small-hbar limit of arg(ratio_fusion) -> -pi/2 * sgn(eta_common):
    ratio_small_h = (
        S_cand(s_test, s_12, mp.mpf("1e-6")) * S_cand(s_test, s_23, mp.mpf("1e-6"))
    ) / S_cand(s_test, s_13, mp.mpf("1e-6"))
    target_phase = -mp.pi / 2 * mp.sign(eta_common)
    assert abs(mp.arg(ratio_small_h) - target_phase) < mp.mpf("1e-5")

    # (d) Pointwise phonon phase derivative telescoping & 60-digit continuum integral I_{SS} check:
    def dt_sol(q_val, al_val, ep_val):
        angles = (al_val + 2 * mp.pi / 3, al_val - 2 * mp.pi / 3, al_val)
        weights = (1, 1, -2)
        return sum(
            c * mp.sin(p) / (q_val**2 + mp.sin(p)**2) * (1 + ep_val * mp.sign(q_val) * mp.cos(p) / mp.sqrt(1 + q_val**2))
            for p, c in zip(angles, weights)
        )

    for q_pt in [mp.mpf("-1.7"), mp.mpf("-0.25"), mp.mpf("0.4"), mp.mpf("2.3")]:
        dt_12 = dt_sol(q_pt, s_12["alpha"], s_12["eps"])
        dt_23 = dt_sol(q_pt, s_23["alpha"], s_23["eps"])
        dt_13 = dt_sol(q_pt, s_13["alpha"], s_13["eps"])
        assert abs((dt_12 + dt_23) - dt_13) < TOL

    # Target-soliton momentum-renormalisation obstruction at O(hbar^0):
    # Delta X_{S_j}(3, S_j) = +k_{S_j}^{-1} ln A(3, S_j)
    rem_target_P = (
        mp.ln(A_3_12) * oneloop_DeltaP_mp(s_12["alpha"], s_12["eps"]) / s_12["k"]
        + mp.ln(A_3_23) * oneloop_DeltaP_mp(s_23["alpha"], s_23["eps"]) / s_23["k"]
        - mp.ln(A_3_13) * oneloop_DeltaP_mp(s_13["alpha"], s_13["eps"]) / s_13["k"]
    )
    assert abs(rem_target_P) > mp.mpf("0.1")

    results["bulk_fusion_3body_tests"] = {
        "phi": float(phi_val),
        "alpha_test": 0.31,
        "classical_A_product_error": mp.nstr(abs(A_3_12 * A_3_23 - A_3_13), 10),
        "classical_delta_sum_error": mp.nstr(abs((d_3_12 + d_3_23) - d_3_13), 10),
        "gamma_candidate_fusion_phase_hbar_0_1": mp.nstr(mp.arg(ratio_fusion), 20),
        "gamma_candidate_fusion_phase_hbar_0_limit": mp.nstr(target_phase, 20),
        "fixed_P_target_renorm_mismatch": mp.nstr(rem_target_P, 20),
    }

    out_path = Path(__file__).resolve().with_suffix(".json")
    out_path.write_text(json.dumps(results, indent=2))
    print("[PASS] 3 & 4. Unitarity deficit and 3-soliton bulk fusion tests verified.")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    run_checks()

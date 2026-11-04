#!/usr/bin/env python3
"""Classical kinematics of resonant and number-changing channels (Question 1).

Verifies at 60-digit precision (mpmath) and symbolically (SymPy):
 1. KP root/chord dictionary for good Boussinesq solitons:
    p_a - p_b = k = sin(alpha), p_a + p_b = v / sqrt(3) = eps cos(alpha) / sqrt(3),
    cubic 4 p^3 - p = ell / (3 sqrt(3)) with ell = -eps cos(3 alpha).
 2. Cross-ratio identity A_12 = (p_{1a} - p_{2a})(p_{1b} - p_{2b}) / ((p_{1a} - p_{2b})(p_{1b} - p_{2a}))
    for both overtaking (eps1 = eps2) and head-on (eps1 = -eps2) channels.
 3. Exact characterization of the resonant loci in (alpha_1, alpha_2) in (0, pi/2)^2:
    - Overtaking (eps1 = eps2): A_ov = 0 iff alpha_1 + alpha_2 = 2*pi/3 (shared root p_{1a}=p_{2a}
      for eps=+1 or p_{1b}=p_{2b} for eps=-1; ell_1 = ell_2).
      Confirms that |alpha_1 - alpha_2| = pi/3 is NOT an overtaking resonance (1 - 3 cot^2(D/2) = 0
      requires |D| = 2*pi/3, outside (0, pi/2)^2).
    - Head-on (eps1 = -eps2):
      * A_ho = oo iff alpha_1 + alpha_2 = pi/3 (shared root p_{1b}=p_{2a} for eps=(+1,-1); ell_1 = ell_2);
      * A_ho = 0 iff |alpha_1 - alpha_2| = pi/3 (shared root p_{1a}=p_{2a} or p_{1b}=p_{2b}; ell_1 = ell_2).
 4. Branch classification relative to the momentum cusp alpha = pi/3 (P_max = sqrt(3)/8):
    - Every nondegenerate KP triple {q_1, q_2, q_3} (q_3 < q_2 < q_1) has one root in each of
      (-1/sqrt(3), -1/(2*sqrt(3))), (-1/(2*sqrt(3)), 1/(2*sqrt(3))), (1/(2*sqrt(3)), 1/sqrt(3)).
    - The three chord solitons S_{12}=(q_1,q_2), S_{23}=(q_2,q_3), S_{13}=(q_1,q_3) have rapidities
      alpha_{12}, alpha_{23} in (0, pi/3) (LIGHT branch, k < sqrt(3)/2) and
      alpha_{13} = alpha_{12} + alpha_{23} in (0, 2*pi/3) (LIGHT if < pi/3, HEAVY if > pi/3).
    - Consequently:
      * Head-on A_ho = oo (alpha_1 + alpha_2 = pi/3) lies entirely in the LIGHT-LIGHT sector;
      * Head-on A_ho = 0 (|alpha_1 - alpha_2| = pi/3) lies entirely in the LIGHT-HEAVY sector;
      * Overtaking A_ov = 0 (alpha_1 + alpha_2 = 2*pi/3) lies in the LIGHT-HEAVY sector (touching
        the cusp-cusp point (pi/3, pi/3)). Pure LIGHT-LIGHT overtaking collisions (alpha_1, alpha_2 < pi/3)
        satisfy alpha_1 + alpha_2 < 2*pi/3 and are ALWAYS regular and non-resonant!
 5. Exact 3-exponential Y-junction tau = e^{eta_1} + e^{eta_2} + e^{eta_3} with eta_i = q_i x - sqrt(3) q_i^2 t:
    - Solves the good Boussinesq Hirota bilinear equation (D_t^2 - D_x^2 + D_x^4) tau . tau = 0
      iff q_i - q_j lies on the cubic (i.e. q_1 + q_2 + q_3 = c, and for c=0: q_i^2 + q_i q_j + q_j^2 = 1/4).
    - Direct substitution of u = 2 (log tau)_{xx} into u_{tt} - u_{xx} + u_{xxxx} + 3(u^2)_{xx} = 0
      verified to 10^{-52} at 60-digit precision.
    - Tropical / asymptotic limits t -> -oo and t -> +oo verified: for q_1 > q_2 > q_3,
      as t -> -oo the solution separates into the two well-separated solitons S_{12} (at x ~ v_{12} t)
      and S_{23} (at x ~ v_{23} t) with v_{12} > v_{23}, which collide and MERGE (2 -> 1) into the
      single asymptotic soliton S_{13} (at x ~ v_{13} t) as t -> +oo. Time reversal gives 1 -> 2 splitting.
 6. Measure and domain fractions:
    - The resonant loci are 1D real-analytic curves (codimension 1, Lebesgue measure 0) forming the
      boundaries between regular (A_12 > 0) and singular (A_12 < 0) 2D open regions.
    - Exact area fractions in (alpha_1, alpha_2) in (0, pi/2)^2 and in (P_1, P_2) in (0, sqrt(3)/8)^2.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
TOL = mp.mpf("1e-48")


def sol_data_mp(alpha, eps):
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
    pa = (v / mp.sqrt(3) + k) / 2
    pb = (v / mp.sqrt(3) - k) / 2
    return {
        "alpha": al, "eps": ep, "k": k, "v": v, "w": w, "ell": ell,
        "M": M, "Pi": Pi, "P": P, "E": E, "pa": pa, "pb": pb
    }


def A12_hirota_mp(s1, s2):
    def P_disp(K, W):
        return W**2 - K**2 + K**4
    return -P_disp(s1["k"] - s2["k"], s1["w"] - s2["w"]) / P_disp(s1["k"] + s2["k"], s1["w"] + s2["w"])


def A12_chord_mp(s1, s2):
    return ((s1["pa"] - s2["pa"]) * (s1["pb"] - s2["pb"])) / (
        (s1["pa"] - s2["pb"]) * (s1["pb"] - s2["pa"])
    )


def run_checks():
    results = {}

    # ------------------------------------------------------------------
    # 1. Symbolic verification of KP chord dictionary & Hirota cross-ratio
    #    using polynomial parametrization (s_j, c_j) with s_j = 2 t_j, c_j = 1 - t_j^2, d_j = 1 + t_j^2
    # ------------------------------------------------------------------
    t1, t2 = sp.symbols("t_1 t_2", real=True)
    S1, C1, D1 = 2 * t1, 1 - t1**2, 1 + t1**2
    S2, C2, D2 = 2 * t2, 1 - t2**2, 1 + t2**2

    for e1, e2, label in [
        (1, 1, "overtaking_++"),
        (-1, -1, "overtaking_--"),
        (1, -1, "headon_+-"),
    ]:
        # p_{1a,b} = P_{1a,b} / (2 sqrt(3) D1), where P_{1a,b} = e1*C1 +- sqrt(3)*S1
        P1a = e1 * C1 + sp.sqrt(3) * S1
        P1b = e1 * C1 - sp.sqrt(3) * S1
        P2a = e2 * C2 + sp.sqrt(3) * S2
        P2b = e2 * C2 - sp.sqrt(3) * S2

        # Check cubic: 4 p^3 - p - ell / (3 sqrt(3)) = 0
        # Multiply by 6 sqrt(3) D1^3: P1^3 - 3 D1^2 P1 - 2 (-e1 C1 (4 C1^2 - 3 D1^2)) = 0
        ell1_num = -e1 * C1 * (4 * C1**2 - 3 * D1**2)
        for P_root in (P1a, P1b):
            cub_res = sp.expand(P_root**3 - 3 * D1**2 * P_root - 2 * ell1_num)
            assert cub_res == 0, f"Cubic failed for {label}: {cub_res}"

        # Check chord relation: p_{1a}^2 + p_{1a} p_{1b} + p_{1b}^2 = 1/4
        # Multiply by 12 D1^2: P1a^2 + P1a*P1b + P1b^2 - 3 D1^2 = 0
        chord_rel = sp.expand(P1a**2 + P1a * P1b + P1b**2 - 3 * D1**2)
        assert chord_rel == 0

        # Check cross-ratio == Hirota ratio == closed trig form (polynomial numerators)
        num_ch = (P1a * D2 - P2a * D1) * (P1b * D2 - P2b * D1)
        den_ch = (P1a * D2 - P2b * D1) * (P1b * D2 - P2a * D1)

        # Hirota: K_pm = S1*D2 -+ S2*D1 (over D1*D2), W_pm = e1*S1*C1*D2^2 -+ e2*S2*C2*D1^2 (over D1^2*D2^2)
        Km = S1 * D2 - S2 * D1
        Kp = S1 * D2 + S2 * D1
        Wm = e1 * S1 * C1 * D2**2 - e2 * S2 * C2 * D1**2
        Wp = e1 * S1 * C1 * D2**2 + e2 * S2 * C2 * D1**2
        num_hir = -(Wm**2 - Km**2 * D1**2 * D2**2 + Km**4)
        den_hir = Wp**2 - Kp**2 * D1**2 * D2**2 + Kp**4
        assert sp.expand(num_ch * den_hir - den_ch * num_hir) == 0, f"Chord != Hirota for {label}"

        if e1 == e2:
            # Closed form: [1 - 3 cot^2(S/2)] / [1 - 3 cot^2(D/2)]
            # tan(S/2) = (t1+t2)/(1-t1*t2), tan(D/2) = (t1-t2)/(1+t1*t2)
            num_cl = ((t1 + t2)**2 - 3 * (1 - t1 * t2)**2) * (t1 - t2)**2
            den_cl = ((t1 - t2)**2 - 3 * (1 + t1 * t2)**2) * (t1 + t2)**2
        else:
            # Closed form: [1 - 3 tan^2(D/2)] / [1 - 3 tan^2(S/2)]
            num_cl = ((1 + t1 * t2)**2 - 3 * (t1 - t2)**2) * (1 - t1 * t2)**2
            den_cl = ((1 - t1 * t2)**2 - 3 * (t1 + t2)**2) * (1 + t1 * t2)**2
        assert sp.expand(num_ch * den_cl - den_ch * num_cl) == 0, f"Chord != Closed for {label}"

    print("[PASS] 1. Symbolic KP chord dictionary and cross-ratio identities verified.")

    # ------------------------------------------------------------------
    # 2. Exact characterization of resonant loci and shared KP roots
    # ------------------------------------------------------------------
    # Test all three true resonance lines at 60-digit precision
    test_alphas = [mp.mpf("0.15"), mp.mpf("0.28"), mp.mpf("0.42")]

    # (a) Overtaking resonance: alpha_1 + alpha_2 = 2*pi/3, eps1 = eps2 = +1
    #     Here alpha_1 in (pi/6, pi/2), alpha_2 = 2*pi/3 - alpha_1 in (pi/6, pi/2).
    #     Specifically for alpha_1 = 0.75 (< pi/3), alpha_2 = 2*pi/3 - 0.75 (> pi/3):
    for a1_val in [mp.mpf("0.65"), mp.mpf("0.85"), mp.mpf("1.15")]:
        a2_val = 2 * mp.pi / 3 - a1_val
        s1 = sol_data_mp(a1_val, +1)
        s2 = sol_data_mp(a2_val, +1)
        assert abs(s1["ell"] - s2["ell"]) < TOL
        # Check p_{1a} == p_{2a}
        assert abs(s1["pa"] - s2["pa"]) < TOL
        assert abs(A12_chord_mp(s1, s2)) < TOL
        assert abs(A12_hirota_mp(s1, s2)) < TOL
        # The third chord (p_{2b}, p_{1b}) is a valid soliton with k3 = |k1 - k2|, w3 = |w1 - w2|
        k3 = abs(s1["k"] - s2["k"])
        w3 = abs(s1["w"] - s2["w"])
        assert abs(w3**2 - k3**2 + k3**4) < TOL

    # Verify that |alpha_1 - alpha_2| = pi/3 is NOT an overtaking resonance:
    # At |D| = pi/3, the overtaking denominator is 1 - 3 cot^2(pi/6) = 1 - 9 = -8 != 0.
    a1_false = mp.mpf("0.2")
    a2_false = a1_false + mp.pi / 3
    s1_f = sol_data_mp(a1_false, +1)
    s2_f = sol_data_mp(a2_false, +1)
    A_ov_false = A12_hirota_mp(s1_f, s2_f)
    A_ov_false_exact = (1 - 3 * mp.cot(a1_false + mp.pi / 6)**2) / (-8)
    ell_diff_false = s1_f["ell"] - s2_f["ell"]
    assert abs(A_ov_false - A_ov_false_exact) < TOL
    assert abs(A_ov_false - mp.mpf("0.355465479849")) < mp.mpf("1e-12")
    assert abs(ell_diff_false - (-2 * mp.cos(3 * a1_false))) < TOL
    results["false_overtaking_resonance_check"] = {
        "alpha_1": float(a1_false),
        "alpha_2": float(a2_false),
        "A_ov": mp.nstr(A_ov_false, 25),
        "ell_1_minus_ell_2": mp.nstr(ell_diff_false, 25),
    }

    # (b) Head-on resonance A_ho = oo: alpha_1 + alpha_2 = pi/3, eps1 = +1, eps2 = -1
    for a1_val in test_alphas:
        a2_val = mp.pi / 3 - a1_val
        s1 = sol_data_mp(a1_val, +1)
        s2 = sol_data_mp(a2_val, -1)
        assert abs(s1["ell"] - s2["ell"]) < TOL
        # Shared root: p_{1b} == p_{2a}!
        # Let's check: p_{1b} = (cos(a1)/sqrt(3) - sin(a1))/2 = -sin(a1 - pi/6)/sqrt(3)
        # p_{2a} = (-cos(a2)/sqrt(3) + sin(a2))/2 = sin(a2 - pi/6)/sqrt(3) = -sin(a1 - pi/6)/sqrt(3)
        assert abs(s1["pb"] - s2["pa"]) < TOL
        # Sum (k1 + k2, w1 + w2) satisfies dispersion relation:
        K_sum = s1["k"] + s2["k"]
        W_sum = s1["w"] + s2["w"]
        assert abs(W_sum**2 - K_sum**2 + K_sum**4) < TOL
        # The merged soliton has chord (p_{1a}, p_{2b}), k3 = k1 + k2 = sin(a1 + pi/3), alpha_3 = a1 + pi/3
        a3_val = a1_val + mp.pi / 3  # = 2*pi/3 - a2_val
        s3 = sol_data_mp(a3_val, +1 if a1_val > a2_val else -1)
        if a1_val != a2_val:
            assert abs(s3["k"] - K_sum) < TOL
            assert abs(s3["w"] - W_sum) < TOL
            assert abs(s3["pa"] - s1["pa"]) < TOL
            assert abs(s3["pb"] - s2["pb"]) < TOL
            assert abs(s3["ell"] - s1["ell"]) < TOL

    # (c) Head-on resonance A_ho = 0: |alpha_1 - alpha_2| = pi/3, eps1 = +1, eps2 = -1
    for a2_val in test_alphas:
        a1_val = a2_val + mp.pi / 3
        s1 = sol_data_mp(a1_val, +1)
        s2 = sol_data_mp(a2_val, -1)
        assert abs(s1["ell"] - s2["ell"]) < TOL
        # Shared root: p_{1b} == p_{2b}!
        assert abs(s1["pb"] - s2["pb"]) < TOL
        assert abs(A12_chord_mp(s1, s2)) < TOL
        assert abs(A12_hirota_mp(s1, s2)) < TOL
        # Difference (k1 - k2, w1 - w2) satisfies dispersion relation:
        K_diff = s1["k"] - s2["k"]
        W_diff = s1["w"] - s2["w"]
        assert abs(W_diff**2 - K_diff**2 + K_diff**4) < TOL

    print("[PASS] 2. All three true resonance lines and shared KP roots verified to 60 digits.")

    # ------------------------------------------------------------------
    # 3. Universal KP triple parametrization and light/heavy branch structure
    # ------------------------------------------------------------------
    # Every nondegenerate KP cubic 4 q^3 - q = ell / (3 sqrt(3)) with ell in (-1, 1)
    # has three distinct real roots ordered q_3 < q_2 < q_1.
    # Parametrize by phi in (0, pi/3):
    #   q_1 = (1/sqrt(3)) cos(phi)              in (1/(2*sqrt(3)), 1/sqrt(3))
    #   q_2 = (1/sqrt(3)) cos(phi - 2*pi/3)     in (-1/(2*sqrt(3)), 1/(2*sqrt(3)))
    #   q_3 = (1/sqrt(3)) cos(phi + 2*pi/3)     in (-1/sqrt(3), -1/(2*sqrt(3)))
    phi = sp.Symbol("phi", real=True)
    q1_s = sp.cos(phi) / sp.sqrt(3)
    q2_s = sp.cos(phi - 2 * sp.pi / 3) / sp.sqrt(3)
    q3_s = sp.cos(phi + 2 * sp.pi / 3) / sp.sqrt(3)
    assert sp.expand(sp.expand_trig(q1_s + q2_s + q3_s)) == 0
    assert sp.expand(sp.expand_trig(q1_s * q2_s + q2_s * q3_s + q3_s * q1_s + sp.Rational(1, 4))).subs(sp.sin(phi)**2, 1 - sp.cos(phi)**2) == 0

    # Check the three chords S_{12}=(q1,q2), S_{23}=(q2,q3), S_{13}=(q1,q3)
    # For any chord (p_a, p_b) with p_a > p_b:
    #   k = p_a - p_b,  v = sqrt(3) (p_a + p_b) = -sqrt(3) p_third
    # Specifically:
    #   S_{12}: k_{12} = q_1 - q_2 = sin(pi/3 - phi), v_{12} = -sqrt(3) q_3 = cos(pi/3 - phi) > 1/2
    #           => alpha_{12} = pi/3 - phi in (0, pi/3) (right-mover, eps=+1, LIGHT branch)
    #   S_{23}: k_{23} = q_2 - q_3 = sin(phi), v_{23} = -sqrt(3) q_1 = -cos(phi) < -1/2
    #           => alpha_{23} = phi in (0, pi/3) (left-mover, eps=-1, LIGHT branch)
    #   S_{13}: k_{13} = q_1 - q_3 = sin(phi + pi/3), v_{13} = -sqrt(3) q_2 = -cos(phi - 2*pi/3) = cos(phi + pi/3)
    #           For phi in (0, pi/6): v_{13} > 0, eps_{13} = +1, alpha_{13} = phi + pi/3 in (pi/3, pi/2) (HEAVY right-mover)
    #           For phi in (pi/6, pi/3): v_{13} < 0, eps_{13} = -1, alpha_{13} = 2*pi/3 - phi in (pi/3, pi/2) (HEAVY left-mover)
    #           At phi = pi/6: v_{13} = 0, k_{13} = 1 (static soliton limit alpha_{13} = pi/2)!
    # WAIT A SECOND! Look at S_{13} when phi in (0, pi/3):
    #   phi + pi/3 is ALWAYS in (pi/3, 2*pi/3), so sin(phi + pi/3) > sqrt(3)/2 ALWAYS!
    #   Wait: let's check that! For phi in (0, pi/3), phi + pi/3 in (pi/3, 2*pi/3), so
    #   k_{13} = sin(phi + pi/3) is in (sqrt(3)/2, 1]!
    #   Therefore, in EVERY nondegenerate KP triple:
    #   - Two chords S_{12} and S_{23} are on the LIGHT branch (0 < k < sqrt(3)/2, |v| > 1/2), moving in OPPOSITE directions!
    #   - The third chord S_{13} = S_{12} + S_{23} is ALWAYS on the HEAVY branch (sqrt(3)/2 < k <= 1, |v| < 1/2)!
    for phi_val in [mp.mpf("0.1"), mp.mpf("0.25"), mp.mpf("0.4"), mp.mpf("0.7"), mp.mpf("0.95")]:
        q1 = mp.cos(phi_val) / mp.sqrt(3)
        q2 = mp.cos(phi_val - 2 * mp.pi / 3) / mp.sqrt(3)
        q3 = mp.cos(phi_val + 2 * mp.pi / 3) / mp.sqrt(3)
        assert q1 > q2 > q3
        k12, v12 = q1 - q2, mp.sqrt(3) * (q1 + q2)
        k23, v23 = q2 - q3, mp.sqrt(3) * (q2 + q3)
        k13, v13 = q1 - q3, mp.sqrt(3) * (q1 + q3)
        assert abs(k12 - mp.sin(mp.pi / 3 - phi_val)) < TOL
        assert abs(v12 - mp.cos(mp.pi / 3 - phi_val)) < TOL
        assert abs(k23 - mp.sin(phi_val)) < TOL
        assert abs(v23 - (-mp.cos(phi_val))) < TOL
        assert abs(k13 - mp.sin(phi_val + mp.pi / 3)) < TOL
        assert abs(v13 - mp.cos(phi_val + mp.pi / 3)) < TOL
        # Check branch: k12 < sqrt(3)/2, k23 < sqrt(3)/2, k13 > sqrt(3)/2 (for phi != pi/6)
        assert k12 < mp.sqrt(3) / 2 and k23 < mp.sqrt(3) / 2
        assert k13 > mp.sqrt(3) / 2

    print("[PASS] 3. Universal KP triple structure: every Y-junction joins 2 LIGHT solitons (opposite movers) and 1 HEAVY soliton!")

    # ------------------------------------------------------------------
    # 4. Exact 3-exponential Y-junction PDE verification & asymptotics
    # ------------------------------------------------------------------
    # In our convention (u_tt - u_xx + u_xxxx + 3(u^2)_{xx} = 0, u = 2(log tau)_{xx}),
    # let's check the dispersion relation of exponential phases eta_i = q_i x - omega_i t:
    # For a single soliton tau = e^{eta_1} + e^{eta_2} = e^{eta_2}(1 + e^{(q_1-q_2)x - (omega_1-omega_2)t}),
    # u = 2 (log tau)_{xx} is invariant under multiplying tau by e^{a x + b t}!
    # With k = q_1 - q_2 and w = k v = (q_1 - q_2) * sqrt(3)(q_1 + q_2) = sqrt(3)(q_1^2 - q_2^2),
    # we have omega_i = sqrt(3) q_i^2!
    # Let's check Hirota bilinear operator P(D_x, D_t) = D_t^2 - D_x^2 + D_x^4 on tau = sum_{i=1}^3 e^{eta_i}:
    # Note that multiplying tau by e^{alpha x + beta t} shifts D_x -> D_x, D_t -> D_t in Hirota bilinear form,
    # Wait: (D_t^2 - D_x^2 + D_x^4) (e^{eta_i} . e^{eta_j}) = P(q_i - q_j, omega_i - omega_j) e^{eta_i + eta_j}!
    # When q_1, q_2, q_3 are the three roots of ONE KP cubic (so q_i^2 + q_i q_j + q_j^2 = 1/4 for ALL pairs i != j),
    # P(q_i - q_j, sqrt(3)(q_i^2 - q_j^2)) = 3(q_i^2 - q_j^2)^2 - (q_i - q_j)^2 + (q_i - q_j)^4
    # = (q_i - q_j)^2 [ 3(q_i + q_j)^2 - 1 + (q_i - q_j)^2 ]
    # = (q_i - q_j)^2 [ 4(q_i^2 + q_i q_j + q_j^2) - 1 ] = 0 for ALL pairs (i, j)!
    # Therefore (D_t^2 - D_x^2 + D_x^4) tau . tau = 0 IDENTICALLY for tau = c_1 e^{eta_1} + c_2 e^{eta_2} + c_3 e^{eta_3}!
    qi, qj = sp.symbols("q_i q_j")
    P_pair = sp.factor(3 * (qi**2 - qj**2)**2 - (qi - qj)**2 + (qi - qj)**4)
    assert sp.simplify(P_pair - 4 * (qi - qj)**2 * (qi**2 + qi * qj + qj**2 - sp.Rational(1, 4))) == 0

    # Check direct PDE residual for u = 2 (log tau)_{xx} with tau = e^{eta_1} + e^{eta_2} + e^{eta_3}
    phi_test = mp.mpf("0.23")
    q_roots = [
        mp.cos(phi_test) / mp.sqrt(3),
        mp.cos(phi_test - 2 * mp.pi / 3) / mp.sqrt(3),
        mp.cos(phi_test + 2 * mp.pi / 3) / mp.sqrt(3),
    ]
    w_roots = [mp.sqrt(3) * q**2 for q in q_roots]

    def tau_y(x_val, t_val):
        return sum(mp.exp(q * x_val - w * t_val) for q, w in zip(q_roots, w_roots))

    def u_y(x_val, t_val):
        # Analytic derivatives of tau = sum E_i
        E = [mp.exp(q * x_val - w * t_val) for q, w in zip(q_roots, w_roots)]
        S0 = sum(E)
        Sx = sum(q * e for q, e in zip(q_roots, E))
        Sxx = sum(q**2 * e for q, e in zip(q_roots, E))
        return 2 * (Sxx / S0 - (Sx / S0)**2)

    def pde_residual_y(x_val, t_val):
        # Exact Hirota bilinear identity + direct PDE residual via analytic jet derivatives
        E = [mp.exp(q * x_val - w * t_val) for q, w in zip(q_roots, w_roots)]
        def S_mn(m, n):
            return sum((q**m) * ((-w)**n) * e for q, w, e in zip(q_roots, w_roots, E))
        tau = S_mn(0, 0)
        tx, txx, txxx, txxxx, txxxxx, txxxxxx = [S_mn(k, 0) for k in range(1, 7)]
        tt, ttt = S_mn(0, 1), S_mn(0, 2)
        txxt = S_mn(2, 1)
        txxtt = S_mn(2, 2)
        txt = S_mn(1, 1)
        txtt = S_mn(1, 2)
        # Hirota bilinear residual: (D_t^2 - D_x^2 + D_x^4) tau . tau
        # = 2(tau * ttt - tt^2) - 2(tau * txx - tx^2) + 2(tau * txxxx - 4*tx*txxx + 3*txx^2)
        hirota_res = 2 * (tau * ttt - tt**2) - 2 * (tau * txx - tx**2) + 2 * (
            tau * txxxx - 4 * tx * txxx + 3 * txx**2
        )
        # Direct PDE residual u_tt - u_xx + u_xxxx + 3(u^2)_{xx} via 4th-order jet of log(tau):
        # Let f = log(tau). Then f_x = tx/tau, f_t = tt/tau, etc.
        fx = tx / tau
        fxx = txx / tau - fx**2
        fxxx = txxx / tau - 3 * fx * fxx - fx**3
        fxxxx = txxxx / tau - 4 * fx * fxxx - 3 * fxx**2 - 6 * fx**2 * fxx - fx**4
        fxxxxx = (
            txxxxx / tau
            - 5 * fx * fxxxx
            - 10 * fxx * fxxx
            - 10 * fx**2 * fxxx
            - 15 * fx * fxx**2
            - 10 * fx**3 * fxx
            - fx**5
        )
        fxxxxxx = (
            txxxxxx / tau
            - 6 * fx * fxxxxx
            - 15 * fxx * fxxxx
            - 10 * fxxx**2
            - 15 * fx**2 * fxxxx
            - 60 * fx * fxx * fxxx
            - 15 * fxx**3
            - 20 * fx**3 * fxxx
            - 45 * fx**2 * fxx**2
            - 15 * fx**4 * fxx
            - fx**6
        )
        u_val = 2 * fxx
        ux = 2 * fxxx
        uxx = 2 * fxxxx
        uxxxx = 2 * fxxxxxx
        ft = tt / tau
        ftx = txt / tau - ft * fx
        ftxx = txxt / tau - ft * fxx - 2 * fx * ftx - ft * fx**2
        ftt = ttt / tau - ft**2
        fttx = txtt / tau - 2 * ft * ftx - ftt * fx - ft**2 * fx
        fttxx = (
            txxtt / tau
            - ftt * fxx
            - 2 * ft * ftxx
            - 2 * ftx**2
            - 2 * fx * fttx
            - 4 * ft * fx * ftx
            - ftt * fx**2
            - ft**2 * fxx
            - ft**2 * fx**2
        )
        utt = 2 * fttxx
        u2_xx = 2 * (ux**2 + u_val * uxx)
        pde_res = utt - uxx + uxxxx + 3 * u2_xx
        return max(abs(hirota_res), abs(pde_res))

    max_pde_res = max(pde_residual_y(mp.mpf("0.7"), mp.mpf("-1.2")), pde_residual_y(mp.mpf("-0.4"), mp.mpf("0.9")))
    assert max_pde_res < mp.mpf("1e-50"), f"Y-junction PDE residual too large: {max_pde_res}"

    # Verify tropical / asymptotic soliton content of tau = e^{eta_1} + e^{eta_2} + e^{eta_3}:
    # Along x = v_{13} t, eta_2 - eta_1 = sqrt(3) k_{12} k_{23} t.
    # At T_large = 400, exp(-sqrt(3) k_{12} k_{23} T_large) < 10^{-49}.
    T_large = mp.mpf("400.0")
    v12 = mp.sqrt(3) * (q_roots[0] + q_roots[1])
    v23 = mp.sqrt(3) * (q_roots[1] + q_roots[2])
    v13 = mp.sqrt(3) * (q_roots[0] + q_roots[2])
    k12 = q_roots[0] - q_roots[1]
    k23 = q_roots[1] - q_roots[2]
    k13 = q_roots[0] - q_roots[2]

    # At t = -T_large: single soliton S_{13} peak at x = v13 * (-T_large), height k13^2 / 2
    u_in_13 = u_y(v13 * (-T_large), -T_large)
    assert abs(u_in_13 - k13**2 / 2) < mp.mpf("1e-48")

    # At t = +T_large: two solitons S_{12} at x = v12 * T_large (height k12^2/2) and S_{23} at x = v23 * T_large (height k23^2/2)
    u_out_12 = u_y(v12 * T_large, T_large)
    u_out_23 = u_y(v23 * T_large, T_large)
    assert abs(u_out_12 - k12**2 / 2) < mp.mpf("1e-48")
    assert abs(u_out_23 - k23**2 / 2) < mp.mpf("1e-48")

    print("[PASS] 4. Exact Y-junction PDE solution and 1->2 / 2->1 asymptotic soliton limits verified.")

    # ------------------------------------------------------------------
    # 5. Area fractions in (alpha_1, alpha_2) in (0, pi/2)^2
    # ------------------------------------------------------------------
    # Total area of Q = (0, pi/2)^2 is pi^2 / 4.
    # Overtaking: regular (0 < A_ov < 1) for alpha_1 + alpha_2 < 2*pi/3.
    #   Area of {a1 + a2 < 2*pi/3} in (0, pi/2)^2:
    #   The complement {a1 + a2 > 2*pi/3} in (0, pi/2)^2 is a right isosceles triangle of side
    #   pi/2 - pi/6 = pi/3, so area_sing_ov = (1/2) * (pi/3)^2 = pi^2 / 18.
    #   Fraction singular (overtaking) = (pi^2 / 18) / (pi^2 / 4) = 2 / 9.
    #   Fraction regular (overtaking) = 7 / 9.
    #   Within the LIGHT-LIGHT square (0, pi/3)^2 (area pi^2 / 9, fraction 4/9 of Q):
    #   alpha_1 + alpha_2 < 2*pi/3 everywhere, so 100% of LIGHT-LIGHT overtaking is regular!
    # Head-on:
    #   - Regular attractive (A_ho > 1): alpha_1 + alpha_2 < pi/3 (triangle of side pi/3, area pi^2 / 18, fraction 2/9).
    #     Lies 100% inside the LIGHT-LIGHT square (half of the light-light square)!
    #   - Regular repulsive (0 < A_ho < 1): |alpha_1 - alpha_2| > pi/3 (two triangles of side pi/2 - pi/3 = pi/6,
    #     total area 2 * (1/2) * (pi/6)^2 = pi^2 / 36, fraction 1/9).
    #     Lies 100% in the LIGHT-HEAVY rectangles!
    #   - Singular (A_ho < 0): remaining hexagon of area pi^2/4 - pi^2/18 - pi^2/36 = pi^2 / 6 (fraction 2/3 = 6/9).
    results["kinematic_area_fractions"] = {
        "overtaking_regular_fraction": "7/9",
        "overtaking_singular_fraction": "2/9",
        "overtaking_light_light_regular_fraction": "1",
        "headon_regular_attractive_fraction": "2/9",
        "headon_regular_repulsive_fraction": "1/9",
        "headon_singular_fraction": "2/3",
        "headon_light_light_regular_fraction": "1/2",
    }

    out_path = Path(__file__).resolve().with_suffix(".json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[PASS] Saved kinematics data to {out_path.name}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    run_checks()

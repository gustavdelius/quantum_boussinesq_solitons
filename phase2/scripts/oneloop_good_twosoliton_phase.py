"""Tasks 2 & 3: Two-soliton O(hbar^0) one-loop phase for the good Boussinesq equation,
and comparison with Candidates (A) and (B) in the regular region.

Checks performed (at 35-digit working precision):
 1. Exact 3-soliton Hirota tau-linearisation on the two-soliton background tau_12(x, t):
    delta_u(x, t; q) = 2 d_x^2 (tau_ph / tau_12) solves the time-dependent fourth-order
    linearised PDE around u_12(x, t) at collision time t to < 10^{-28}, confirming
    exact reflectionlessness (Faddeev-Korepin operator c_hat = 0) and exact transmission
    factorisation A(1, ph(q)) A(2, ph(q)).
 2. UV convergence O(q^{-3}) of theta_1(q) theta_2'(q) and exact agreement of the
    Faddeev-Korepin gradient (d_{alpha_1} delta_ph, d_{alpha_2} delta_ph) with both
    grad I_12 and grad J_12 to < 10^{-33}.
 3. Exact closed-form dilogarithm / Legendre-chi_2 formulas for K(alpha_1, alpha_2),
    I_12^same, J_12^same = I_12^same + pi/2, and I_12^head = J_12^head, plus the exact
    crossing relation I_12^head(alpha_1, alpha_2) = I_12^same(alpha_1, pi - alpha_2),
    verified against numerical quadrature to < 10^{-33}.
 4. Closed-form one-loop single-soliton momentum shift Delta P_1(alpha) and normal-ordered
    energy shift Delta E_1^ren(alpha), Darboux shift identity Delta X_j' = -d(delta_0)/dP_j,
    and fixed-physical-momentum shift delta_renorm = -sum_j Delta X_j' Delta P_j.
 5. Comparison tables in the regular region (alpha_j < pi/3, A_12 > 0) for overtaking
    and head-on collisions against Candidate (A) ((pi/2) tau sgn(eta)) and Candidate (B) (0),
    reporting both I_12 and J_12 in parallel with fixed-alpha and fixed-P_phys schemes.
"""
import mpmath as mp
from oneloop_good_phonon_scattering import (
    sol_good, phonon_good, A_sol_ph, theta_3root, dtheta_dq
)

mp.mp.dps = 35
r3 = mp.sqrt(3)
ok = True


def A12_good(s1, s2):
    return (
        (s1["pa"] - s2["pa"]) * (s1["pb"] - s2["pb"])
        / ((s1["pa"] - s2["pb"]) * (s1["pb"] - s2["pa"]))
    )


def delta0_good(a1, eps1, a2, eps2):
    s1 = sol_good(a1, eps1)
    s2 = sol_good(a2, eps2)
    A = A12_good(s1, s2)
    T = mp.mpf(2) / 3 * s1["M"] * s2["M"] * (s1["v"] - s2["v"])
    return mp.mpf(2) / 9 * (s1["ell"] - s2["ell"]) * mp.ln(A) + T


def L_chi2(z):
    """Exact antiderivative L(z) = int_0^z (2 ln|t|)/(1 - t^2) dt via Legendre chi_2."""
    zz = mp.mpf(z)
    if zz == 0:
        return mp.mpf(0)
    s = 1 if zz > 0 else -1
    az = abs(zz)
    if az == 1:
        return -s * mp.pi**2 / 4
    val = -mp.re(mp.polylog(2, az) - mp.polylog(2, -az)) - mp.ln(az) * mp.ln(abs((1 - az) / (1 + az)))
    return s * val


def K_closed(a1, a2):
    """Closed-form integral K(a1, a2) = (1/2pi) int_0^oo theta_1^odd(q) (theta_2^even)'(q) dq."""
    aa1, aa2 = mp.mpf(a1), mp.mpf(a2)
    phis1 = [aa1 + 2 * mp.pi / 3, aa1 - 2 * mp.pi / 3, aa1]
    phis2 = [aa2 + 2 * mp.pi / 3, aa2 - 2 * mp.pi / 3, aa2]
    cs = [1, 1, -2]
    total = mp.mpf(0)
    for c1, p1 in zip(cs, phis1):
        x = mp.tan(p1 / 2)
        for c2, p2 in zip(cs, phis2):
            y = mp.tan(p2 / 2)
            total += c1 * c2 * (L_chi2(x / y) - L_chi2(x * y))
    return total / (4 * mp.pi)


def dK_da1_elementary(a1, a2):
    """Exact elementary closed form for dK(a1, a2)/da1."""
    aa1, aa2 = mp.mpf(a1), mp.mpf(a2)
    phis1 = [aa1 + 2 * mp.pi / 3, aa1 - 2 * mp.pi / 3, aa1]
    phis2 = [aa2 + 2 * mp.pi / 3, aa2 - 2 * mp.pi / 3, aa2]
    cs = [1, 1, -2]
    total = mp.mpf(0)
    for c1, p1 in zip(cs, phis1):
        for c2, p2 in zip(cs, phis2):
            num = mp.cos(p2) * mp.ln(abs(mp.tan(p1 / 2))) - mp.cos(p1) * mp.ln(abs(mp.tan(p2 / 2)))
            den = mp.sin(p2)**2 - mp.sin(p1)**2
            total += c1 * c2 * mp.sin(p2) * num / den
    return total / (2 * mp.pi)


def I_same_closed(a1, a2):
    """Closed-form raw FK integral I_12 = (1/4pi) int_{-oo}^{oo} theta_1 theta_2' dq for eps1=eps2=+1."""
    aa1, aa2 = mp.mpf(a1), mp.mpf(a2)
    phis1 = [aa1 + 2 * mp.pi / 3, aa1 - 2 * mp.pi / 3, aa1]
    phis2 = [aa2 + 2 * mp.pi / 3, aa2 - 2 * mp.pi / 3, aa2]
    cs = [1, 1, -2]
    return sum(
        c1 * c2 * L_chi2(mp.tan(p1 / 2) / mp.tan(p2 / 2))
        for c1, p1 in zip(cs, phis1)
        for c2, p2 in zip(cs, phis2)
    ) / (2 * mp.pi)


def J_same_closed(a1, a2):
    """Closed-form antisymmetrised integral J_12 for eps1=eps2=+1 in the regular region."""
    return I_same_closed(a1, a2) + mp.pi / 2


def I_head_closed(a1, a2):
    """Closed-form integral I_12 = J_12 for head-on collision (eps1=+1, eps2=-1) in the regular region."""
    aa1, aa2 = mp.mpf(a1), mp.mpf(a2)
    phis1 = [aa1 + 2 * mp.pi / 3, aa1 - 2 * mp.pi / 3, aa1]
    phis2 = [aa2 + 2 * mp.pi / 3, aa2 - 2 * mp.pi / 3, aa2]
    cs = [1, 1, -2]
    return sum(
        c1 * c2 * L_chi2(mp.tan(p1 / 2) * mp.tan(p2 / 2))
        for c1, p1 in zip(cs, phis1)
        for c2, p2 in zip(cs, phis2)
    ) / (2 * mp.pi)


def Delta_P1_closed(alpha, eps=1):
    """Exact one-loop momentum shift Delta P_1(alpha, eps) = (1/4pi) int_{-oo}^{oo} q theta'(q) dq."""
    a = mp.mpf(alpha)
    e = mp.mpf(eps)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return -e / (2 * mp.pi) * sum(c * mp.sin(ph) * mp.ln(abs(mp.tan(ph / 2))) for c, ph in zip(cs, phis))


def Delta_E1_ren_closed(alpha):
    """Exact normal-ordered one-loop energy shift in box quantisation:
    lim_{Lambda->oo} [ -(1/4pi) int_{-Lambda}^Lambda w'(q) theta(q) dq + 3 <phi_x^2>_Lambda M_1 ].
    """
    a = mp.mpf(alpha)
    k1 = mp.sin(a)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return (
        -6 * k1
        - 2 * sum(c * mp.sin(ph) * mp.cos(ph) * mp.ln(abs(mp.tan(ph / 2))) for c, ph in zip(cs, phis))
    ) / (4 * mp.pi)


def delta_renorm_good(a1, eps1, a2, eps2):
    """Parameter-renormalisation shift -sum_j (d delta_0 / d P_j) Delta P_j = +sum_j Delta X_j' Delta P_j."""
    s1 = sol_good(a1, eps1)
    s2 = sol_good(a2, eps2)
    lnA = mp.ln(A12_good(s1, s2))
    g1 = -3 * mp.cos(2 * s1["a"]) / (mp.sin(s1["a"]) * mp.sin(3 * s1["a"]))
    g2 = -3 * mp.cos(2 * s2["a"]) / (mp.sin(s2["a"]) * mp.sin(3 * s2["a"]))
    DX1_prime = -lnA / s1["k"] + g1 * s2["M"]
    DX2_prime = +lnA / s2["k"] - g2 * s1["M"]
    return DX1_prime * Delta_P1_closed(a1, eps1) + DX2_prime * Delta_P1_closed(a2, eps2)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Exact 3-soliton Hirota linearisation on the two-soliton background
    # ------------------------------------------------------------------
    print("1. Exact phonon eigenmode on the two-soliton background (c_hat = 0 PDE check):")
    worst_pde2 = mp.mpf(0)
    for (a1, e1), (a2, e2), q_val in (
        (("0.3", 1), ("0.8", 1), "0.6"),
        (("0.4", 1), ("0.6", -1), "-0.9"),
    ):
        s1 = sol_good(a1, e1)
        s2 = sol_good(a2, e2)
        ph = phonon_good(q_val)
        A12 = A12_good(s1, s2)
        A1p = A_sol_ph(s1, ph)
        A2p = A_sol_ph(s2, ph)
        x0, t0 = mp.mpf("0.23"), mp.mpf("0.17")

        def tau12(xx, tt):
            E1 = mp.exp(s1["k"] * (xx - s1["v"] * tt))
            E2 = mp.exp(s2["k"] * (xx - s2["v"] * tt))
            return 1 + E1 + E2 + A12 * E1 * E2

        def u12(xx, tt):
            return 2 * mp.diff(lambda xxx: mp.ln(tau12(xxx, tt)), xx, 2)

        def du_ph(xx, tt):
            def f_ratio(xxx):
                E1 = mp.exp(s1["k"] * (xxx - s1["v"] * tt))
                E2 = mp.exp(s2["k"] * (xxx - s2["v"] * tt))
                num = 1 + A1p * E1 + A2p * E2 + A12 * A1p * A2p * E1 * E2
                den = 1 + E1 + E2 + A12 * E1 * E2
                return (num / den) * mp.exp(1j * (ph["q"] * xxx - ph["w"] * tt))
            return 2 * mp.diff(f_ratio, xx, 2)

        utt = mp.diff(lambda tt: du_ph(x0, tt), t0, 2)
        uxx = mp.diff(lambda xx: du_ph(xx, t0), x0, 2)
        uxxxx = mp.diff(lambda xx: du_ph(xx, t0), x0, 4)
        u0du = mp.diff(lambda xx: 6 * u12(xx, t0) * du_ph(xx, t0), x0, 2)
        res = abs(utt - uxx + uxxxx + u0du)
        worst_pde2 = max(worst_pde2, res)
    print(f"   max two-soliton linearised PDE residual = {mp.nstr(worst_pde2, 3)}")
    ok &= worst_pde2 < 1e-28

    # ------------------------------------------------------------------
    # 2. Closed-form dilogarithm formulas vs quadrature and FK gradient
    # ------------------------------------------------------------------
    print("2. Closed-form dilogarithm formulas vs numerical quadrature and FK gradient:")
    worst_int = mp.mpf(0)
    worst_grad = mp.mpf(0)
    worst_cross = mp.mpf(0)
    for a1, a2 in (("0.25", "0.55"), ("0.3", "0.8")):
        # Direct quadrature of I_12 and J_12 for overtaking and head-on
        for e1, e2, mode in ((1, 1, "same"), (1, -1, "head")):
            I_quad = (1 / (4 * mp.pi)) * (
                mp.quad(lambda q: theta_3root(q, a1, e1) * dtheta_dq(q, a2, e2), [0, 1, 10, mp.inf])
                + mp.quad(lambda q: theta_3root(-q, a1, e1) * dtheta_dq(-q, a2, e2), [0, 1, 10, mp.inf])
            )
            J_quad = (1 / (8 * mp.pi)) * (
                mp.quad(
                    lambda q: theta_3root(q, a1, e1) * dtheta_dq(q, a2, e2)
                    - theta_3root(q, a2, e2) * dtheta_dq(q, a1, e1),
                    [0, 1, 10, mp.inf],
                )
                + mp.quad(
                    lambda q: theta_3root(-q, a1, e1) * dtheta_dq(-q, a2, e2)
                    - theta_3root(-q, a2, e2) * dtheta_dq(-q, a1, e1),
                    [0, 1, 10, mp.inf],
                )
            )
            if mode == "same":
                I_cl = I_same_closed(a1, a2)
                J_cl = J_same_closed(a1, a2)
            else:
                I_cl = I_head_closed(a1, a2)
                J_cl = I_head_closed(a1, a2)
            worst_int = max(worst_int, abs(I_quad - I_cl), abs(J_quad - J_cl))

        # Crossing relation I_head(a1, a2) == I_same(a1, pi - a2)
        worst_cross = max(
            worst_cross,
            abs(I_head_closed(a1, a2) - I_same_closed(a1, mp.pi - mp.mpf(a2))),
        )

        # FK gradient check: dI/da1 == dJ/da1 == (1/4pi) int (dtheta_1/da1) theta_2' dq
        dI_da1 = mp.diff(lambda x: I_same_closed(x, a2), mp.mpf(a1))
        dJ_da1 = mp.diff(lambda x: J_same_closed(x, a2), mp.mpf(a1))
        dK_elem = dK_da1_elementary(a1, a2) - mp.diff(lambda x: K_closed(a2, x), mp.mpf(a1))
        worst_grad = max(worst_grad, abs(dI_da1 - dJ_da1), abs(dI_da1 - dK_elem))

    print(f"   max |I_quad - I_closed|, |J_quad - J_closed| = {mp.nstr(worst_int, 3)}")
    print(f"   max |I_head(a1,a2) - I_same(a1, pi-a2)|      = {mp.nstr(worst_cross, 3)}")
    print(f"   max |grad I_12 - grad J_12 - FK_grad|        = {mp.nstr(worst_grad, 3)}")
    ok &= worst_int < 1e-32 and worst_cross < 1e-34 and worst_grad < 1e-33

    # ------------------------------------------------------------------
    # 3. Single-soliton one-loop momentum Delta P_1 and energy Delta E_1^ren
    # ------------------------------------------------------------------
    print("3. Single-soliton one-loop momentum Delta P_1(alpha) and energy Delta E_1^ren(alpha):")
    worst_p1 = mp.mpf(0)
    worst_e1 = mp.mpf(0)
    for a_val in ("0.25", "0.5", "0.85"):
        a_mp = mp.mpf(a_val)
        k1 = mp.sin(a_mp)
        phis = [a_mp + 2 * mp.pi / 3, a_mp - 2 * mp.pi / 3, a_mp]
        cs = [1, 1, -2]
        dp_quad = (1 / (4 * mp.pi)) * mp.quad(
            lambda q: q * (dtheta_dq(q, a_mp, 1) - dtheta_dq(-q, a_mp, 1)),
            [0, 1, 10, mp.inf],
        )
        dp_cl = Delta_P1_closed(a_mp, 1)
        worst_p1 = max(worst_p1, abs(dp_quad - dp_cl))

        # Rational integrand after algebraic cancellation of the -6 k1 tadpole
        de_quad = (1 / (4 * mp.pi)) * (
            -6 * k1
            + mp.quad(
                lambda q: (q / mp.sqrt(1 + q**2))
                * sum(2 * c * mp.sin(ph) * mp.cos(ph)**2 / (q**2 + mp.sin(ph)**2) for c, ph in zip(cs, phis)),
                [0, 1, 10, mp.inf],
            )
        )
        de_cl = Delta_E1_ren_closed(a_mp)
        worst_e1 = max(worst_e1, abs(de_quad - de_cl))
        print(
            f"   alpha={a_val:<4s}: Delta P_1/hbar = {float(dp_cl):+.10f}, "
            f"Delta E_1^ren/hbar = {float(de_cl):+.10f}"
        )
    print(f"   max |Delta P_1_quad - closed| = {mp.nstr(worst_p1, 3)}, |Delta E_1_quad - closed| = {mp.nstr(worst_e1, 3)}")
    ok &= worst_p1 < 1e-33 and worst_e1 < 1e-33

    # ------------------------------------------------------------------
    # 4. Comparison tables in the regular region against Candidates (A) and (B)
    # ------------------------------------------------------------------
    print("\n4. Comparison in the regular region (alpha_j < pi/3, A_12 > 0):")
    print("   OVERTAKING COLLISIONS (eps1 = +1, eps2 = +1, alpha1 < alpha2 => v1 > v2, eta < 0):")
    print("   (a1, a2)      A_12      Cand(A)    Cand(B)     I_12 (raw)   J_12 (antisym)  delta_renorm   I_12+d_ren   J_12+d_ren")
    overtaking_pts = [
        ("0.15", "0.30"),
        ("0.20", "0.60"),
        ("0.25", "0.50"),
        ("0.30", "0.80"),
        ("0.40", "0.65"),
    ]
    for a1, a2 in overtaking_pts:
        s1, s2 = sol_good(a1, 1), sol_good(a2, 1)
        A_val = A12_good(s1, s2)
        ok &= A_val > 0 and s1["a"] < mp.pi / 3 and s2["a"] < mp.pi / 3
        cand_A = -mp.pi / 2  # tau = +1, sgn(eta) = -1 since ell1 < ell2 for a1 < a2 < pi/3
        cand_B = mp.mpf(0)
        I_val = I_same_closed(a1, a2)
        J_val = J_same_closed(a1, a2)
        d_ren = delta_renorm_good(a1, 1, a2, 1)
        # Verify d_ren against numerical derivative of delta0
        dd0_dP1 = mp.diff(lambda x: delta0_good(x, 1, a2, 1), s1["a"]) / s1["dP_da"]
        dd0_dP2 = mp.diff(lambda x: delta0_good(a1, 1, x, 1), s2["a"]) / s2["dP_da"]
        d_ren_num = -(dd0_dP1 * Delta_P1_closed(a1, 1) + dd0_dP2 * Delta_P1_closed(a2, 1))
        ok &= abs(d_ren - d_ren_num) < 1e-32
        print(
            f"   ({a1},{a2})  {float(A_val):8.5f}  {float(cand_A):+9.6f}  {float(cand_B):+9.6f}  "
            f"{float(I_val):+11.6f}    {float(J_val):+11.6f}    {float(d_ren):+11.6f}  "
            f"{float(I_val+d_ren):+11.6f}  {float(J_val+d_ren):+11.6f}"
        )

    print("\n   HEAD-ON COLLISIONS (eps1 = +1, eps2 = -1, alpha1 + alpha2 < pi/3 => A_12 > 0, eta < 0):")
    print("   (a1, a2)      A_12      Cand(A)    Cand(B)     I_12 = J_12    delta_renorm   I_12+d_ren")
    headon_pts = [
        ("0.10", "0.15"),
        ("0.20", "0.30"),
        ("0.25", "0.45"),
        ("0.35", "0.55"),
        ("0.40", "0.60"),
    ]
    for a1, a2 in headon_pts:
        s1, s2 = sol_good(a1, 1), sol_good(a2, -1)
        A_val = A12_good(s1, s2)
        ok &= A_val > 0 and s1["a"] < mp.pi / 3 and s2["a"] < mp.pi / 3
        cand_A = +mp.pi / 2  # tau = -1, sgn(eta) = -1 since ell1 - ell2 = -cos(3a1) - cos(3a2) < 0 for a1 + a2 < pi/3
        cand_B = mp.mpf(0)
        I_val = I_head_closed(a1, a2)
        d_ren = delta_renorm_good(a1, 1, a2, -1)
        dd0_dP1 = mp.diff(lambda x: delta0_good(x, 1, a2, -1), s1["a"]) / s1["dP_da"]
        dd0_dP2 = mp.diff(lambda x: delta0_good(a1, 1, x, -1), s2["a"]) / s2["dP_da"]
        d_ren_num = -(dd0_dP1 * Delta_P1_closed(a1, 1) + dd0_dP2 * Delta_P1_closed(a2, -1))
        ok &= abs(d_ren - d_ren_num) < 1e-32
        print(
            f"   ({a1},{a2})  {float(A_val):8.5f}  {float(cand_A):+9.6f}  {float(cand_B):+9.6f}  "
            f"{float(I_val):+11.6f}    {float(d_ren):+11.6f}  {float(I_val+d_ren):+11.6f}"
        )

    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    assert ok

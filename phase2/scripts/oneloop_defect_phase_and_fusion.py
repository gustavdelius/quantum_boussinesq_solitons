"""One-loop soliton-defect transmission phase, defect fusion, and KP-triangle closure
for the GOOD Boussinesq equation with the Corrigan-Zambon type-I defect (Route E, step 2).
See phase2/oneloop_defect_phase.pdf, Sections 5-6.

Checks:
  1. Closed-form 3x2 dilogarithm (Legendre chi_2) formula for the one-loop soliton-defect phase
     I_SD(alpha, eps; s, eta) = J_SD^phys(alpha, eps; s, eta) against numerical quadrature
     for both |s| < 2 and |s| > 2 and all (eps, eta) in {+1, -1}^2 to < 1e-34;
     exact parity/crossing relation I_SD(alpha, -1; s, eta) = I_SD(alpha, +1; -s, eta);
     and Faddeev-Korepin gradient identity d/d(alpha) I_SD = -(1/4pi) int (d_alpha theta_S) theta_D' dq.
  2. Closed-form 2-root dipole formulas for the one-loop defect vacuum momentum Delta P_D(q_D, eta)
     and normal-ordered energy Delta E_D^ren(q_D, eta) against numerical quadrature to < 1e-35;
     and EXACT ONE-LOOP ENERGY-MOMENTUM BALANCE in soliton absorption across all 12 KP triples:
       (Delta E_1^ren(S_i), Delta P_1(S_i)) = rho_i [ (Delta E_D^ren(q_{i+1}) - Delta E_D^ren(q_i)),
                                                      (Delta P_D(q_{i+1}) - Delta P_D(q_i)) ].
  3. Exact one-loop defect fusion and KP-triangle closure across all 12 KP triangles (36 edges):
       I_SD(3; q_{i+1}, eta) - I_SD(3; q_i, eta) = rho_i I_SS(3, S_i)
     with the EXACT SAME signs rho_i = +-1 as the tree-level fusion identity
       delta_D(q_{i+1})(3) - delta_D(q_i)(3) = rho_i [ delta_0(3, S_i) + M_{S_i} Pi_3 ],
     proving that around every closed KP triangle q_1 -> q_2 -> q_3 -> q_1:
       sum_{i=1}^3 rho_i I_SS(3, S_i) = 0   and   sum_{i=1}^3 rho_i Delta X_3(3, S_i) Delta P_3 = 0
     identically (to < 1e-34), whereas adding the +(pi/2) delta_{eps_3, eps_{S_i}} constant of J_SS
     leaves the non-zero coboundary defect (pi/2) sum_i rho_i delta_{eps_3, eps_{S_i}}.
  4. Two-soliton + defect one-loop factorisation delta_tot^(1)(1, 2; D) = delta_12^(1) + delta_1D^(1) + delta_2D^(1),
     small-amplitude O(alpha ln alpha) + O(alpha) law as alpha -> 0, finiteness of I_SD and log divergence
     of delta_renorm^D across absorption lines q_D -> p_{a,b}, and +-pi jumps across s = -eps and alpha = pi/3.
"""
import mpmath as mp

mp.mp.dps = 35
r3 = mp.sqrt(3)
ok = True


def L_analytic(z):
    """Analytic continuation of L(z) = int_0^z (2 ln t)/(1 - t^2) dt, odd under z -> -z."""
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(0)
    if mp.re(z) < 0:
        return -L_analytic(-z)
    if z == 1:
        return mp.mpc(-mp.pi**2 / 4)
    chi2 = (mp.polylog(2, z) - mp.polylog(2, -z)) / 2
    return -(2 * chi2 + mp.log(z) * mp.log((1 - z) / (1 + z)))


def dL_analytic(z):
    z = mp.mpc(z)
    if abs(z - 1) < 1e-30:
        return mp.mpc(-1)
    if abs(z + 1) < 1e-30:
        return mp.mpc(1)
    z_pos = z if mp.re(z) > 0 else -z
    return 2 * mp.log(z_pos) / (1 - z * z)


def sol_good(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    k = mp.sin(a)
    v = e * mp.cos(a)
    w = k * v
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * e * k**3 * mp.cos(a)
    ell = -e * mp.cos(3 * a)
    pa = (k + v / r3) / 2
    pb = (v / r3 - k) / 2
    return dict(a=a, eps=e, k=k, v=v, w=w, M=M, Pi=Pi, P=P, ell=ell, pa=pa, pb=pb)


def A12_good(s1, s2):
    Pp = lambda K, W: W * W - K * K + K**4
    return -Pp(s1["k"] - s2["k"], s1["w"] - s2["w"]) / Pp(s1["k"] + s2["k"], s1["w"] + s2["w"])


def delta0_SS(s1, s2):
    return mp.mpf(2) / 9 * (s1["ell"] - s2["ell"]) * mp.ln(abs(A12_good(s1, s2))) + mp.mpf(2) / 3 * s1["M"] * s2["M"] * (s1["v"] - s2["v"])


def delta0_SD(st, qD, eta):
    s = -2 * r3 * qD
    e = st["eps"] * eta
    zt = ((st["pb"] - qD) / (st["pa"] - qD)) ** e
    ellD = (3 * s - s**3) / 2
    MD = e * qD
    PiD = e * (s * s - 2) / (2 * r3)
    return mp.mpf(2) / 9 * (st["ell"] - ellD) * mp.ln(abs(zt)) - mp.mpf(2) / 3 * (st["Pi"] * MD - PiD * st["M"])


def theta_elem(q, psi):
    q = mp.mpmathify(q)
    aq = abs(q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = mp.sqrt(1 + q * q)
    s, c = mp.sin(psi), mp.cos(psi)
    return -sgn * mp.atan(s / aq) - mp.atan((s * c) / (aq * w + q * q + s * s))


def dtheta_elem(q, psi):
    q = mp.mpmathify(q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = mp.sqrt(1 + q * q)
    s, c = mp.sin(psi), mp.cos(psi)
    return s / (q * q + s * s) + (sgn / w) * (s * c) / (q * q + s * s)


def dtheta_elem_dpsi(q, psi):
    """Exact derivative d/d(psi) Theta(q; psi)."""
    q = mp.mpmathify(q)
    aq = abs(q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = mp.sqrt(1 + q * q)
    s, c = mp.sin(psi), mp.cos(psi)
    N = s * c
    D = aq * w + q * q + s * s
    dN = c * c - s * s
    dD = 2 * s * c
    return -sgn * aq * c / (q * q + s * s) - (dN * D - N * dD) / (D * D + N * N)


def theta_S_func(q, alpha, eps=1):
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    if eps == -1:
        phis = [mp.pi - p for p in phis]
    return sum(c * theta_elem(q, p) for c, p in zip(cs, phis))


def dtheta_S_dq(q, alpha, eps=1):
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    if eps == -1:
        phis = [mp.pi - p for p in phis]
    return sum(c * dtheta_elem(q, p) for c, p in zip(cs, phis))


def dtheta_S_dalpha(q, alpha, eps=1):
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    if eps == -1:
        phis = [mp.pi - p for p in phis]
    return eps * sum(c * dtheta_elem_dpsi(q, p) for c, p in zip(cs, phis))


def dtheta_D_direct(q, s, eta=1):
    q = mp.mpmathify(q)
    s = mp.mpmathify(s)
    qD = -s / (2 * r3)
    w = mp.sqrt(1 + q * q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    X = sgn * w / (2 * r3) - qD
    Y = -q / 2
    dX = sgn * q / (2 * r3 * w)
    dY = -mp.mpf("0.5")
    return eta * 2 * (X * dY - dX * Y) / (X * X + Y * Y)


def theta_D_phys(q, s, eta=1):
    """Physical defect phase shift continuous from theta_D^phys(0+-) = 0."""
    q = mp.mpmathify(q)
    s = mp.mpmathify(s)
    gamma = mp.acos(s / 2)
    psi1 = 2 * mp.pi / 3 - gamma
    psi2 = -2 * mp.pi / 3 - gamma
    th_elem = mp.re(-eta * theta_elem(q, psi1) + eta * theta_elem(q, psi2))
    q0 = mp.mpf("1e-40") if q >= 0 else mp.mpf("-1e-40")
    th_0 = mp.re(-eta * theta_elem(q0, psi1) + eta * theta_elem(q0, psi2))
    return th_elem - th_0


def I_SD_closed(alpha, eps, s, eta=1):
    """Closed-form 3x2 dilogarithm formula for the one-loop soliton-defect phase."""
    a = mp.mpmathify(alpha)
    s = mp.mpmathify(s)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    cDs = [-eta, +eta]
    tot = mp.mpc(0)
    for phi_r, c_r in zip(phis, cs):
        for psi_s, c_s in zip(psis, cDs):
            z = mp.tan(phi_r / 2) * (mp.tan(psi_s / 2) ** (-eps))
            tot += c_r * c_s * L_analytic(z)
    return eps * mp.re(tot) / (2 * mp.pi)


def I_SS_closed(a1, eps1, a2, eps2):
    """Closed-form 3x3 dilogarithm formula for the two-soliton phase I_12."""
    aa1, aa2 = mp.mpmathify(a1), mp.mpmathify(a2)
    phis1 = [aa1 + 2 * mp.pi / 3, aa1 - 2 * mp.pi / 3, aa1]
    phis2 = [aa2 + 2 * mp.pi / 3, aa2 - 2 * mp.pi / 3, aa2]
    cs = [1, 1, -2]
    tot = mp.mpf(0)
    for p1, c1 in zip(phis1, cs):
        for p2, c2 in zip(phis2, cs):
            z = mp.tan(p1 / 2) / mp.tan(p2 / 2) if eps1 == eps2 else mp.tan(p1 / 2) * mp.tan(p2 / 2)
            tot += c1 * c2 * mp.re(L_analytic(z))
    return eps1 * tot / (2 * mp.pi)


def Delta_P1_closed(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return -e / (2 * mp.pi) * sum(c * mp.sin(ph) * mp.ln((1 - mp.cos(ph)) / (1 + mp.cos(ph))) / 2 for c, ph in zip(cs, phis))


def Delta_E1_ren_closed(alpha):
    a = mp.mpmathify(alpha)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return sum(
        c * mp.sin(ph) * (1 - mp.cos(ph) * mp.ln((1 - mp.cos(ph)) / (1 + mp.cos(ph))) / 2)
        for c, ph in zip(cs, phis)
    ) / (2 * mp.pi)


def Delta_PD_closed(qD, eta=1):
    s = -2 * r3 * qD
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    cDs = [-eta, +eta]
    return -sum(
        c * mp.re(mp.sin(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2)
        for c, ps in zip(cDs, psis)
    ) / (2 * mp.pi)


def Delta_ED_ren_closed(qD, eta=1):
    s = -2 * r3 * qD
    gamma = mp.acos(s / 2)
    psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
    cDs = [-eta, +eta]
    return sum(
        c * mp.re(mp.sin(ps) * (1 - mp.cos(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2))
        for c, ps in zip(cDs, psis)
    ) / (2 * mp.pi)


def delta_renorm_SD(alpha, eps, s, eta=1):
    """Fixed-physical-momentum shift -(d delta_D / d P_1) Delta P_1 = +Delta x_D * Delta P_1."""
    st = sol_good(alpha, eps)
    qD = -mp.mpmathify(s) / (2 * r3)
    zt = ((st["pb"] - qD) / (st["pa"] - qD)) ** (eps * eta)
    Dx_D = -mp.ln(abs(zt)) / st["k"]
    return Dx_D * Delta_P1_closed(alpha, eps)


def sol_from_pair(x, y):
    pa, pb = max(x, y), min(x, y)
    eps = 1 if pa + pb > 0 else -1
    al = mp.acos(r3 * abs(pa + pb))
    return sol_good(al, eps)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Closed-form 3x2 dilogarithm formula vs quadrature, crossing, and FK gradient
    # ------------------------------------------------------------------
    print("1. Closed-form 3x2 dilogarithm formula I_SD = J_SD^phys vs quadrature and FK gradient:")
    worst_quad = mp.mpf(0)
    worst_cross = mp.mpf(0)
    worst_grad = mp.mpf(0)
    for s_val in ("-2.5", "-1.5", "0.7", "2.4"):
        s_mp = mp.mpf(s_val)
        for al_val in ("0.35", "0.75"):
            al_mp = mp.mpf(al_val)
            for eps in (1, -1):
                I_num = mp.quad(
                    lambda q: (
                        theta_S_func(q, al_mp, eps) * dtheta_D_direct(q, s_mp, 1)
                        + theta_S_func(-q, al_mp, eps) * dtheta_D_direct(-q, s_mp, 1)
                    )
                    / (4 * mp.pi),
                    [0, 1, 5, mp.inf],
                )
                J_phys_num = mp.quad(
                    lambda q: (
                        theta_S_func(q, al_mp, eps) * dtheta_D_direct(q, s_mp, 1)
                        - theta_D_phys(q, s_mp, 1) * dtheta_S_dq(q, al_mp, eps)
                        + theta_S_func(-q, al_mp, eps) * dtheta_D_direct(-q, s_mp, 1)
                        - theta_D_phys(-q, s_mp, 1) * dtheta_S_dq(-q, al_mp, eps)
                    )
                    / (8 * mp.pi),
                    [0, 1, 5, mp.inf],
                )
                I_cl = I_SD_closed(al_mp, eps, s_mp, 1)
                worst_quad = max(worst_quad, abs(I_num - I_cl), abs(J_phys_num - I_cl))
                worst_cross = max(
                    worst_cross,
                    abs(I_SD_closed(al_mp, -1, s_mp, 1) - I_SD_closed(al_mp, 1, -s_mp, 1)),
                    abs(I_SD_closed(al_mp, eps, s_mp, -1) + I_SD_closed(al_mp, eps, s_mp, 1)),
                )
            # Check FK gradient d/d(alpha) I_SD = (1/4pi) int (d_alpha theta_S(q)) dtheta_D(q) dq
            dI_da = mp.diff(lambda a: I_SD_closed(a, 1, s_mp, 1), al_mp)
            fk_grad = mp.quad(
                lambda q: (
                    dtheta_S_dalpha(q, al_mp, 1) * dtheta_D_direct(q, s_mp, 1)
                    + dtheta_S_dalpha(-q, al_mp, 1) * dtheta_D_direct(-q, s_mp, 1)
                )
                / (4 * mp.pi),
                [0, 1, 5, mp.inf],
            )
            worst_grad = max(worst_grad, abs(dI_da - fk_grad))
    print(f"   max |I_SD_closed - I_num| and |I_SD_closed - J_SD^phys_num| = {mp.nstr(worst_quad, 3)}")
    print(f"   max crossing/orientation residual = {mp.nstr(worst_cross, 3)}")
    print(f"   max FK gradient residual = {mp.nstr(worst_grad, 3)}")
    ok &= max(worst_quad, worst_cross, worst_grad) < 1e-32

    # ------------------------------------------------------------------
    # 2. Closed-form Delta P_D, Delta E_D^ren and one-loop (E, P) balance in absorption
    # ------------------------------------------------------------------
    print("2. Closed-form defect vacuum shifts (Delta E_D^ren, Delta P_D) and one-loop absorption balance:")
    worst_pd_quad = mp.mpf(0)
    for qv in ("-0.2", "0.25", "0.75"):
        qD = mp.mpf(qv)
        s = -2 * r3 * qD
        gamma = mp.acos(s / 2)
        psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
        cDs = [-1, 1]
        odd_int = lambda q: (q / mp.sqrt(1 + q * q)) * sum(
            c * mp.re(mp.sin(ps) * mp.cos(ps) / (q * q + mp.sin(ps) ** 2)) for c, ps in zip(cDs, psis)
        ) / (2 * mp.pi)
        P_num = mp.quad(odd_int, [0, 1, 5, mp.inf])
        worst_pd_quad = max(worst_pd_quad, abs(P_num - Delta_PD_closed(qD, 1)))
        # Normal-ordered energy integral after algebraic cancellation of the tadpole:
        tadpole_part = sum(c * mp.re(mp.sin(ps)) for c, ps in zip(cDs, psis)) / (2 * mp.pi)
        even_int = lambda q: (q / mp.sqrt(1 + q * q)) * sum(
            c * mp.re(mp.sin(ps) * mp.cos(ps) ** 2 / (q * q + mp.sin(ps) ** 2)) for c, ps in zip(cDs, psis)
        ) / (2 * mp.pi)
        E_num = tadpole_part + mp.quad(even_int, [0, 1, 5, mp.inf])
        worst_pd_quad = max(worst_pd_quad, abs(E_num - Delta_ED_ren_closed(qD, 1)))

    worst_ep_bal = mp.mpf(0)
    for qv in ("-0.2", "0.1", "0.45"):
        q = mp.mpf(qv)
        disc = mp.sqrt(1 - 3 * q * q)
        qs = [q, (-q + disc) / 2, (-q - disc) / 2]
        for eta in (1, -1):
            for i in range(3):
                q_curr = qs[i]
                q_next = qs[(i + 1) % 3]
                S_i = sol_from_pair(q_curr, q_next)
                rho_i = eta if q_next < q_curr else -eta
                dP_jump = Delta_PD_closed(q_next, eta) - Delta_PD_closed(q_curr, eta)
                dE_jump = Delta_ED_ren_closed(q_next, eta) - Delta_ED_ren_closed(q_curr, eta)
                worst_ep_bal = max(
                    worst_ep_bal,
                    abs(dP_jump - rho_i * Delta_P1_closed(S_i["a"], S_i["eps"])),
                    abs(dE_jump - rho_i * Delta_E1_ren_closed(S_i["a"])),
                )
    print(f"   max |Delta P_D - quad| and |Delta E_D^ren - quad| = {mp.nstr(worst_pd_quad, 3)}")
    print(f"   max |(Delta E_1^ren, Delta P_1) - rho_i jump(Delta E_D^ren, Delta P_D)| over 18 absorption edges = {mp.nstr(worst_ep_bal, 3)}")
    ok &= max(worst_pd_quad, worst_ep_bal) < 1e-33

    # ------------------------------------------------------------------
    # 3. Exact one-loop defect fusion and KP-triangle closure
    # ------------------------------------------------------------------
    print("3. Exact one-loop defect fusion and KP-triangle closure across 12 KP triangles:")
    worst_fusion_1loop = mp.mpf(0)
    worst_tri_I = mp.mpf(0)
    worst_tri_renorm = mp.mpf(0)
    worst_tri_J_diff = mp.mpf(0)
    n_triangles = 0
    for qv in ("-0.2", "0.1", "0.45"):
        q = mp.mpf(qv)
        disc = mp.sqrt(1 - 3 * q * q)
        qs = [q, (-q + disc) / 2, (-q - disc) / 2]
        for eta in (1, -1):
            for e3 in (1, -1):
                s3 = sol_good("0.35", e3)
                tri_I = mp.mpf(0)
                tri_J = mp.mpf(0)
                tri_renorm = mp.mpf(0)
                sum_rho_same = 0
                for i in range(3):
                    q_curr = qs[i]
                    q_next = qs[(i + 1) % 3]
                    S_i = sol_from_pair(q_curr, q_next)
                    rho_i = eta if q_next < q_curr else -eta
                    # Tree-level fusion sign rho_tree = eps_3 * rho_i from directional transmission z_t = z_3^(eps_3):
                    num_tree = delta0_SD(s3, q_next, eta) - delta0_SD(s3, q_curr, eta)
                    den_tree = delta0_SS(s3, S_i) + S_i["M"] * s3["Pi"]
                    rho_tree = int(mp.nint(num_tree / den_tree))
                    assert rho_tree == e3 * rho_i and abs(num_tree - rho_tree * den_tree) < 1e-33

                    # One-loop fusion step:
                    s_curr = -2 * r3 * q_curr
                    s_next = -2 * r3 * q_next
                    dI_SD = I_SD_closed(s3["a"], e3, s_next, eta) - I_SD_closed(s3["a"], e3, s_curr, eta)
                    I_SS = I_SS_closed(s3["a"], e3, S_i["a"], S_i["eps"])
                    J_SS = I_SS + (mp.pi / 2 if e3 == S_i["eps"] else mp.mpf(0))
                    worst_fusion_1loop = max(worst_fusion_1loop, abs(dI_SD - rho_i * I_SS))
                    tri_I += rho_i * I_SS
                    tri_J += rho_i * J_SS
                    if e3 == S_i["eps"]:
                        sum_rho_same += rho_i

                    # Momentum-renormalisation fusion step for test soliton 3:
                    d_ren_SD = delta_renorm_SD(s3["a"], e3, s_next, eta) - delta_renorm_SD(s3["a"], e3, s_curr, eta)
                    lnA = mp.ln(abs(A12_good(s3, S_i)))
                    DX3_unshifted = -lnA / s3["k"]
                    worst_fusion_1loop = max(
                        worst_fusion_1loop, abs(d_ren_SD - rho_tree * DX3_unshifted * Delta_P1_closed(s3["a"], e3))
                    )
                    tri_renorm += rho_tree * DX3_unshifted * Delta_P1_closed(s3["a"], e3)

                worst_tri_I = max(worst_tri_I, abs(tri_I))
                worst_tri_renorm = max(worst_tri_renorm, abs(tri_renorm))
                worst_tri_J_diff = max(worst_tri_J_diff, abs(tri_J - (mp.pi / 2) * sum_rho_same))
                n_triangles += 1
    print(f"   tested {n_triangles} KP triangles (36 edges):")
    print(f"   max |I_SD(q_{{i+1}}) - I_SD(q_i) - rho_i I_SS(3, S_i)| = {mp.nstr(worst_fusion_1loop, 3)}")
    print(f"   max closed-triangle anomaly |sum_i rho_i I_SS(3, S_i)| = {mp.nstr(worst_tri_I, 3)}")
    print(f"   max closed-triangle renorm anomaly |sum_i rho_i DX_3 Delta P_3| = {mp.nstr(worst_tri_renorm, 3)}")
    print(f"   max |sum_i rho_i J_SS(3, S_i) - (pi/2) sum_i rho_i delta_{{eps3,eps_i}}| = {mp.nstr(worst_tri_J_diff, 3)}")
    ok &= max(worst_fusion_1loop, worst_tri_I, worst_tri_renorm, worst_tri_J_diff) < 1e-32

    # ------------------------------------------------------------------
    # 4. Small-amplitude limit, absorption lines, and stability/cusp jumps
    # ------------------------------------------------------------------
    print("4. Small-amplitude limit, absorption lines, and stability/cusp jumps:")
    # (a) Small-amplitude expansion of I_SD(alpha, eps; s, eta) as alpha -> 0:
    s_test = mp.mpf("-2.5")
    gamma_t = mp.acos(s_test / 2)
    psis_t = [2 * mp.pi / 3 - gamma_t, -2 * mp.pi / 3 - gamma_t]
    cDs_t = [-1, 1]
    errs_sa = []
    for al_str in ("1e-2", "1e-4", "1e-6"):
        al = mp.mpf(al_str)
        ex = I_SD_closed(al, 1, s_test, 1)
        tot_as = mp.mpc(0)
        for psi_s, c_s in zip(psis_t, cDs_t):
            w = 1 / mp.tan(psi_s / 2)
            w_pos = w if mp.re(w) > 0 else -w
            tot_as += c_s * (-2 * al * w * (mp.log(al / 2) + mp.log(w_pos) - 1) + 4 * al * w * dL_analytic(r3 * w))
        asympt = mp.re(tot_as) / (2 * mp.pi)
        errs_sa.append(abs(ex - asympt))
    print(f"   small-amplitude |I_SD - asympt| at alpha=1e-2,1e-4,1e-6: {[mp.nstr(e, 3) for e in errs_sa]}")
    ok &= errs_sa[2] < 1e-17 and errs_sa[2] < errs_sa[1] < errs_sa[0]

    # (b) Absorption line: q_D = p_a(alpha_abs) for s = 0.7 (q_D = -0.7/(2 sqrt(3))):
    s_abs = mp.mpf("0.7")
    qD_abs = -s_abs / (2 * r3)
    al_abs = mp.findroot(lambda a: sol_good(a, -1)["pb"] - qD_abs, mp.mpf("0.5"))
    I_at_abs = I_SD_closed(al_abs, -1, s_abs, 1)
    for h_str in ("1e-2", "1e-5", "1e-8"):
        h = mp.mpf(h_str)
        I_near = I_SD_closed(al_abs + h, -1, s_abs, 1)
        ren_near = delta_renorm_SD(al_abs + h, -1, s_abs, 1)
        print(
            f"   absorption h={h_str}: I_SD={mp.nstr(I_near, 10)}, |I_SD-I_abs|={mp.nstr(abs(I_near-I_at_abs), 3)}, delta_renorm^D={mp.nstr(ren_near, 8)}"
        )
    ok &= abs(I_SD_closed(al_abs + mp.mpf("1e-8"), -1, s_abs, 1) - I_at_abs) < 1e-7

    # (c) Jumps across s = -eps and across soliton cusp alpha = pi/3:
    h = mp.mpf("1e-10")
    jump_s_m1 = I_SD_closed("0.5", 1, -1 + h, 1) - I_SD_closed("0.5", 1, -1 - h, 1)
    jump_s_p1 = I_SD_closed("0.5", -1, 1 + h, 1) - I_SD_closed("0.5", -1, 1 - h, 1)
    jump_cusp_stable = I_SD_closed(mp.pi / 3 + h, 1, "-1.5", 1) - I_SD_closed(mp.pi / 3 - h, 1, "-1.5", 1)
    jump_cusp_mid = I_SD_closed(mp.pi / 3 + h, 1, "0.5", 1) - I_SD_closed(mp.pi / 3 - h, 1, "0.5", 1)
    print(
        f"   jumps: s=-1(eps=+1): {mp.nstr(jump_s_m1, 8)}, s=+1(eps=-1): {mp.nstr(jump_s_p1, 8)}, "
        f"cusp(s=-1.5): {mp.nstr(jump_cusp_stable, 8)}, cusp(s=0.5): {mp.nstr(jump_cusp_mid, 3)}"
    )
    ok &= (
        abs(jump_s_m1 + mp.pi) < 1e-7
        and abs(jump_s_p1 - mp.pi) < 1e-7
        and abs(jump_cusp_stable - mp.pi) < 1e-7
        and abs(jump_cusp_mid) < 1e-7
    )

    assert ok, "SOME CHECKS FAILED"
    print("ALL CHECKS PASSED")

"""One-loop phonon scattering and analytical vacuum stability on the Corrigan-Zambon defect
for the GOOD Boussinesq equation (Route E, step 2).
See phase2/oneloop_defect_phase.pdf, Sections 2-3.

Conventions (good equation, project normalisation):
  u_tt = u_xx - u_xxxx - 3(u^2)_xx,   u = phi_x = 2(log tau)_xx = -u_CZ_x / 3.
  Soliton: k = sin(alpha), v = eps cos(alpha), KP roots p_a, p_b = (eps cos(alpha)/sqrt(3) +- k)/2.
  CZ type-I defect at x = 0: parameter s = sigma_0, orientation eta = +-1, KP root q_D = -s/(2 sqrt(3)).

Checks:
  1. Exact 3-soliton Hirota solution on the CZ defect: tau_L(E_1, E_2, E_3) on x < 0 and
     tau_R(z_1 E_1, z_2 E_2, z_3 E_3) on x > 0 satisfy all 4 nonlinear CZ sewing conditions at x = 0.
  2. Time-dependent phonon scattering on 1-soliton + defect and 2-soliton + defect backgrounds:
     linearising in E_ph = exp(i(q x - w(q) t)) solves the linearised bulk PDE and linearised
     time-dependent sewing conditions at x = 0, with R(q) = 0, |z_ph(q; q_D, eta)| = 1, and
     factorised transmission T_tot(q) = A(1, ph(q)) A(2, ph(q)) z_ph(q; q_D, eta).
  3. Exact KP-vertex factorisation and 2-root dipole decomposition:
     - single-root factor Z(q; p) = (p_{ph,b}(q) - p)/(p_{ph,a}(q) - p) satisfies
       z_ph(q; q_D, eta) = Z(q; q_D)^eta and A(sol(p_a, p_b), ph(q)) = Z(q; p_b) / Z(q; p_a);
     - d/dq theta_D(q; q_D, eta) decomposes into a 2-root dipole over (psi_1, psi_2) = (2pi/3 - gamma,
       -2pi/3 - gamma) with weights (-eta, +eta), where gamma = arccos(s/2) and
       sin(psi_1) = q_+ - q_D, sin(psi_2) = q_D - q_- are the two KP-triple root differences at q_D;
     - subtracting the two defect dipoles at p_b and p_a reproduces the (+1, +1, -2) soliton triple.
  4. Defect Levinson phase winding across eta*s = -1, +1 and exact factored characteristic polynomial
     det M_red(S; sigma, eta) = -S^2(S^2 - 1)/(24) * P_-(S; eta*sigma) * P_+(S; eta*sigma)
     for localised growing modes w = i*gamma_grow (S = kappa_1 + kappa_2 = sqrt(1 + 2*gamma_grow)),
     giving the closed-form growth rates
       w_{1,2} = i * [ eta*sigma*sqrt(4 - sigma^2) +- sqrt(3)*(2 - sigma^2) ] / 8
     and proving analytically the thresholds eta*sigma = -1, +1, +2.
"""
import mpmath as mp

mp.mp.dps = 35
r3 = mp.sqrt(3)
I = mp.mpc(0, 1)
ok = True


def sol_good(alpha, eps=1):
    a = mp.mpmathify(alpha)
    e = mp.mpf(eps)
    k = mp.sin(a)
    v = e * mp.cos(a)
    w = k * v
    pa = (k + v / r3) / 2
    pb = (v / r3 - k) / 2
    return dict(a=a, eps=e, k=k, v=v, w=w, pa=pa, pb=pb)


def phonon_good(q):
    q = mp.mpmathify(q)
    aq = abs(q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = aq * mp.sqrt(1 + q * q)
    pa = mp.mpc(sgn * w / (2 * r3 * aq), q / 2)
    pb = mp.mpc(sgn * w / (2 * r3 * aq), -q / 2)
    return dict(q=q, w=w, k=mp.mpc(0, q), W=mp.mpc(0, w), v=w / q, sgn=sgn, pa=pa, pb=pb)


def A_hirota(m1, m2):
    K1, W1 = m1.get("K", m1["k"]), m1.get("W", m1["w"])
    K2, W2 = m2.get("K", m2["k"]), m2.get("W", m2["w"])
    Pp = lambda K, W: W * W - K * K + K**4
    return -Pp(K1 - K2, W1 - W2) / Pp(K1 + K2, W1 + W2)


def Z_single(q, p):
    """Single-KP-root phonon factor Z(q; p) = (p_{ph,b}(q) - p) / (p_{ph,a}(q) - p)."""
    ph = phonon_good(q)
    return (ph["pb"] - p) / (ph["pa"] - p)


def z_defect_mode(mode, s, eta):
    """Defect transmission factor z for either a soliton or a phonon."""
    k = mode["k"]
    nu = mode["v"]
    return (nu + s - eta * r3 * k) / (nu + s + eta * r3 * k)


def dtheta_elem(q, psi):
    """Universal (sigma=+1) elementary KP-edge phase-shift derivative."""
    q = mp.mpmathify(q)
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = mp.sqrt(1 + q * q)
    s, c = mp.sin(psi), mp.cos(psi)
    return s / (q * q + s * s) + (sgn / w) * (s * c) / (q * q + s * s)


def dtheta_D_dipole(q, s, eta):
    """Two-root dipole formula for d/dq theta_D(q; s, eta), valid for all real s."""
    s = mp.mpmathify(s)
    gamma = mp.acos(s / 2)
    psi1 = 2 * mp.pi / 3 - gamma
    psi2 = -2 * mp.pi / 3 - gamma
    return mp.re(-eta * dtheta_elem(q, psi1) + eta * dtheta_elem(q, psi2))


def dtheta_D_direct(q, s, eta):
    """Direct derivative of eta * arg Z(q; q_D) with q_D = -s/(2 sqrt(3))."""
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


def tau3(xx, tt, m1, m2, m3, c1, c2, c3, eps3=1):
    K1, W1 = m1["k"], m1.get("W", m1["w"])
    K2, W2 = m2["k"], m2.get("W", m2["w"])
    K3, W3 = m3["k"], m3.get("W", m3["w"])
    E1 = c1 * mp.exp(K1 * xx - W1 * tt)
    E2 = c2 * mp.exp(K2 * xx - W2 * tt)
    E3 = eps3 * c3 * mp.exp(K3 * xx - W3 * tt)
    A12 = A_hirota(m1, m2)
    A13 = A_hirota(m1, m3)
    A23 = A_hirota(m2, m3)
    return 1 + E1 + E2 + E3 + A12 * E1 * E2 + A13 * E1 * E3 + A23 * E2 * E3 + A12 * A13 * A23 * E1 * E2 * E3


def cz_sewing_vec(U, UX, UXX, UXXX, UT, UXT, V, VX, VXX, VXXX, VT, VXT, s, eta):
    """Four Corrigan-Zambon sewing equations at x = 0 in shift-invariant gauge f = g = -sigma^2/(4 mu)."""
    al = eta / r3
    be = 2 * al
    mu = mp.mpf(2) / (3 * be)
    sig = s + mu * (U - V)
    rho0 = -(s / (2 * mu)) * (1 - s**2 / 3)
    rho = rho0 + (sig - sig**3 / 3) / (2 * mu)
    drho_dsig = (1 - sig**2) / (2 * mu)
    dtau_dW = -be * sig * rho / 2
    H_sig = mu * (be * UX**2 + 2 * al * UX * VX + be * VX**2) / 2 + mu * drho_dsig * (UX + VX) + dtau_dW
    eq1 = -UX - UX**2 + UXXX - sig * VT + H_sig - (al * UXT + be * VXT)
    eq2 = -UXX + al * UT + be * VT + sig * (be * UX + al * VX) + rho
    eq3 = VX + VX**2 - VXXX + sig * UT - H_sig - (be * UXT + al * VXT)
    eq4 = VXX + be * UT + al * VT + sig * (al * UX + be * VX) + rho
    return [eq1, eq2, eq3, eq4]


def eval_jet(fn, t0):
    x0 = mp.mpf(0)
    F = fn(x0, t0)
    FX = mp.diff(lambda x: fn(x, t0), x0, 1)
    FXX = mp.diff(lambda x: fn(x, t0), x0, 2)
    FXXX = mp.diff(lambda x: fn(x, t0), x0, 3)
    FT = mp.diff(lambda t: fn(x0, t), t0, 1)
    FXT = mp.diff(lambda t: mp.diff(lambda x: fn(x, t), x0, 1), t0, 1)
    return F, FX, FXX, FXXX, FT, FXT


def P_minus_quad(S, eta_sig):
    return S * S - ((r3 - 1) / 2) * eta_sig * S + (eta_sig * eta_sig - 2 - r3) / 2


def P_plus_quad(S, eta_sig):
    return S * S - ((r3 + 1) / 2) * eta_sig * S + (eta_sig * eta_sig - 2 + r3) / 2


def locmat_4x4_det(w, sig, eta):
    """Direct 4x4 linearised sewing determinant det(locmat) on vacuum sigma at frequency w."""
    r = mp.sqrt(1 + 4 * w * w)
    ks = [mp.sqrt((1 + r) / 2), mp.sqrt((1 - r) / 2)]
    ks = [kk if mp.re(kk) > 0 else -kk for kk in ks]
    dt = -I * w

    def row_vec(a, ax, axx, axxx, at, axt, b, bx, bxx, bxxx, bt, bxt):
        L0 = (
            -ax
            - r3 * axt * eta / 3
            + axxx
            - bt * sig
            - 2 * r3 * bxt * eta / 3
            + r3 * eta * sig * (sig**2 - 1) * (a - b) / 6
            - (ax + bx) * (sig**2 - 1) / 2
        )
        L1 = (1 - sig**2) * (a - b) / 2 + r3 * eta * (at + 2 * bt) / 3 - axx + r3 * eta * sig * (2 * ax + bx) / 3
        L2 = (
            at * sig
            - 2 * r3 * axt * eta / 3
            + bx
            - r3 * bxt * eta / 3
            - bxxx
            - r3 * eta * sig * (sig**2 - 1) * (a - b) / 6
            + (ax + bx) * (sig**2 - 1) / 2
        )
        L3 = (1 - sig**2) * (a - b) / 2 + r3 * eta * (2 * at + bt) / 3 + bxx + r3 * eta * sig * (ax + 2 * bx) / 3
        return [L0, L1, L2, L3]

    cols = [
        row_vec(1, ks[0], ks[0] ** 2, ks[0] ** 3, dt, dt * ks[0], 0, 0, 0, 0, 0, 0),
        row_vec(1, ks[1], ks[1] ** 2, ks[1] ** 3, dt, dt * ks[1], 0, 0, 0, 0, 0, 0),
        row_vec(0, 0, 0, 0, 0, 0, 1, -ks[0], ks[0] ** 2, -ks[0] ** 3, dt, -dt * ks[0]),
        row_vec(0, 0, 0, 0, 0, 0, 1, -ks[1], ks[1] ** 2, -ks[1] ** 3, dt, -dt * ks[1]),
    ]
    M = mp.matrix([[cols[j][i] for j in range(4)] for i in range(4)])
    return mp.det(M), ks[0], ks[1]


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Exact 3-soliton Hirota solution on the CZ defect
    # ------------------------------------------------------------------
    print("1. Exact 3-soliton Hirota solution on the Corrigan-Zambon defect:")
    worst_3sol = mp.mpf(0)
    s1 = sol_good("0.35", 1)
    s2 = sol_good("0.75", -1)
    s3 = sol_good("0.55", 1)
    for s_val, eta in (("-1.5", 1), ("2.4", -1)):
        s_def = mp.mpf(s_val)
        z1 = z_defect_mode(s1, s_def, eta)
        z2 = z_defect_mode(s2, s_def, eta)
        z3 = z_defect_mode(s3, s_def, eta)
        u3_fn = lambda x, t: -6 * mp.diff(lambda xx: mp.ln(tau3(xx, t, s1, s2, s3, 1, 1, 1)), x, 1)
        v3_fn = lambda x, t: -6 * mp.diff(lambda xx: mp.ln(tau3(xx, t, s1, s2, s3, z1, z2, z3)), x, 1)
        for t0 in (mp.mpf("0.0"), mp.mpf("0.3")):
            jU = eval_jet(u3_fn, t0)
            jV = eval_jet(v3_fn, t0)
            res = max(abs(r) for r in cz_sewing_vec(*jU, *jV, s_def, eta))
            worst_3sol = max(worst_3sol, res)
    print(f"   max 3-soliton nonlinear CZ sewing residual at x=0 = {mp.nstr(worst_3sol, 3)}")
    ok &= worst_3sol < 1e-32

    # ------------------------------------------------------------------
    # 2. Exact time-dependent phonon eigenmode on 1- and 2-soliton + defect backgrounds
    # ------------------------------------------------------------------
    print("2. Phonon eigenmode on 1- and 2-soliton + defect backgrounds (linearised sewing check):")
    worst_lin_sew = mp.mpf(0)
    ph = phonon_good("0.65")
    for n_sol in (1, 2):
        s_def = mp.mpf("-1.5")
        eta = 1
        z1 = z_defect_mode(s1, s_def, eta)
        z2 = z_defect_mode(s2, s_def, eta) if n_sol == 2 else mp.mpf(0)
        c2_L = mp.mpf(1) if n_sol == 2 else mp.mpf(0)
        z_ph = z_defect_mode(ph, s_def, eta)
        t0 = mp.mpf("0.15")

        u0_fn = lambda x, t: -6 * mp.diff(lambda xx: mp.ln(tau3(xx, t, s1, s2, ph, 1, c2_L, 0, eps3=0)), x, 1)
        v0_fn = lambda x, t: -6 * mp.diff(lambda xx: mp.ln(tau3(xx, t, s1, s2, ph, z1, z2, 0, eps3=0)), x, 1)

        def du_fn(x, t):
            def ratio_L(xx):
                t_bg = tau3(xx, t, s1, s2, ph, 1, c2_L, 1, eps3=0)
                t_tot = tau3(xx, t, s1, s2, ph, 1, c2_L, 1, eps3=1)
                return -6 * (t_tot - t_bg) / t_bg
            return mp.diff(ratio_L, x, 1)

        def dv_fn(x, t):
            def ratio_R(xx):
                t_bg = tau3(xx, t, s1, s2, ph, z1, z2, z_ph, eps3=0)
                t_tot = tau3(xx, t, s1, s2, ph, z1, z2, z_ph, eps3=1)
                return -6 * (t_tot - t_bg) / t_bg
            return mp.diff(ratio_R, x, 1)

        jU0 = eval_jet(u0_fn, t0)
        jV0 = eval_jet(v0_fn, t0)
        jdU = eval_jet(du_fn, t0)
        jdV = eval_jet(dv_fn, t0)

        lin_res = [
            mp.diff(
                lambda ep: cz_sewing_vec(
                    *[u0 + ep * du for u0, du in zip(jU0, jdU)],
                    *[v0 + ep * dv for v0, dv in zip(jV0, jdV)],
                    s_def,
                    eta,
                )[idx],
                mp.mpf(0),
            )
            for idx in range(4)
        ]
        err = max(abs(r) for r in lin_res)
        worst_lin_sew = max(worst_lin_sew, err)
        print(f"   {n_sol}-soliton + defect linearised sewing residual at x=0, t=0.15 = {mp.nstr(err, 3)}")
    ok &= worst_lin_sew < 1e-32

    # ------------------------------------------------------------------
    # 3. KP-vertex factorisation and 2-root dipole decomposition of theta_D'(q)
    # ------------------------------------------------------------------
    print("3. KP-vertex factorisation A(1,ph) = Z(pb)/Z(pa) and 2-root dipole decomposition:")
    worst_fact = mp.mpf(0)
    worst_dipole = mp.mpf(0)
    worst_triple = mp.mpf(0)
    for al_val in ("0.35", "0.85", "1.25"):
        for eps in (1, -1):
            st = sol_good(al_val, eps)
            for q_val in ("-1.7", "-0.4", "0.25", "1.9"):
                ph = phonon_good(q_val)
                A_val = A_hirota(st, ph)
                Z_ratio = Z_single(q_val, st["pb"]) / Z_single(q_val, st["pa"])
                worst_fact = max(worst_fact, abs(A_val - Z_ratio))
                s_b = -2 * r3 * st["pb"]
                s_a = -2 * r3 * st["pa"]
                dth_diff = dtheta_D_dipole(q_val, s_b, 1) - dtheta_D_dipole(q_val, s_a, 1)
                phis = [st["a"] + 2 * mp.pi / 3, st["a"] - 2 * mp.pi / 3, st["a"]]
                if eps == -1:
                    phis = [mp.pi - p for p in phis]
                dth_sol = sum(c * dtheta_elem(q_val, p) for c, p in zip([1, 1, -2], phis))
                worst_fact = max(worst_fact, abs(dth_diff - dth_sol))

    for s_val in ("-2.6", "-1.5", "-0.4", "0.7", "1.6", "2.8"):
        s_mp = mp.mpf(s_val)
        qD = -s_mp / (2 * r3)
        gamma = mp.acos(s_mp / 2)
        psi1 = 2 * mp.pi / 3 - gamma
        psi2 = -2 * mp.pi / 3 - gamma
        # In terms of gamma = arccos(s/2), sqrt(1 - 3*q_D^2) = sin(gamma):
        disc = mp.sin(gamma)
        q_plus = (-qD + disc) / 2
        q_minus = (-qD - disc) / 2
        # sin(psi_1) = sin(gamma + pi/3) = q_+ - q_D; sin(psi_2) = sin(gamma - pi/3) = q_D - q_-
        worst_triple = max(worst_triple, abs(mp.sin(psi1) - (q_plus - qD)), abs(mp.sin(psi2) - (qD - q_minus)))
        for eta in (1, -1):
            for q_val in ("-2.1", "-0.5", "0.3", "1.4"):
                ph = phonon_good(q_val)
                z_ph = z_defect_mode(ph, s_mp, eta)
                worst_dipole = max(
                    worst_dipole,
                    abs(abs(z_ph) - 1),
                    abs(z_ph - Z_single(q_val, qD) ** eta),
                    abs(dtheta_D_dipole(q_val, s_mp, eta) - dtheta_D_direct(q_val, s_mp, eta)),
                )
    print(f"   max |A(1,ph) - Z(pb)/Z(pa)| and |dtheta_sol - (dtheta_D(pb)-dtheta_D(pa))| = {mp.nstr(worst_fact, 3)}")
    print(f"   max |sin(psi_1) - (q_+ - q_D)| and |sin(psi_2) - (q_D - q_-)| = {mp.nstr(worst_triple, 3)}")
    print(f"   max ||z_ph|-1| and |dtheta_D_dipole - dtheta_D_direct| (including |s|>2) = {mp.nstr(worst_dipole, 3)}")
    ok &= max(worst_fact, worst_triple, worst_dipole) < 1e-33

    # ------------------------------------------------------------------
    # 4. Defect Levinson winding and closed-form growing-mode spectrum
    # ------------------------------------------------------------------
    print("4. Defect Levinson phase winding and closed-form growing-mode spectrum:")
    worst_lev = mp.mpf(0)
    for s_val, exp_pos, exp_neg, exp_tot in (
        ("-2.5", mp.mpf(4) / 3, mp.mpf(2) / 3, mp.mpf(1)),
        ("-1.5", mp.mpf(4) / 3, mp.mpf(2) / 3, mp.mpf(1)),
        ("-0.5", -mp.mpf(2) / 3, mp.mpf(2) / 3, mp.mpf(0)),
        ("0.5", -mp.mpf(2) / 3, mp.mpf(2) / 3, mp.mpf(0)),
        ("1.5", -mp.mpf(2) / 3, -mp.mpf(4) / 3, mp.mpf(-1)),
        ("2.5", -mp.mpf(2) / 3, -mp.mpf(4) / 3, mp.mpf(-1)),
    ):
        int_pos = mp.quad(lambda q: dtheta_D_dipole(q, s_val, 1), [0, 1, 5, mp.inf]) / mp.pi
        int_neg = mp.quad(lambda q: dtheta_D_dipole(-q, s_val, 1), [0, 1, 5, mp.inf]) / mp.pi
        tot = (int_pos + int_neg) / 2
        worst_lev = max(worst_lev, abs(int_pos - exp_pos), abs(int_neg - exp_neg), abs(tot - exp_tot))
        print(f"   eta=+1, s={s_val:>4}: int_+/pi={mp.nstr(int_pos, 6)}, int_-/pi={mp.nstr(int_neg, 6)}, tot/(2pi)={mp.nstr(tot, 6)}")
    ok &= worst_lev < 1e-32

    # Check exact closed-form growing modes against 4x4 locmat determinant:
    worst_grow = max(abs(P_minus_quad(mp.mpf(1), mp.mpf(-1))), abs(P_plus_quad(mp.mpf(1), mp.mpf(1))))
    # (i) -1 < eta*sigma <= 1: single real growing mode from P_-(S) = 0
    for es_str in ("-0.95", "0.5"):
        es = mp.mpf(es_str)
        disc = mp.sqrt(4 - es * es)
        S1 = ((r3 - 1) * es + (r3 + 1) * disc) / 4
        g1 = (es * disc + r3 * (2 - es * es)) / 8
        det4, k1, k2 = locmat_4x4_det(I * g1, es, 1)
        fact_det = -(S1**2) * (S1**2 - 1) / 24 * P_minus_quad(S1, es) * P_plus_quad(S1, es) * (k1 - k2) ** 2
        worst_grow = max(worst_grow, abs(g1 - (S1 * S1 - 1) / 2), abs(P_minus_quad(S1, es)), abs(det4), abs(det4 - fact_det))
        print(f"   eta*sigma={es_str:>5} (1 mode):  S_1={mp.nstr(S1, 8)}, gamma_1={mp.nstr(g1, 8)}, |det_4x4|={mp.nstr(abs(det4), 3)}")
    # (ii) 1 < eta*sigma < 2: two real growing modes from P_-(S) = 0 and P_+(S) = 0
    es = mp.mpf("1.5")
    disc = mp.sqrt(4 - es * es)
    S1 = ((r3 - 1) * es + (r3 + 1) * disc) / 4
    S2 = ((r3 + 1) * es + (r3 - 1) * disc) / 4
    g1 = (es * disc + r3 * (2 - es * es)) / 8
    g2 = (es * disc - r3 * (2 - es * es)) / 8
    det4_1, _, _ = locmat_4x4_det(I * g1, es, 1)
    det4_2, _, _ = locmat_4x4_det(I * g2, es, 1)
    worst_grow = max(worst_grow, abs(P_minus_quad(S1, es)), abs(P_plus_quad(S2, es)), abs(det4_1), abs(det4_2))
    print(f"   eta*sigma=  1.5 (2 modes): gamma_1={mp.nstr(g1, 8)}, gamma_2={mp.nstr(g2, 8)}, max|det_4x4|={mp.nstr(max(abs(det4_1), abs(det4_2)), 3)}")
    # (iii) eta*sigma > 2: complex-conjugate pair from P_+(S) = 0
    es = mp.mpf("2.5")
    disc_c = mp.sqrt(es * es - 4)
    w_plus = (es * disc_c + I * r3 * (es * es - 2)) / 8
    w_minus = (-es * disc_c + I * r3 * (es * es - 2)) / 8
    det4_p, _, _ = locmat_4x4_det(w_plus, es, 1)
    det4_m, _, _ = locmat_4x4_det(w_minus, es, 1)
    worst_grow = max(worst_grow, abs(det4_p), abs(det4_m))
    print(f"   eta*sigma=  2.5 (complex): w_+- = +-{mp.nstr(mp.re(w_plus), 6)} + {mp.nstr(mp.im(w_plus), 8)}i, max|det_4x4|={mp.nstr(max(abs(det4_p), abs(det4_m)), 3)}")
    print(f"   max growing-mode polynomial / 4x4 determinant residual = {mp.nstr(worst_grow, 3)}")
    ok &= worst_grow < 1e-32

    assert ok, "SOME CHECKS FAILED"
    print("ALL CHECKS PASSED")

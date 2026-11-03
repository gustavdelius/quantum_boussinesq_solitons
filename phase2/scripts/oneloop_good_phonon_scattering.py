"""Task 1: Linearisation about the good-Boussinesq one-soliton and phonon scattering.

Equation (good Boussinesq):
    u_tt = u_xx - u_xxxx - 3 (u^2)_xx,   u = phi_x = 2 (log tau)_xx,
    L = 1/2 phi_t^2 - 1/2 phi_x^2 + phi_x^3 - 1/2 phi_xx^2.

One-soliton:
    k = sin(alpha),  v = eps cos(alpha),  ell = -eps cos(3 alpha),
    M = 2 k,  Pi = -M v,  P = (2/3) eps k^3 cos(alpha),
    u_0(xi) = (k^2 / 2) sech^2(k xi / 2),  xi = x - v t.

Checks performed (at 35-digit working precision):
 1. Fourth-order co-moving dispersion q^4 + k_1^2 q^2 - 2 v_1 tilde_w q - tilde_w^2 = 0
    is strictly convex in q for tilde_w != 0: exactly 2 real propagating roots and
    1 complex-conjugate evanescent pair.
 2. Exact Hirota tau-linearisation eigenmode delta_u(x, t; q) satisfies the full
    fourth-order linearised PDE H_u[delta_u] = 0 to < 10^{-30}, and is reflectionless
    (R(q) = 0, C_+(q) = C_-(q) = 0) with transmission amplitude A(1, ph(q)).
 3. KP root representation A(1, ph(q)) = (z_a / z_b) / conj(z_a / z_b) proves unitarity
    |A(1, ph(q))| = 1 for all real q != 0 (no unstable phonon band).
 4. Exact 3-root arctan decomposition of the unwrapped phase shift theta(q; alpha, eps)
    and exact rational/hyperbolic formulas for theta'(q; alpha, eps) +- theta'(-q; alpha, eps),
    verified against direct differentiation to < 10^{-33}.
 5. Levinson-type mode count N_Lev = -(Delta theta_+ + Delta theta_-) / (2 pi):
    N_Lev = 1 in the regular region alpha < pi/3, and N_Lev = 2 on the heavy branch alpha > pi/3.
 6. Discrete spectrum from KP triple {p_a, p_b, p_c} zeros of A(1, K):
    - Translational zero mode K = k_1 (tilde_w = 0) and secular partner d_P u_0.
    - Secondary modes K_{cb} = sin(alpha - pi/3), K_{ac} = sin(alpha + pi/3) with
      tilde_w = +-i sqrt(3) (sin^2(alpha) - 3/4): non-normalisable for alpha < pi/3,
      and normalisable exponentially growing/decaying linear instability modes for
      alpha > pi/3, verified directly in the linearised PDE to < 10^{-30}.
"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 35
r3 = mp.sqrt(3)
ok = True


def sol_good(alpha, eps=1):
    a = mp.mpf(alpha)
    e = mp.mpf(eps)
    k = mp.sin(a)
    v = e * mp.cos(a)
    ell = -e * mp.cos(3 * a)
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * e * k**3 * mp.cos(a)
    dP_da = mp.mpf(2) / 3 * e * mp.sin(a) * mp.sin(3 * a)
    pa = (k + v / r3) / 2
    pb = (v / r3 - k) / 2
    pc = -v / r3
    return dict(
        a=a, eps=e, k=k, v=v, ell=ell, M=M, Pi=Pi, P=P, dP_da=dP_da,
        pa=pa, pb=pb, pc=pc
    )


def phonon_good(q):
    qq = mp.mpf(q)
    s = 1 if qq >= 0 else -1
    R = mp.sqrt(1 + qq**2)
    w = abs(qq) * R
    p_re = s * R / (2 * r3)
    p_im = qq / 2
    return dict(q=qq, s=s, w=w, pa=p_re + 1j * p_im, pb=p_re - 1j * p_im)


def A_sol_ph(s1, ph):
    return (
        (s1["pa"] - ph["pa"]) * (s1["pb"] - ph["pb"])
        / ((s1["pa"] - ph["pb"]) * (s1["pb"] - ph["pa"]))
    )


def theta_unwrapped(q, alpha, eps=1):
    """Exact unwrapped transmission phase shift theta(q; alpha, eps) with theta(+-oo) = 0."""
    s1 = sol_good(alpha, eps)
    qq = mp.mpf(q)
    s = 1 if qq > 0 else -1
    R = mp.sqrt(1 + qq**2)
    zb = (s1["pb"] - s * R / (2 * r3)) - 1j * qq / 2
    w_c = 1 + s1["k"] / zb
    # Principal arg of w_c is in (0, pi) for q > 0 and (-pi, 0) for q < 0:
    # To match the continuous branch on (0, oo) and (-oo, 0) with theta(+-oo) = 0:
    return 2 * mp.atan2(mp.im(w_c), mp.re(w_c))


def theta_3root(q, alpha, eps=1):
    """Closed-form 3-root arctan decomposition of theta(q; alpha, eps)."""
    a = mp.mpf(alpha)
    e = mp.mpf(eps)
    qq = mp.mpf(q)
    s = 1 if qq > 0 else -1
    aq = abs(qq)
    R = mp.sqrt(1 + qq**2)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    return -sum(
        c * (
            s * mp.atan(mp.sin(ph) / aq)
            + e * mp.atan(mp.sin(ph) * mp.cos(ph) / (aq * R + qq**2 + mp.sin(ph)**2))
        )
        for c, ph in zip(cs, phis)
    )


def dtheta_dq(q, alpha, eps=1):
    """Exact derivative dtheta(q; alpha, eps)/dq."""
    a = mp.mpf(alpha)
    e = mp.mpf(eps)
    qq = mp.mpf(q)
    R = mp.sqrt(1 + qq**2)
    phis = [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    cs = [1, 1, -2]
    even_part = sum(c * mp.sin(ph) / (qq**2 + mp.sin(ph)**2) for c, ph in zip(cs, phis))
    odd_part = (
        (e if qq > 0 else -e)
        / R
        * sum(c * mp.sin(ph) * mp.cos(ph) / (qq**2 + mp.sin(ph)**2) for c, ph in zip(cs, phis))
    )
    return even_part + odd_part


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Co-moving dispersion roots: 2 real propagating + 1 evanescent pair
    # ------------------------------------------------------------------
    print("1. Co-moving dispersion root structure (2 real + 2 complex-conjugate evanescent):")
    worst_disp = mp.mpf(0)
    for a_val in ("0.3", "0.7", "1.2"):
        s1 = sol_good(a_val, 1)
        for q_in in ("0.4", "1.5", "-0.8"):
            ph = phonon_good(q_in)
            tw = ph["w"] - s1["v"] * ph["q"]
            # Polynomial q^4 + k1^2 q^2 - 2 v1 tw q - tw^2 = 0
            roots = mp.polyroots([1, 0, s1["k"]**2, -2 * s1["v"] * tw, -tw**2])
            real_roots = [r for r in roots if abs(mp.im(r)) < 1e-25]
            evan_roots = [r for r in roots if abs(mp.im(r)) >= 1e-25]
            ok &= len(real_roots) == 2 and len(evan_roots) == 2
            ok &= abs(evan_roots[0] - mp.conj(evan_roots[1])) < 1e-28
            min_dist = min(abs(r - ph["q"]) for r in real_roots)
            worst_disp = max(worst_disp, min_dist)
    print(f"   max |q_root - q_in| = {mp.nstr(worst_disp, 3)} (2 real + 2 evanescent at all test points)")
    ok &= worst_disp < 1e-30

    # ------------------------------------------------------------------
    # 2. Exact Hirota linearised eigenmode PDE check and reflectionlessness
    # ------------------------------------------------------------------
    print("2. Hirota linearised eigenmode PDE residual and reflectionlessness:")
    x_s, t_s, k_s, v_s, q_s, w_s, A_s = sp.symbols("x t k v q w A")
    E1 = sp.exp(k_s * (x_s - v_s * t_s))
    u0_sym = 2 * sp.diff(sp.log(1 + E1), x_s, 2)
    f_ph = (1 + A_s * E1) / (1 + E1) * sp.exp(sp.I * (q_s * x_s - w_s * t_s))
    du_sym = 2 * sp.diff(f_ph, x_s, 2)
    pde_lhs = sp.diff(du_sym, t_s, 2) - sp.diff(du_sym, x_s, 2) + sp.diff(du_sym, x_s, 4) + 6 * sp.diff(u0_sym * du_sym, x_s, 2)

    worst_pde = mp.mpf(0)
    for a_val, eps_val in (("0.4", 1), ("0.85", -1), ("1.25", 1)):
        s1 = sol_good(a_val, eps_val)
        for q_val in ("0.5", "-1.1"):
            ph = phonon_good(q_val)
            A_val = A_sol_ph(s1, ph)
            # Numeric fourth-order PDE check at (x, t) = (0.37, 0.19) using mpmath derivatives
            x0, t0 = mp.mpf("0.37"), mp.mpf("0.19")
            def u0_fn(xx, tt):
                th = s1["k"] * (xx - s1["v"] * tt)
                return s1["k"]**2 / (2 * mp.cosh(th / 2)**2)
            def du_fn(xx, tt):
                def g(xxx):
                    e1 = mp.exp(s1["k"] * (xxx - s1["v"] * tt))
                    return (1 + A_val * e1) / (1 + e1) * mp.exp(1j * (ph["q"] * xxx - ph["w"] * tt))
                return 2 * mp.diff(g, xx, 2)
            utt = mp.diff(lambda tt: du_fn(x0, tt), t0, 2)
            uxx = mp.diff(lambda xx: du_fn(xx, t0), x0, 2)
            uxxxx = mp.diff(lambda xx: du_fn(xx, t0), x0, 4)
            u0du_xx = mp.diff(lambda xx: 6 * u0_fn(xx, t0) * du_fn(xx, t0), x0, 2)
            res = abs(utt - uxx + uxxxx + u0du_xx)
            worst_pde = max(worst_pde, res)
    print(f"   max linearised PDE residual |H_u[delta_u]| = {mp.nstr(worst_pde, 3)}")
    ok &= worst_pde < 1e-28

    # ------------------------------------------------------------------
    # 3. Unitarity |A(1, ph(q))| = 1 and 3-root arctan/rational identities
    # ------------------------------------------------------------------
    print("3. Unitarity |A(1, ph(q))| = 1 and 3-root phase-shift identities:")
    worst_unit = mp.mpf(0)
    worst_th = mp.mpf(0)
    worst_dth = mp.mpf(0)
    for a_val in ("0.25", "0.6", "0.95", "1.2"):
        for eps_val in (1, -1):
            s1 = sol_good(a_val, eps_val)
            for q_val in ("0.15", "0.8", "2.7", "-0.25", "-1.4", "-3.2"):
                ph = phonon_good(q_val)
                A_val = A_sol_ph(s1, ph)
                worst_unit = max(worst_unit, abs(abs(A_val) - 1))
                th1 = theta_unwrapped(q_val, a_val, eps_val)
                th2 = theta_3root(q_val, a_val, eps_val)
                worst_th = max(worst_th, abs(th1 - th2))
                ok &= abs(mp.exp(1j * th1) - A_val) < 1e-32
                dth_ana = dtheta_dq(q_val, a_val, eps_val)
                dth_num = mp.diff(lambda qq: theta_3root(qq, a_val, eps_val), mp.mpf(q_val))
                worst_dth = max(worst_dth, abs(dth_ana - dth_num))
    print(f"   max ||A| - 1| = {mp.nstr(worst_unit, 3)}")
    print(f"   max |theta_KP - theta_3root| = {mp.nstr(worst_th, 3)}")
    print(f"   max |dtheta_exact - dtheta_num| = {mp.nstr(worst_dth, 3)}")
    ok &= worst_unit < 1e-33 and worst_th < 1e-33 and worst_dth < 1e-32

    # ------------------------------------------------------------------
    # 4. Levinson-type mode count across the cusp alpha = pi/3
    # ------------------------------------------------------------------
    print("4. Levinson mode count N_Lev = -(Delta theta_+ + Delta theta_-)/(2 pi):")
    for a_val, expected_N in (("0.2", 1), ("0.5", 1), ("0.9", 1), ("1.15", 2), ("1.35", 2)):
        a_mp = mp.mpf(a_val)
        # Quadrature of dtheta/dq over (-oo, 0) and (0, oo)
        dth_pos = mp.quad(lambda q: dtheta_dq(q, a_mp, 1), [0, 1, 10, mp.inf])
        dth_neg = mp.quad(lambda q: dtheta_dq(-q, a_mp, 1), [0, 1, 10, mp.inf])
        N_lev = -(dth_pos + dth_neg) / (2 * mp.pi)
        # Closed-form Lorentzian sum: -(1/2) sum_r c_r sgn(sin(phi_r))
        phis = [a_mp + 2 * mp.pi / 3, a_mp - 2 * mp.pi / 3, a_mp]
        cs = [1, 1, -2]
        N_closed = -mp.mpf(1) / 2 * sum(c * (1 if mp.sin(ph) > 0 else -1) for c, ph in zip(cs, phis))
        err = max(abs(N_lev - expected_N), abs(N_closed - expected_N))
        print(
            f"   alpha={a_val:<4s} ({'regular < pi/3' if a_mp < mp.pi/3 else 'heavy   > pi/3'}): "
            f"Delta theta_+/(2pi)={float(dth_pos/(2*mp.pi)):+.6f}, "
            f"Delta theta_-/(2pi)={float(dth_neg/(2*mp.pi)):+.6f}, "
            f"N_Lev={float(N_lev):.10f} (exact {int(N_closed)})"
        )
        ok &= err < 1e-30

    # ------------------------------------------------------------------
    # 5. Discrete spectrum: zero mode (all alpha) and heavy-branch instability (alpha > pi/3)
    # ------------------------------------------------------------------
    print("5. Discrete spectrum (translational zero mode and heavy-branch instability modes):")
    worst_disc = mp.mpf(0)
    for a_val in ("0.4", "0.8", "1.15", "1.3"):
        s1 = sol_good(a_val, 1)
        xi0 = mp.mpf("0.42")
        u0_xi = lambda xi: s1["k"]**2 / (2 * mp.cosh(s1["k"] * xi / 2)**2)
        # Co-moving static operator L_static[f] = d_xi^4 f - (1 - v1^2) d_xi^2 f + 6 d_xi^2(u0 f)
        def L_static(f, xi):
            return (
                mp.diff(f, xi, 4)
                - s1["k"]**2 * mp.diff(f, xi, 2)
                + 6 * mp.diff(lambda x: u0_xi(x) * f(x), xi, 2)
            )
        # Translational zero mode f_0(xi) = u0'(xi)
        f0 = lambda xi: mp.diff(u0_xi, xi)
        res_zero = abs(L_static(f0, xi0))
        # Secular partner f_sec(xi, t) = t * d_v u0 ? More precisely: d_alpha u0(x - v t):
        # d/dalpha [u0(xi)]_{xi = x - v t} = (d_alpha u0)(xi) + (d_alpha k / d_alpha v ...):
        # At t = 0, u_alpha(xi) = (d u0 / d alpha) satisfies L_static[u_alpha] = -2 v1 (dv1/dalpha) u0''(xi)
        u_alpha = lambda xi: mp.diff(lambda aa: sol_good(aa, 1)["k"]**2 / (2 * mp.cosh(sol_good(aa, 1)["k"] * xi / 2)**2), s1["a"])
        rhs_sec = -2 * s1["v"] * (-mp.sin(s1["a"])) * mp.diff(u0_xi, xi0, 2)
        res_sec = abs(L_static(u_alpha, xi0) - rhs_sec)
        worst_disc = max(worst_disc, res_zero, res_sec)

        # Secondary KP-triple modes K_cb = sin(a - pi/3) and K_ac = sin(a + pi/3) = k1 - K_cb
        K_cb = mp.sin(s1["a"] - mp.pi / 3)
        K_ac = mp.sin(s1["a"] + mp.pi / 3)
        gamma = r3 * (mp.sin(s1["a"])**2 - mp.mpf(3) / 4)
        # Both K_cb and K_ac produce the SAME spatial profile sech^2(k1 xi/2) exp((K_cb - k1/2) xi) up to x -> -x:
        # Specifically, f_inst(xi) = d_xi^2 [ exp(K_cb xi) / (1 + exp(k1 xi)) ]
        # solves the full linearised PDE with exp(+- gamma t) time dependence!
        for sign_t, K_mode in ((+1, K_cb), (-1, K_ac)):
            def du_inst(xx, tt):
                xi = xx - s1["v"] * tt
                return mp.exp(sign_t * gamma * tt) * mp.diff(
                    lambda y: mp.exp(K_mode * y) / (1 + mp.exp(s1["k"] * y)), xi, 2
                )
            t_test = mp.mpf("0.13")
            utt = mp.diff(lambda tt: du_inst(xi0, tt), t_test, 2)
            uxx = mp.diff(lambda xx: du_inst(xx, t_test), xi0, 2)
            uxxxx = mp.diff(lambda xx: du_inst(xx, t_test), xi0, 4)
            u0du = mp.diff(lambda xx: 6 * u0_xi(xx - s1["v"] * t_test) * du_inst(xx, t_test), xi0, 2)
            res_inst = abs(utt - uxx + uxxxx + u0du)
            worst_disc = max(worst_disc, res_inst)
        normalisable = (0 < K_cb < s1["k"]) and (0 < K_ac < s1["k"])
        ok &= normalisable == (s1["a"] > mp.pi / 3)
        print(
            f"   alpha={a_val:<4s}: K_cb={float(K_cb):+.5f}, K_ac={float(K_ac):+.5f}, "
            f"k1={float(s1['k']):.5f}, gamma={float(gamma):+.5f}, normalisable={normalisable}"
        )
    print(f"   max discrete-mode PDE residual = {mp.nstr(worst_disc, 3)}")
    ok &= worst_disc < 1e-28

    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    assert ok

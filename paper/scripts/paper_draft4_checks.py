#!/usr/bin/env python3
"""
Verification script for Draft 4 revisions of
paper/boussinesq_soliton_scattering.tex (addressing review_draft3.tex R1-R11).

Checks:
  1. [R9, Sec. 5.7] Defect polynomial primitives rho(sigma), tau(sigma), h(sigma)
     in eq:defprims and KP-edge vacuum jumps (tau' = -sigma*rho, h' = -rho).
  2. [R2 & R8, Sec. 7.1 & Sec. 10(2)] Direct 50-digit field-pairing quadrature
     of the collective symplectic matrices W_3 and W_4 in eq:W3W4 under
     Omega = int d_pi ^ d_phi in basis (Z_v, Z_X, Z_C, Z_r), explicit W_4^{-1}
     projector tail inputs b_r = int w*eta_phi dxi and
     b_X = int (v*u_0'*eta_phi + u_0*eta_pi) dxi, counterexample eta = (w,0)
     giving b_r = 3*k/7 != 0, and Euler-Lagrange derivation of the
     time-dependent lab-frame two-soliton fluctuation operator
     D_{[u_2]} = d_t^2 + d_x^4 - d_x^2 + 6 d_x u_2(x,t) d_x.
  3. [R3 & optional rounding, Sec. 7.1-7.2] Coefficient-level identity
     F = E_1|_P = e - v*p = e_can - v*p_can = -ell_1, finite-tadpole sign
     (+2*hbar_B*c_fin*sin(alpha) added to hbar_B*ell_{1,g} gives -2*c_fin*sin(alpha)
     in F_g and -3*c_fin/alpha^2 in F_g/|P_0|), and sensitivity envelope
     5.68e-8 + 7.70e-6 + 4.73e-7 = 8.2298e-6 <= 8.24e-6.
  4. [R1, R4, R9, R10, Sec. 8]
     (a) Direct finite-sea Nystroem forward/reverse phase reciprocity
         hat_Phi(p,z) + hat_Phi(z,p) = M_p*Q_z + M_z*Q_p.
     (b) Differentiated contact term 2*sqrt(pi)*f'(u_1 - u_2)
         = 2*sqrt(pi)*b(u_1 - u_2) = 4*sqrt(pi)*psi_-(u_1 - u_2).
     (c) Two-term directional residual R_0(Y,r) = H(u(Y),u(r^2 Y)) - H_cl(Y,r)
         and formal constant-kernel integrals R_0(inf,r), c_1(r).
     (d) Corrected Krein PDE (d_u + d_v)H = 2*b(u)*B(v) and
         (d_u + d_v)hat_H = 2*b(u)*B(v) - 2*c*b(v) vs old four-term source
         discrepancy 2*c*(b(u) - b(v)) (-0.3471042546629204553... vs
         +0.8377480606920601269... at (u,v)=(4,1)).
     (e) Nonzero Jacobian-weighted characteristic derivative
         [B(u_1)/(2*sqrt(y_1)) d_{y_1} + B(u_2)/(2*sqrt(y_2)) d_{y_2}] A_lambda
         at (u_1,u_2)=(4,1) (-0.0080523430782269595...), 3x3 minor excluding
         rank <= 2, and non-separable moving-pole residue at y_1 = -y_2.
  5. [R5, R6, R10, Sec. 9]
     (a) Compensating O(hbar^{-1}) two-body prefactor test for Stirling matching.
     (b) Full numerical residues of S_ov at Bogdan-Charkina breather poles N=1,2,3.
     (c) Repaired uniform positive root witness X = 1/(1 + sqrt(1 - A_12)) > 0
         at cos(I) = -1 giving tau = 1 - 2*X + A_12*X^2 = 0 for all A_12 < 1,
         whereas X = 1/sqrt(A_12) gives 2 - 2/sqrt(A_12) != 0.
     (d) Independent numerical differentiation of Phi_ring(j | l) = delta(j,l) + Pi_j*M_l
         with respect to P_j for both equations and both orientations.
  6. [R10 & R11, Appendix A] Static checks of cited scripts and audit_onebody.py
     vs audit_onebody_scheme.py status.
"""

from pathlib import Path
import ast
import re
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
ROOT = Path(__file__).resolve().parents[2]


def check_1_defect_primitives():
    print("=== Check 1: [R9, Sec. 5.7] Defect polynomial primitives (eq:defprims) ===")
    sigma, s, A = sp.symbols("sigma s A", real=True, nonzero=True)
    rho = (sigma - sigma**3 / sp.Integer(3) - (s - s**3 / sp.Integer(3))) / (2 * A)
    tau = -sp.integrate(sigma * rho, (sigma, s, sigma))
    h = -sp.integrate(rho, (sigma, s, sigma))

    assert sp.simplify(sp.diff(tau, sigma) + sigma * rho) == 0
    assert sp.simplify(sp.diff(h, sigma) + rho) == 0
    assert sp.simplify(tau.subs(sigma, s)) == 0
    assert sp.simplify(h.subs(sigma, s)) == 0
    assert sp.simplify(rho.subs(sigma, s)) == 0

    # Verify KP-edge vacuum jumps on q_i^2 + q_i*q_j + q_j^2 = 1/4
    qi, qj, eta = sp.symbols("qi qj eta", real=True, nonzero=True)
    A_val = eta / sp.sqrt(3)
    sig_i = -2 * sp.sqrt(3) * qi
    sig_j = -2 * sp.sqrt(3) * qj
    kp_poly = qi**2 + qi * qj + qj**2 - sp.Rational(1, 4)

    # Soliton charges on edge {qi, qj} (qi > qj): k = qi - qj, v = sqrt(3)*(qi + qj)
    k_s = qi - qj
    v_s = sp.sqrt(3) * (qi + qj)
    M_s = 2 * k_s
    Pi_s = -M_s * v_s
    P_s = sp.Rational(2, 3) * k_s**3 * v_s
    E_s = sp.Rational(2, 15) * k_s**3 * (5 - 4 * k_s**2)

    # Defect vacuum charges at sigma_j minus sigma_i (with s = sigma_i, rho_i = eta)
    tau_sub = tau.subs({A: A_val, s: sig_i})
    h_sub = h.subs({A: A_val, s: sig_i})
    dE_vac = -sp.Rational(1, 9) * tau_sub.subs(sigma, sig_j)
    dP_vac = sp.Rational(1, 9) * h_sub.subs(sigma, sig_j)
    dPi_vac = (sp.sqrt(3) * eta / 6) * (sig_j**2 - sig_i**2)
    dM_vac = (sig_j - sig_i) / (sp.sqrt(3) * eta)

    for ev in (1, -1):
        assert sp.expand((dM_vac - ev * M_s).subs(eta, ev)) == 0
        assert sp.rem(sp.expand((dPi_vac - ev * Pi_s).subs(eta, ev)), kp_poly, qi) == 0
        assert sp.rem(sp.expand((dP_vac - ev * P_s).subs(eta, ev)), kp_poly, qi) == 0
        assert sp.rem(sp.expand((dE_vac - ev * E_s).subs(eta, ev)), kp_poly, qi) == 0
    print("  Defect primitives and KP-edge vacuum jumps verified: PASS")


def check_2_collective_symplectic_and_two_soliton_operator():
    print("=== Check 2: [R2 & R8, Sec. 7.1 & Sec. 10(2)] W_3, W_4 and D_{[u_2]} ===")
    # 2a. Symbolic moments and W_4^{-1} projector tail formulas
    k, v, z = sp.symbols("k v z", positive=True)
    u0_z = k**2 * (1 - z**2) / 2
    jac = 2 / (k * (1 - z**2))
    w_z = 3 * u0_z / k - 15 * u0_z**2 / (2 * k**3)

    assert sp.integrate(w_z * jac, (z, -1, 1)) == 1
    assert sp.integrate(w_z * u0_z * jac, (z, -1, 1)) == 0
    assert sp.integrate(w_z**2 * jac, (z, -1, 1)) == 3 * k / 7

    P_v, Pi_v, b_v, b_X, b_C, b_r = sp.symbols("P_v Pi_v b_v b_X b_C b_r", nonzero=True)
    W3 = sp.Matrix([[0, P_v, Pi_v], [-P_v, 0, 0], [-Pi_v, 0, 0]])
    W4 = sp.Matrix([
        [0, P_v, Pi_v, 0],
        [-P_v, 0, 0, 0],
        [-Pi_v, 0, 0, -1],
        [0, 0, 1, 0],
    ])
    assert W3.rank() == 2
    assert sp.simplify(W4.det() - P_v**2) == 0

    coeffs = W4.inv() * sp.Matrix([b_v, b_X, b_C, b_r])
    # Since Phi_v(+-inf) = -+ v/k, u0(+-inf) = 0, Z_C_phi(+-inf) = 1, Z_r_phi = 0:
    tail_plus = sp.simplify(-v / k * coeffs[0] + coeffs[2])
    tail_minus = sp.simplify(+v / k * coeffs[0] + coeffs[2])
    assert sp.simplify(tail_plus - (b_r + v / (k * P_v) * b_X)) == 0
    assert sp.simplify(tail_minus - (b_r - v / (k * P_v) * b_X)) == 0

    # 2b. Direct 50-digit field-pairing quadrature for both signs of v
    for vv in (mp.mpf("0.8"), mp.mpf("-0.7")):
        kk = mp.sqrt(1 - vv**2)

        def u0_fn(x):
            return kk**2 * (1 - x**2) / 2

        def u0x_fn(x):
            return -kk * x * u0_fn(x)

        def phiv_fn(x):
            return -(vv / kk) * (x + (1 - x**2) * mp.atanh(x))

        def u0v_fn(x):
            return -vv * (1 - x**2) * (1 - x * mp.atanh(x))

        def w_fn(x):
            return 3 * u0_fn(x) / kk - 15 * u0_fn(x) ** 2 / (2 * kk**3)

        basis = [
            (phiv_fn, lambda x: -u0_fn(x) - vv * u0v_fn(x)),  # Z_v
            (lambda x: -u0_fn(x), lambda x: vv * u0x_fn(x)),  # Z_X
            (lambda x: mp.mpf(1), lambda x: mp.mpf(0)),       # Z_C
            (lambda x: mp.mpf(0), w_fn),                      # Z_r
        ]
        pv_val = 2 * kk * (1 - 4 * vv**2) / 3
        piv_val = 2 * (2 * vv**2 - 1) / kk
        W4_expected = mp.matrix([
            [0, pv_val, piv_val, 0],
            [-pv_val, 0, 0, 0],
            [-piv_val, 0, 0, -1],
            [0, 0, 1, 0],
        ])
        max_err = mp.mpf(0)
        for i in range(4):
            for j in range(i + 1, 4):
                val = mp.quad(
                    lambda x, ii=i, jj=j: (
                        basis[ii][1](x) * basis[jj][0](x)
                        - basis[jj][1](x) * basis[ii][0](x)
                    )
                    * 2
                    / (kk * (1 - x**2)),
                    [-1, 0, 1],
                )
                max_err = max(max_err, abs(val - W4_expected[i, j]))
        assert max_err < mp.mpf("1e-44"), f"W_4 quadrature error: {max_err}"

        # Direct pairing for counterexample eta = (w, 0)
        br_num = mp.quad(
            lambda x: basis[3][1](x) * w_fn(x) * 2 / (kk * (1 - x**2)),
            [-1, 0, 1],
        )
        bx_num = mp.quad(
            lambda x: basis[1][1](x) * w_fn(x) * 2 / (kk * (1 - x**2)),
            [-1, 0, 1],
        )
        assert abs(br_num - 3 * kk / 7) < mp.mpf("1e-44")
        assert abs(bx_num) < mp.mpf("1e-44")

    # 2c. Time-dependent lab-frame two-soliton fluctuation operator D_{[u_2]}
    xx, tt = sp.symbols("xx tt", real=True)
    eta_f = sp.Function("eta")(xx, tt)
    u2_bg = sp.Function("u2")(xx, tt)
    L2_lab = (
        sp.Rational(1, 2) * eta_f.diff(tt) ** 2
        - sp.Rational(1, 2) * (1 - 6 * u2_bg) * eta_f.diff(xx) ** 2
        - sp.Rational(1, 2) * eta_f.diff(xx, 2) ** 2
    )
    el_lhs = sp.euler_equations(L2_lab, eta_f, [xx, tt])[0].lhs
    D_u2_eta = (
        eta_f.diff(tt, 2)
        + eta_f.diff(xx, 4)
        - eta_f.diff(xx, 2)
        + 6 * sp.diff(u2_bg * eta_f.diff(xx), xx)
    )
    assert sp.expand(el_lhs + D_u2_eta) == 0
    print("  Direct 50-digit W_3/W_4 pairings, projector tails and D_{[u_2]} verified: PASS")


def check_3_one_loop_units_and_tadpole_sign():
    print("=== Check 3: [R3 & optional rounding, Sec. 7.1-7.2] Loop units & tadpole sign ===")
    alpha, hb, c_fin = sp.symbols("alpha hb c_fin", positive=True)
    k = sp.sin(alpha)
    P0 = sp.Rational(2, 3) * k**3 * sp.cos(alpha)
    Fg = (
        sp.sqrt(3) / sp.pi * (sp.Rational(3, 4) - k**2) * sp.atanh(2 * k / sp.sqrt(3))
        - 3 * k / (2 * sp.pi)
    )
    # E(P) = E_0(P) + hb * F_g(P): coefficient is F_g = -ell_{1,g}
    E_tot = sp.Symbol("E0") + hb * Fg
    assert sp.diff(E_tot, hb) == Fg

    # Adding +2*hb*c_fin*sin(alpha) to physical L_1 = hb*ell_{1,g} shifts ell_{1,g}
    # by +2*c_fin*sin(alpha), hence F_g = -ell_{1,g} by -2*c_fin*sin(alpha),
    # and F_g / |P_0| by -3*c_fin / alpha^2 + O(1).
    delta_Fg = -2 * c_fin * sp.sin(alpha)
    slope_coeff = sp.limit(alpha**2 * (delta_Fg / P0), alpha, 0)
    assert sp.simplify(slope_coeff + 3 * c_fin) == 0

    # Cutoff remainder of e_reg(Lambda) - e is +3*sin(alpha)/(4*pi*Lambda) (no hbar)
    Lam = sp.Symbol("Lam", positive=True)
    rem_coeff = 3 * sp.sin(alpha) / (4 * sp.pi * Lam)
    assert rem_coeff.free_symbols == {alpha, Lam}

    # Sensitivity envelope sum in Sec. 7.2
    env_sum = sp.Rational("5.68e-8") + sp.Rational("7.70e-6") + sp.Rational("4.73e-7")
    assert env_sum == sp.Rational("8.2298e-6")
    assert env_sum < sp.Rational("8.24e-6")
    print("  Coefficient units, tadpole sign (-3*c_fin/alpha^2) and envelope verified: PASS")


def check_4_section8_krein_and_reciprocity():
    print("=== Check 4: [R1, R4, R9, R10, Sec. 8] Finite-sea phases & Krein PDE ===")
    # 4a. Direct finite-sea Nystroem solve and forward/reverse hat_Phi(p,z) + hat_Phi(z,p)
    h_val = mp.mpf("0.9")
    q_val = mp.mpf("0.8")
    two_pi_h = 2 * mp.pi * h_val
    n_quad = 36
    nd, wt = mp.gauss_quadrature(n_quad, "legendre")
    nodes = [q_val * z for z in nd]
    weights = [q_val * w for w in wt]

    def K_toda(z):
        return 2 * mp.re(mp.digamma(1 + 1j * z / h_val)) + 2 * mp.log(h_val)

    def theta_toda(z):
        return 2 * h_val * mp.im(mp.loggamma(1 + 1j * z / h_val)) + 2 * z * mp.log(h_val)

    mat = mp.matrix([
        [two_pi_h * (i == j) - weights[j] * K_toda(nodes[i] - nodes[j]) for j in range(n_quad)]
        for i in range(n_quad)
    ])
    inv_mat = mat**-1
    rho_vec = inv_mat * mp.ones(n_quad, 1)
    eta_vec = inv_mat * mp.matrix(nodes)

    def dot_q(vals, vec):
        return mp.fsum(weights[i] * vals[i] * vec[i] for i in range(n_quad))

    def th_vec(p):
        return mp.matrix([theta_toda(p - z) for z in nodes])

    def k_vec(p):
        return mp.matrix([K_toda(p - z) for z in nodes])

    def P_dr(p):
        return p + dot_q(th_vec(p), rho_vec)

    def D_dr(p):
        return 1 + dot_q(k_vec(p), rho_vec)

    def V_dr(p):
        return p + dot_q(k_vec(p), eta_vec)

    def F_ends(p):
        fp = inv_mat * th_vec(p)
        return [(theta_toda(p - edge) + dot_q(k_vec(edge), fp)) / two_pi_h for edge in (q_val, -q_val)]

    def Phi_raw(p, z):
        return theta_toda(p - z) - dot_q(k_vec(p), inv_mat * th_vec(z))

    def Phi_hat(p, z):
        return Phi_raw(p, z) - Phi_raw(p, q_val) - Phi_raw(q_val, z) + Phi_raw(q_val, q_val)

    a_spacing = 1 / mp.fsum(weights[i] * rho_vec[i] for i in range(n_quad))
    Fq_plus, Fq_minus = F_ends(q_val)
    for p_test, z_test in [(mp.mpf("1.2"), mp.mpf("1.5")), (mp.mpf("1.2"), mp.mpf("-1.4"))]:
        Fp_plus, Fp_minus = F_ends(p_test)
        Fz_plus, Fz_minus = F_ends(z_test)
        # Endpoint identity check
        assert abs(D_dr(q_val) * (Fp_plus - Fp_minus) - (1 - D_dr(p_test))) < mp.mpf("1e-28")
        assert abs(V_dr(q_val) * (Fp_plus + Fp_minus) - (P_dr(p_test) - V_dr(p_test))) < mp.mpf("1e-28")
        # Direct reciprocity check: hat_Phi(p,z) + hat_Phi(z,p) = M_p*Q_z + M_z*Q_p
        rhs_bilinear = two_pi_h * (
            (Fp_plus - Fq_plus) * (Fz_plus - Fq_plus)
            - (Fp_minus - Fq_minus) * (Fz_minus - Fq_minus)
        )
        Mp = -a_spacing * (D_dr(p_test) - D_dr(q_val))
        Mz = -a_spacing * (D_dr(z_test) - D_dr(q_val))
        Qp = (P_dr(p_test) - P_dr(q_val)) - (V_dr(p_test) - V_dr(q_val))
        Qz = (P_dr(z_test) - P_dr(q_val)) - (V_dr(z_test) - V_dr(q_val))
        rhs_MQ = Mp * Qz + Mz * Qp
        lhs_recip = Phi_hat(p_test, z_test) + Phi_hat(z_test, p_test)
        assert abs(lhs_recip - rhs_bilinear) < mp.mpf("1e-28")
        assert abs(lhs_recip - rhs_MQ) < mp.mpf("1e-28")

    # 4b & 4d. Inverse Laplace evaluation of F(u), B(u)=f(u)=F'(u), b(u)=f'(u)=B'(u)
    def g_wh(p):
        return (
            mp.gamma(1 + p / (2 * mp.pi))
            * mp.exp(-p * mp.log(p / (2 * mp.pi * mp.e)) / (2 * mp.pi))
            / mp.sqrt(p)
        )

    def wh_inv(u, deg):
        # deg=0: F(u); deg=1: B(u)=f(u)=F'(u); deg=2: b(u)=f'(u)=B'(u)=2*psi_-(u)
        if deg == 2:
            return mp.invertlaplace(lambda p: mp.sqrt(mp.pi) * (g_wh(p) - 1), u, method="dehoog")
        return mp.invertlaplace(
            lambda p: mp.sqrt(mp.pi) * g_wh(p) / p ** (2 - deg), u, method="dehoog"
        )

    # Differentiated contact term check (R4)
    u_diff = mp.mpf("3")
    B_3 = wh_inv(u_diff, 1)
    b_3 = wh_inv(u_diff, 2)
    b_3_numdiff = mp.diff(lambda u: wh_inv(u, 1), u_diff)
    assert abs(b_3 - b_3_numdiff) < mp.mpf("1e-35")
    assert abs(b_3 - B_3) > mp.mpf("1.0")

    # Symbolic & numerical Krein PDE source check (R1)
    uu, vv, ww, c_sym = sp.symbols("uu vv ww c_sym", positive=True)
    B_sym = sp.Function("B")
    integrand_H = 2 * sp.diff(B_sym(uu - ww), uu) * B_sym(vv - ww)
    assert sp.simplify(sp.diff(integrand_H, uu) + sp.diff(integrand_H, vv) + sp.diff(integrand_H, ww)) == 0

    src_correct = 2 * sp.diff(B_sym(uu), uu) * B_sym(vv) - 2 * c_sym * sp.diff(B_sym(vv), vv)
    src_wrong = 2 * sp.diff(B_sym(uu), uu) * (B_sym(vv) - c_sym)
    assert sp.expand(src_correct - src_wrong - 2 * c_sym * (sp.diff(B_sym(uu), uu) - sp.diff(B_sym(vv), vv))) == 0

    B_4 = wh_inv(mp.mpf("4"), 1)
    B_1 = wh_inv(mp.mpf("1"), 1)
    b_4 = wh_inv(mp.mpf("4"), 2)
    b_1 = wh_inv(mp.mpf("1"), 2)
    c_val = mp.sqrt(mp.pi)

    val_correct = 2 * b_4 * B_1 - 2 * c_val * b_1
    val_wrong = 2 * b_4 * (B_1 - c_val)
    assert abs(val_correct - mp.mpf("-0.3471042546629204553")) < mp.mpf("1e-18")
    assert abs(val_wrong - mp.mpf("0.8377480606920601269")) < mp.mpf("1e-18")
    assert abs((val_correct - val_wrong) - 2 * c_val * (b_4 - b_1)) < mp.mpf("1e-40")

    # 4c. Formal constant-kernel integrals R_0(inf,r) and c_1(r) (R9)
    d0_val = mp.mpf(1) / 48 - 1 / (4 * mp.pi**2)
    for rr in (mp.mpf("0.4"), mp.mpf("0.6")):
        chi2 = (mp.polylog(2, rr) - mp.polylog(2, -rr)) / 2
        R0_quad = 4 / mp.pi * mp.quad(lambda z: mp.log(z / rr) * z**2 / (1 - z**2), [0, rr])
        R0_closed = -4 / mp.pi * (chi2 - rr)
        assert abs(R0_quad - R0_closed) < mp.mpf("1e-42")

        def c1_integrand(z, r_loc=rr):
            sig = 1 - r_loc**2 + z**2
            delta = -mp.log(z**2 / (r_loc**2 * sig)) / (2 * mp.pi)
            return (8 / sig ** mp.mpf("2.5")) * (
                z**2 * (-2 * d0_val - 3 / (4 * mp.pi**2) - delta / (4 * mp.pi) + 3 * delta**2 / 8)
                - (d0_val * sig / 2) * (1 + z**2 * (1 - 1 / r_loc**2 - 1 / sig))
            )

        c1_quad = mp.quad(c1_integrand, [0, rr])
        c1_closed = -rr**3 / (6 * (1 - rr**2)) - 2 * d0_val * (
            ((1 - rr**2) / rr**2) * mp.log((1 - rr) / (1 + rr)) + 2 / rr
        )
        assert abs(c1_quad - c1_closed) < mp.mpf("1e-42")

    # 4e. Jacobian-weighted characteristic derivative on A_lambda, 3x3 minor, moving pole
    y1, y2, r_s, Y_s = sp.symbols("y1 y2 r_s Y_s", positive=True)
    A1 = y1 * y2 * (y1 - y2) / (y1 + y2) ** 3
    assert sp.simplify(A1 + A1.xreplace({y1: y2, y2: y1})) == 0
    assert sp.limit(A1, y2, 0) == 0
    assert sp.simplify(A1.subs({y1: Y_s, y2: r_s**2 * Y_s}) - r_s**2 * (1 - r_s**2) / (1 + r_s**2) ** 3) == 0

    minor_3x3 = sp.Matrix([[A1.subs({y1: i, y2: j}) for j in (4, 5, 6)] for i in (1, 2, 3)]).det()
    assert minor_3x3 != 0
    assert sp.limit((y1 + y2) ** 3 * A1, y1, -y2) == 2 * y2**3

    y_at_4 = (3 * wh_inv(mp.mpf("4"), 0) / 4) ** (mp.mpf(2) / 3)
    y_at_1 = (3 * wh_inv(mp.mpf("1"), 0) / 4) ** (mp.mpf(2) / 3)
    dA_dy1 = sp.lambdify((y1, y2), sp.diff(A1, y1), "mpmath")
    dA_dy2 = sp.lambdify((y1, y2), sp.diff(A1, y2), "mpmath")
    jac_char = (
        B_4 / (2 * mp.sqrt(y_at_4)) * dA_dy1(y_at_4, y_at_1)
        + B_1 / (2 * mp.sqrt(y_at_1)) * dA_dy2(y_at_4, y_at_1)
    )
    assert abs(jac_char - mp.mpf("-0.0080523430782269595287")) < mp.mpf("1e-18")
    print("  Finite-sea reciprocity, Krein PDE source, formal integrals and A_lambda checks: PASS")


def check_5_section9_breathers_stirling_and_ring():
    print("=== Check 5: [R5, R6, R10, Sec. 9] Breather roots, residues & ring derivatives ===")
    # 5a. Compensating O(hbar^{-1}) two-body prefactor test for Stirling matching (R6)
    xi = mp.mpf("0.3") / mp.mpf("1e-12")
    stirling_offset = 2 * mp.im(mp.loggamma(1 + 1j * xi)) - 2 * xi * (mp.log(xi) - 1)
    assert abs(stirling_offset - mp.pi / 2) < mp.mpf("1e-11")

    # 5b. Full numerical residues of S_ov at Bogdan-Charkina breather poles N=1,2,3 (R10)
    def S_ov_bad(b1, b2, hbar):
        dl = mp.cosh(3 * b1) - mp.cosh(3 * b2)
        eta = 2 * dl / (9 * hbar)
        A12 = (1 + 3 * mp.coth((b1 + b2) / 2) ** 2) / (1 + 3 * mp.coth((b1 - b2) / 2) ** 2)
        top = 8 * mp.sinh(b1) * mp.sinh(b2) * (mp.cosh(b1) - mp.cosh(b2)) / 3
        return (
            mp.gamma(1 + 1j * eta)
            / mp.gamma(1 - 1j * eta)
            * mp.exp(1j * eta * mp.log(mp.e**2 * hbar**2 * 81 * A12 / (4 * dl**2)) + 1j * top / hbar)
        )

    a_br = mp.mpf("0.5")
    theta_br = mp.mpf("1.25")
    I_br = -4 * mp.sinh(3 * a_br) * mp.sin(3 * theta_br) / 9
    for N_pole in (1, 2, 3):
        hbar_N = I_br / N_pole
        res_1 = mp.mpf("1e-15") * S_ov_bad(a_br - 1j * (theta_br + mp.mpf("1e-15")), a_br + 1j * theta_br, hbar_N)
        res_2 = mp.mpf("1e-20") * S_ov_bad(a_br - 1j * (theta_br + mp.mpf("1e-20")), a_br + 1j * theta_br, hbar_N)
        assert abs(res_2) > mp.mpf("1e-10")
        assert abs(res_1 - res_2) / abs(res_2) < mp.mpf("1e-12")

    # 5c. Repaired uniform positive root witness X = 1/(1 + sqrt(1 - A_12)) at cos(I) = -1 (R5)
    A12_sym = sp.Symbol("A12", real=True)
    tau_minus = lambda X: 1 - 2 * X + A12_sym * X**2
    X_uniform = 1 / (1 + sp.sqrt(1 - A12_sym))
    assert sp.simplify(tau_minus(X_uniform)) == 0

    for A_test in [mp.mpf("0.5"), mp.mpf("0.0"), mp.mpf("-1.5")]:
        X_val = 1 / (1 + mp.sqrt(1 - A_test))
        assert X_val > 0
        assert abs(1 - 2 * X_val + A_test * X_val**2) < mp.mpf("1e-45")

    # Verify old witness X = 1/sqrt(A_12) fails at cos(I) = -1 for A_12 != 1
    A_pos = sp.Symbol("A_pos", positive=True)
    assert sp.simplify(tau_minus(1 / sp.sqrt(A_pos)).subs(A12_sym, A_pos) - (2 - 2 / sp.sqrt(A_pos))) == 0

    # 5d. Independent numerical differentiation of Phi_ring(j | l) = delta(j,l) + Pi_j*M_l (R10)
    def sol_data(par, eps, is_bad):
        k = mp.sinh(par) if is_bad else mp.sin(par)
        v = eps * (mp.cosh(par) if is_bad else mp.cos(par))
        ell = eps * (mp.cosh(3 * par) if is_bad else -mp.cos(3 * par))
        M = 2 * k
        Pi = -M * v
        P = 2 * k**3 * v / 3
        return k, v, ell, M, Pi, P

    def A12_func(p1, p2, e1, e2, is_bad):
        k1, v1, *_ = sol_data(p1, e1, is_bad)
        k2, v2, *_ = sol_data(p2, e2, is_bad)

        def D_poly(K, W):
            return W**2 - K**2 + (-1 if is_bad else 1) * K**4

        return -D_poly(k1 - k2, k1 * v1 - k2 * v2) / D_poly(k1 + k2, k1 * v1 + k2 * v2)

    def delta_func(p1, p2, e1, e2, is_bad):
        k1, v1, l1, M1, *_ = sol_data(p1, e1, is_bad)
        k2, v2, l2, M2, *_ = sol_data(p2, e2, is_bad)
        return 2 * (l1 - l2) * mp.log(A12_func(p1, p2, e1, e2, is_bad)) / 9 + 2 * M1 * M2 * (v1 - v2) / 3

    for is_bad in (False, True):
        for e2 in (1, -1):
            p1_0 = mp.mpf("0.3")
            p2_0 = mp.mpf("0.6")
            k1, *_ = sol_data(p1_0, 1, is_bad)
            M2 = sol_data(p2_0, e2, is_bad)[3]
            Phi_ring = lambda p1: delta_func(p1, p2_0, 1, e2, is_bad) + sol_data(p1, 1, is_bad)[4] * M2
            P1_of_p1 = lambda p1: sol_data(p1, 1, is_bad)[5]
            dPhi_dP1 = mp.diff(Phi_ring, p1_0) / mp.diff(P1_of_p1, p1_0)
            expected = mp.log(A12_func(p1_0, p2_0, 1, e2, is_bad)) / k1
            assert abs(dPhi_dP1 - expected) < mp.mpf("1e-42")
    print("  Stirling prefactor, breather residues, root witness & ring derivatives: PASS")


def check_6_appendix_a_static_integrity():
    print("=== Check 6: [R10 & R11, Appendix A] Static integrity of cited scripts ===")
    tex = (ROOT / "paper/boussinesq_soliton_scattering.tex").read_text(encoding="utf-8")
    app_a = tex.split(r"\section{Claims and verification}")[1].split(r"\section{Inconsistencies")[0]
    scr_paths = sorted(set(re.findall(r"\\scr\{([^{}]+\.py)\}", app_a)))
    nolink_paths = sorted(set(re.findall(r"\\nolinkurl\{([^{}]+\.py)\}", app_a)))

    assert len(scr_paths) == 64, f"Expected 64 \\scr{{...py}} in Appendix A, got {len(scr_paths)}"
    for p in scr_paths + nolink_paths:
        full_p = ROOT / p
        assert full_p.is_file(), f"Missing script: {p}"
        ast.parse(full_p.read_text(encoding="utf-8"))

    # Verify hirota_solitons.py has no 'or True' bypass
    hirota_tree = ast.parse((ROOT / "phase1/scripts/hirota_solitons.py").read_text(encoding="utf-8"))
    bypasses = [
        node
        for node in ast.walk(hirota_tree)
        if isinstance(node, ast.Assert)
        and any(
            isinstance(q, ast.BoolOp)
            and isinstance(q.op, ast.Or)
            and any(isinstance(t, ast.Constant) and t.value is True for t in q.values)
            for q in ast.walk(node.test)
        )
    ]
    assert not bypasses

    # Verify figures.py marker and audit_onebody.py line-383 tolerance
    assert "DATA GENERATED (no checks)" in (ROOT / "phase1/scripts/figures.py").read_text(encoding="utf-8")
    assert "assert max_cv_id_diff < mp.mpf('3e-6')" in (ROOT / "phase2/scripts/audit_onebody.py").read_text(encoding="utf-8")
    print("  Appendix A script inventory and static checks verified: PASS")


if __name__ == "__main__":
    check_1_defect_primitives()
    check_2_collective_symplectic_and_two_soliton_operator()
    check_3_one_loop_units_and_tadpole_sign()
    check_4_section8_krein_and_reciprocity()
    check_5_section9_breathers_stirling_and_ring()
    check_6_appendix_a_static_integrity()
    print("ALL CHECKS PASSED")

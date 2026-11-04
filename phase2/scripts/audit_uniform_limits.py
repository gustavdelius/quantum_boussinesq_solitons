#!/usr/bin/env python3
"""
Independent verification and audit script for phase2/uniform_limits.tex
and phase2/scripts/uniform_limits_*.py.

Implements independent symbolic derivations (SymPy) and high-precision
numerical checks (mpmath / NumPy / SciPy) without importing or reusing
any code from uniform_limits_*.py.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.special import psi as scipy_psi

mp.mp.dps = 60

ROOT = Path(__file__).resolve().parent


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"CHECK FAILED [{name}]: {detail}")
    print(f"[PASS] {name}" + (f" ({detail})" if detail else ""))


def audit_conventions_and_scales():
    """
    Section 1 & item 3 of audit scope:
    - Good vs bad Boussinesq PDE/Lagrangian/one-soliton dictionary.
    - Integrable defect s = sigma_vac (scattering) vs c = eta * sigma_vac (spectrum).
    - Bad-Boussinesq / Toda dictionary (a=1, b=e^{-1/2}, h=3*b*hbar_B).
    - Monotonicity P_0'(kappa) > 0 and order-of-limits scale separation (fixed Y vs fixed P).
    """
    alpha, beta, eps, b, h, kappa, Y = sp.symbols(
        "alpha beta eps b h kappa Y", positive=True
    )

    # Good Boussinesq one-soliton data: k = sin(alpha), v = eps*cos(alpha), M = 2k, P = (2/3)*eps*k^3*cos(alpha)
    P_good = sp.Rational(2, 3) * eps * sp.sin(alpha) ** 3 * sp.cos(alpha)
    P_bad_cont = sp.Rational(2, 3) * sp.sinh(beta) ** 3 * sp.cosh(beta)
    check(
        "good_bad_analytic_continuation",
        sp.simplify(P_good.subs({alpha: sp.I * beta, eps: 1}) / (-sp.I) - P_bad_cont) == 0,
        "P_good(i*beta) continues to -i * (2/3) sinh^3(beta) cosh(beta)",
    )

    # Toda / bad-Boussinesq dictionary (Eq. 1 of uniform_limits.tex)
    P0 = 4 * b * (kappa * sp.cosh(kappa) - sp.sinh(kappa))
    E0 = 2 * b**2 * (sp.sinh(kappa) * sp.cosh(kappa) - kappa)
    dP0 = sp.simplify(sp.diff(P0, kappa))
    dE0 = sp.simplify(sp.diff(E0, kappa))
    v_toda = sp.simplify(dE0 / dP0)

    check(
        "toda_dP0_positive_monotone",
        sp.simplify(dP0 - 4 * b * kappa * sp.sinh(kappa)) == 0,
        "P_0'(kappa) = 4*b*kappa*sinh(kappa) > 0 for kappa > 0",
    )
    check(
        "toda_hamiltonian_velocity",
        sp.simplify(v_toda - b * sp.sinh(kappa) / kappa) == 0,
        "dE_0/dP_0 = b*sinh(kappa)/kappa",
    )

    # Small-kappa series for P0, E0, and E0 - b*P0
    P0_ser = sp.series(P0, kappa, 0, 8).removeO()
    check(
        "toda_P0_small_kappa_series",
        sp.simplify(
            P0_ser
            - (
                sp.Rational(4, 3) * b * kappa**3
                + sp.Rational(2, 15) * b * kappa**5
                + sp.Rational(1, 210) * b * kappa**7
            )
        )
        == 0,
    )
    check(
        "toda_E0_minus_bP0_series",
        sp.simplify(
            sp.series(E0 - b * P0, kappa, 0, 8).removeO()
            - (
                sp.Rational(2, 15) * b**2 * kappa**5
                + sp.Rational(13, 630) * b**2 * kappa**7
            )
        )
        == 0,
    )

    # Leading kappa = sqrt(3)*beta bad-Boussinesq match
    P_B_lead = sp.series(P0.subs(kappa, sp.sqrt(3) * beta) / (6 * sp.sqrt(3) * b), beta, 0, 5).removeO()
    P_bad_lead = sp.series(P_bad_cont, beta, 0, 5).removeO()
    check(
        "bad_boussinesq_leading_cubic_momentum_match",
        sp.simplify(P_B_lead - P_bad_lead) == 0,
        "P_B and P_bad agree to O(beta^3): (2/3)*beta^3",
    )

    # Scale separation: at fixed Y = b*kappa^2/h, P0 = (4/3)*b^{-1/2}*(h*Y)^{3/2} + (2/15)*b^{-3/2}*(h*Y)^{5/2} + O(h^{7/2})
    P0_fixed_Y = sp.series(P0.subs(kappa, sp.sqrt(h * Y / b)), h, 0, 3).removeO()
    check(
        "fixed_Y_shrinking_momentum",
        sp.simplify(
            P0_fixed_Y
            - (
                sp.Rational(4, 3) * b ** (-sp.Rational(1, 2)) * (h * Y) ** sp.Rational(3, 2)
                + sp.Rational(2, 15) * b ** (-sp.Rational(3, 2)) * (h * Y) ** sp.Rational(5, 2)
            )
        )
        == 0,
        "Fixed Y implies P_0 = (4/3)*b^{-1/2}*(h*Y)^{3/2} + (2/15)*b^{-3/2}*(h*Y)^{5/2} + O(h^{7/2}) -> 0",
    )

    # Check that uniform_limits.tex does not conflate defect s = sigma_vac with eta*sigma_vac
    tex_text = (ROOT.parent / "uniform_limits.tex").read_text(encoding="utf-8")
    check(
        "defect_convention_not_conflated",
        "sigma_{vac}" not in tex_text and "\\sigma_{\\mathrm{vac}}" not in tex_text,
        "uniform_limits.tex works purely in the bulk Toda/bad-Boussinesq sector without defect s conflation",
    )

    return {
        "P0_prime": "4*b*kappa*sinh(kappa)",
        "fixed_Y_P0_leading": "(4/3)*b^(-1/2)*(h*Y)^(3/2)",
    }


def audit_kernel_boundary_layer():
    """
    Section 2 (Eqs. 2-8):
    - Kernel decomposition K_h(d) = 2*log|d| + R(d/h), R(x) = 2*Re psi(1+i*x) - 2*log|x|.
    - Exact antiderivative G(L) = int_0^L R(x) dx = 2*Im log Gamma(1+i*L) - 2*L*log(L) + 2*L.
    - Stirling tail of G(L) and R(x), total mass pi, near-zero expansion.
    - Direct evaluation of cusp integral int_{-1}^1 R(z/h)|z| dz = (h^2/3)*log(1/h) + 2*C_cusp*h^2 + O(h^4)
      (extending uniform_limits_analytic.py, which only checked the moment increment on [1000, 2000]).
    - Direct numerical test of the operator estimate on an interior Lipschitz function
      (extending uniform_limits_analytic.py, which did not test A_h f on a Lipschitz function).
    """
    R_mp = lambda x: 2 * mp.re(mp.psi(0, 1 + 1j * x)) - 2 * mp.log(abs(x))
    G_mp = lambda L: 2 * mp.im(mp.loggamma(1 + 1j * L)) - 2 * L * mp.log(L) + 2 * L

    for L_val in [mp.mpf("0.7"), mp.mpf("3.5"), mp.mpf("25.0")]:
        dG = mp.diff(G_mp, L_val)
        err_deriv = abs(dG - R_mp(L_val))
        check(
            f"kernel_primitive_derivative_L={L_val}",
            err_deriv < mp.mpf("1e-45"),
            f"err={float(err_deriv):.2e}",
        )

    quad_G = mp.quad(R_mp, [0, mp.mpf("0.5"), mp.mpf("3.5")])
    err_quad = abs(quad_G - G_mp(mp.mpf("3.5")))
    check("kernel_primitive_quadrature", err_quad < mp.mpf("1e-40"), f"err={float(err_quad):.2e}")

    L_big = mp.mpf("100")
    G_asymp = (
        mp.pi / 2
        - 1 / (6 * L_big)
        - 1 / (180 * L_big**3)
        - 1 / (630 * L_big**5)
        - 1 / (840 * L_big**7)
        - 1 / (594 * L_big**9)
    )
    err_G_asymp = abs(G_mp(L_big) - G_asymp)
    check("kernel_G_asymptotic_L100", err_G_asymp < mp.mpf("1e-20"), f"err={float(err_G_asymp):.2e}")

    R_asymp = (
        1 / (6 * L_big**2)
        + 1 / (60 * L_big**4)
        + 1 / (126 * L_big**6)
        + 1 / (120 * L_big**8)
        + 1 / (66 * L_big**10)
    )
    err_R_asymp = abs(R_mp(L_big) - R_asymp)
    check("kernel_R_asymptotic_L100", err_R_asymp < mp.mpf("1e-22"), f"err={float(err_R_asymp):.2e}")

    x_small = mp.mpf("1e-5")
    reg_exact = 2 * mp.re(mp.psi(0, 1 + 1j * x_small))
    reg_series = -2 * mp.euler + 2 * mp.zeta(3) * x_small**2 - 2 * mp.zeta(5) * x_small**4
    err_small = abs(reg_exact - reg_series)
    check("kernel_R_small_x_series", err_small < mp.mpf("1e-28"), f"err={float(err_small):.2e}")

    # Direct evaluation of the cusp integral I_cusp(h) = int_{-1}^1 R(z/h)|z| dz
    # Avoid mp.inf in mp.quad(u*(R(u)-1/(6u^2))) because 2*Re psi(1+i*u) - 2*log(u)
    # loses precision when u > 10^(dps/2); instead split at u0 = 10 and use exact Bernoulli tail.
    u0_cut = mp.mpf("10")
    tail_u0 = sum(
        abs(mp.bernoulli(2 * m)) / (m * (2 * m - 2) * u0_cut ** (2 * m - 2))
        for m in range(2, 14)
    )
    C_cusp = (
        mp.quad(lambda u: u * R_mp(u), [0, mp.mpf("0.2"), 1])
        + mp.quad(lambda u: u * (R_mp(u) - 1 / (6 * u**2)), [1, 3, u0_cut])
        + tail_u0
    )
    for h_val in [mp.mpf("2e-2"), mp.mpf("1e-2")]:
        Lam = 1 / h_val
        with mp.workdps(30):
            I_cusp_exact = 2 * h_val**2 * mp.quad(lambda u: u * R_mp(u), [0, 1, 10, 30, Lam])
        I_cusp_lead = (h_val**2 / 3) * mp.log(1 / h_val)
        I_cusp_two = I_cusp_lead + 2 * C_cusp * h_val**2
        I_cusp_three = I_cusp_two - (h_val**4) / 60
        check(
            f"cusp_integral_asymptotic_h={float(h_val):.0e}",
            abs(I_cusp_exact - I_cusp_three) < 5 * h_val**6,
            f"exact={float(I_cusp_exact):.12e}, two_term_err={float(abs(I_cusp_exact - I_cusp_two)):.3e}, three_term_err={float(abs(I_cusp_exact - I_cusp_three)):.3e}",
        )

    # Direct numerical verification of the Lipschitz operator remainder at an interior point
    x0 = mp.mpf("0.2")
    f_lip = lambda z: abs(z - x0) + mp.mpf("0.5") * mp.cos(z)
    scaled_rems = []
    for h_val in [mp.mpf("0.04"), mp.mpf("0.02"), mp.mpf("0.01")]:
        L_left = (1 + x0) / h_val
        L_right = (1 - x0) / h_val
        with mp.workdps(20):
            int_diff = h_val * mp.quad(
                lambda u: R_mp(u)
                * (f_lip(x0 - h_val * u) + f_lip(x0 + h_val * u) - 2 * f_lip(x0)),
                [0, 1, 10, min(L_left, L_right)],
            ) + h_val * mp.quad(
                lambda u: R_mp(u) * (f_lip(x0 - h_val * u) - f_lip(x0)),
                [min(L_left, L_right), max(L_left, L_right)],
            )
            mass_defect = f_lip(x0) * (mp.pi * h_val - h_val * (G_mp(L_left) + G_mp(L_right)))
            op_rem = -int_diff + mass_defect
            scale = h_val**2 * mp.log(1 / h_val)
            scaled_rems.append(float(abs(op_rem / scale)))

    check(
        "lipschitz_operator_remainder_scaling",
        all(0.15 < r < 0.50 for r in scaled_rems),
        f"|rem|/(h^2 log(1/h)) = {[round(r, 4) for r in scaled_rems]}",
    )

    return {
        "C_cusp": float(C_cusp),
        "lipschitz_scaled_remainders": scaled_rems,
    }


def audit_wh_reversion_and_matched_proof():
    """
    Sections 3 & 4 (Eqs. 9-21):
    - Wiener-Hopf factor g(s), reflection identity, Laplace transforms.
    - Independent derivation of y(u) and u(y) through O(y^{-3}), verifying A, c, d, e,
      absence of log^k(y)/y^m terms in u(y) for m=1,2,3, and presence of (log u)^2/u^2 in y(u).
    - One-body C_E(y) and C_v(y) large-y AND small-y expansions (verifying C_E(0^+) = C_v(0^+) = 3/(4*pi)
      vs C_E(inf) = C_v(inf) = 1/(2*pi), filling the gap where uniform_limits_analytic.py only checked C_v(0^+)).
    - Symbolic derivation of I_0(t) and I_1(t) directly from 8*Y^{3/2}*t^2*f(u(Y)-u(r^2*Y)+u(Y*t^2))
      (filling the gap where uniform_limits_matched.py typed I_0, I_1 by hand).
    - Exact closed-form symbolic primitive for int_0^r I_0(t) dt and 50-digit cross-check of c_0(r), c_1(r).
    - Classical (log Y)^2 / Y Hessian term and exact cancellation with outer integral.
    - Endpoint moment mu and K = -2*mu = 0.
    """
    g_mp = lambda s: (
        s ** (-mp.mpf("0.5"))
        * mp.gamma(1 + s / (2 * mp.pi))
        * mp.exp(-(s / (2 * mp.pi)) * mp.log(s / (2 * mp.pi * mp.e)))
    )
    for w_val in [mp.mpf("0.3"), mp.mpf("1.7"), mp.mpf("6.5")]:
        lhs = 2 * mp.pi * g_mp(1j * w_val) * g_mp(-1j * w_val)
        rhs = 2 * mp.pi / (1 - mp.exp(-w_val))
        check(
            f"wh_reflection_identity_w={w_val}",
            abs(lhs - rhs) < mp.mpf("1e-45"),
            f"err={float(abs(lhs - rhs)):.2e}",
        )

    s, u, y, nu = sp.symbols("s u y nu", positive=True)
    L = sp.Symbol("L", real=True)
    log_s = sp.Symbol("log_s", real=True)
    z = s / (2 * sp.pi)
    log_z = log_s - sp.log(2) - sp.log(sp.pi)
    log_g_small = (
        -z * (log_z - 1)
        - sp.EulerGamma * z
        + (sp.pi**2 / 6) * z**2 / 2
        - sp.zeta(3) * z**3 / 3
        + (sp.pi**4 / 90) * z**4 / 4
    )
    g_ratio_small = sp.series(sp.exp(log_g_small), s, 0, 4).removeO()

    def polygamma_half_int(r_ord, nu_val):
        # Exact polygamma^{(r)}(nu_val) for half-integer nu_val using recurrence from 1/2
        base = {
            0: -sp.EulerGamma - 2 * sp.log(2),
            1: sp.pi**2 / 2,
            2: -14 * sp.zeta(3),
            3: sp.pi**4,
        }[r_ord]
        cur = sp.Rational(1, 2)
        val = base
        while cur < nu_val:
            val += (-1) ** r_ord * sp.factorial(r_ord) / (cur ** (r_ord + 1))
            cur += 1
        while cur > nu_val:
            cur -= 1
            val -= (-1) ** r_ord * sp.factorial(r_ord) / (cur ** (r_ord + 1))
        return sp.expand(val)

    eps_nu = sp.Symbol("eps_nu")

    def inv_laplace_term(k_pow, m_log, base_nu=sp.Rational(5, 2)):
        nu_val = base_nu - k_pow
        if m_log == 0:
            return sp.expand(sp.sqrt(sp.pi) / sp.gamma(nu_val))
        exponent = (
            (L - polygamma_half_int(0, nu_val)) * eps_nu
            - polygamma_half_int(1, nu_val) * eps_nu**2 / 2
            - polygamma_half_int(2, nu_val) * eps_nu**3 / 6
            - polygamma_half_int(3, nu_val) * eps_nu**4 / 24
        )
        ser = sp.series(sp.exp(exponent), eps_nu, 0, m_log + 1).removeO()
        coeff = ser.coeff(eps_nu, m_log) if m_log > 0 else ser.subs(eps_nu, 0)
        return sp.expand(
            (sp.sqrt(sp.pi) / sp.gamma(nu_val)) * ((-1) ** m_log) * sp.factorial(m_log) * coeff
        )

    poly_g = sp.expand(g_ratio_small)
    F_over_u32 = sp.Integer(0)
    J_over_u52 = sp.Integer(0)
    for k_pow in range(4):
        coeff_s = poly_g.coeff(s, k_pow) if k_pow > 0 else poly_g.subs(s, 0)
        poly_logs = sp.Poly(coeff_s, log_s)
        for (m_log,), c_kl in poly_logs.terms():
            F_over_u32 += c_kl * inv_laplace_term(k_pow, m_log, sp.Rational(5, 2)) * u ** (-k_pow)
            J_over_u52 += c_kl * inv_laplace_term(k_pow, m_log, sp.Rational(7, 2)) * u ** (-k_pow)

    eps_u = sp.Symbol("eps_u", positive=True)
    ratio_F = sp.Rational(3, 4) * F_over_u32.subs(u, 1 / eps_u)
    y_over_u_ser = sp.expand(sp.series(ratio_F ** sp.Rational(2, 3), eps_u, 0, 4).removeO())
    y_of_u = sp.expand(u * y_over_u_ser.subs(eps_u, 1 / u))

    coeff_logu2_over_u2 = sp.expand(y_of_u.coeff(u, -2).coeff(L, 2))
    check(
        "forward_y_of_u_has_log2_over_u2",
        sp.expand(coeff_logu2_over_u2 + 1 / (16 * sp.pi**3)) == 0,
        f"coeff of (log u)^2/u^2 in y(u) is {coeff_logu2_over_u2}",
    )

    Ly = sp.Symbol("Ly", real=True)
    A_ex = 1 / (2 * sp.pi)
    c_ex = (1 - 3 * sp.log(2) - sp.log(sp.pi)) / (2 * sp.pi)
    d_ex = sp.Rational(1, 48) - 1 / (4 * sp.pi**2)
    e_ex = (4 + 6 * sp.zeta(3) - sp.pi**2) / (96 * sp.pi**3)

    delta_u = -A_ex * Ly + c_ex + d_ex / y + e_ex / y**2
    L_sub = Ly + sp.series(sp.log(1 + delta_u / y), y, sp.oo, 4).removeO()
    y_check = y_of_u.subs(L, L_sub).subs(u, y + delta_u)
    res_reversion = sp.expand(sp.series(y_check - y, y, sp.oo, 3).removeO())
    check(
        "reversion_u_of_y_through_y_minus_2",
        res_reversion == 0,
        "y(u(y)) - y = O(y^{-3}) with pure powers d/y + e/y^2 and no log terms at y^{-1}, y^{-2}",
    )

    # Verify J(u(y)) and C_E(y), C_v(y) large-y series
    # From J_over_u52, J(u) = u^{5/2} * J_over_u52 has only half-integer powers of y when u = u(y),
    # which proves K = 0 in J(u(y)) = (8/15)*y^{5/2} - (4*d/3)*y^{1/2} + K + (8*e/3)*y^{-1/2} + ...
    J_of_y_ratio = (1 + delta_u / y) ** sp.Rational(5, 2) * J_over_u52.subs(L, L_sub).subs(u, y + delta_u)
    J_of_y_ser = sp.expand(sp.series(J_of_y_ratio, y, sp.oo, 4).removeO())
    check(
        "J_of_u_y_expansion_and_K_zero",
        sp.expand(
            J_of_y_ser
            - (
                sp.Rational(8, 15)
                - (8 * A_ex / 9) / y
                - (8 * d_ex / 3) / y**2
                + (16 * e_ex / 3) / y**3
            )
        )
        == 0,
        "J(u(y)) = (8/15)*y^{5/2} - (8*A/9)*y^{3/2} - (8*d/3)*y^{1/2} + 0 + (16*e/3)*y^{-1/2} + O(y^{-3/2}), so K = 0",
    )

    CE_large = sp.expand(
        sp.series(
            sp.Rational(3, 4) / sp.pi
            + sp.Rational(3, 4) * (y * J_of_y_ser) / sp.Rational(4, 3)
            - sp.Rational(3, 10) * y,
            y,
            sp.oo,
            3,
        ).removeO()
    )
    u_prime = 1 - A_ex / y - d_ex / y**2 - 2 * e_ex / y**3
    Cv_large = sp.expand(sp.series(sp.Rational(3, 4) / sp.pi + (y / 2) * (u_prime - 1), y, sp.oo, 3).removeO())
    check(
        "CE_and_Cv_large_y_series",
        sp.expand(CE_large - (1 / (2 * sp.pi) - 3 * d_ex / (2 * y) + 3 * e_ex / y**2)) == 0
        and sp.expand(Cv_large - (1 / (2 * sp.pi) - d_ex / (2 * y) - e_ex / y**2)) == 0,
    )

    # Small-y (phonon-first, y -> 0^+) series of BOTH C_E(y) and C_v(y)
    u_s = sp.Symbol("u_s", positive=True)
    F_small_u = sp.sqrt(sp.pi) * u_s + (sp.pi ** sp.Rational(3, 2) / 12) * u_s**2
    J_small_u = (sp.sqrt(sp.pi) / 2) * u_s**2 + (sp.pi ** sp.Rational(3, 2) / 36) * u_s**3
    u_small_y = (sp.Rational(4, 3) / sp.sqrt(sp.pi)) * y ** sp.Rational(3, 2) - (
        sp.Rational(4, 27) / sp.sqrt(sp.pi)
    ) * y**3
    CE_small_y = sp.series(
        sp.Rational(3, 4) / sp.pi
        + sp.Rational(3, 4) * (J_small_u / F_small_u).subs(u_s, u_small_y)
        - sp.Rational(3, 10) * y,
        y,
        0,
        3,
    ).removeO()
    Cv_small_y = sp.series(
        sp.Rational(3, 4) / sp.pi + (y / 2) * (sp.diff(u_small_y, y) - 1),
        y,
        0,
        3,
    ).removeO()
    check(
        "CE_and_Cv_small_y_limits",
        sp.limit(CE_small_y, y, 0) == 3 / (4 * sp.pi)
        and sp.limit(Cv_small_y, y, 0) == 3 / (4 * sp.pi)
        and sp.simplify((sp.limit(CE_small_y, y, 0) - 1 / (2 * sp.pi)) - 1 / (4 * sp.pi)) == 0,
        "Both C_E(0^+) and C_v(0^+) equal 3/(4*pi), differing from 1/(2*pi) by 1/(4*pi)",
    )

    # Symbolic derivation of I_0(t) and I_1(t) from 8*Y^{3/2}*t^2 * f(u(x_W))
    A_s, c_s, d_s, e_s, r_s, t_s, Y_s, sig, Delta0 = sp.symbols(
        "A_s c_s d_s e_s r_s t_s Y_s sig Delta0", positive=True
    )
    x_s = sp.Symbol("x_s", positive=True)
    ux = x_s - A_s * sp.log(x_s) + c_s + d_s / x_s + e_s / x_s**2
    dux = sp.diff(ux, x_s)
    d2ux = sp.diff(ux, x_s, 2)
    f_of_ux = sp.series(
        sp.sqrt(x_s) * (1 / (sp.sqrt(x_s) * dux**2) - 2 * sp.sqrt(x_s) * d2ux / dux**3),
        x_s,
        sp.oo,
        3,
    ).removeO()
    check(
        "f_of_u_x_asymptotic",
        sp.simplify(f_of_ux - (1 - (2 * d_s + 3 * A_s**2) / x_s**2)) == 0,
        "sqrt(x)*f(u(x)) = 1 - (2*d + 3*A^2)/x^2 + O(x^{-3})",
    )

    # Solve u(x_W) = u(Y) - u(r^2*Y) + u(Y*t^2) for x_W = Y*sig + x0 + x1/Y + O(Y^{-2}):
    # Here u(Y) - u(r^2*Y) + u(Y*t^2) = Y*sig - A*log(Y*sig) + Delta0 + c + (d/Y)*(1 - 1/r^2 + 1/t^2)
    # where Delta0 = -A*log(t^2/(r^2*sig)), so c cancels against +c in u(x_W):
    x1_W = A_s * Delta0 / sig + d_s * (1 - 1 / r_s**2 + 1 / t_s**2 - 1 / sig)
    x_W_exp = Y_s * sig + Delta0 + x1_W / Y_s
    u_W_minus_W = (
        x_W_exp
        - A_s
        * (
            sp.log(Y_s * sig)
            + sp.series(sp.log(1 + (Delta0 + x1_W / Y_s) / (Y_s * sig)), Y_s, sp.oo, 2).removeO()
        )
        + c_s
        + d_s / (Y_s * sig)
        - (
            Y_s * sig
            - A_s * sp.log(Y_s * sig)
            + Delta0
            + c_s
            + d_s * (1 - 1 / r_s**2 + 1 / t_s**2) / Y_s
        )
    )
    check(
        "dressed_coordinate_x_W_matching",
        sp.simplify(sp.series(u_W_minus_W, Y_s, sp.oo, 2).removeO()) == 0,
        "u(x_W) matches u(Y)-u(r^2*Y)+u(Y*t^2) through O(Y^{-1})",
    )

    # Expand 8*Y^{3/2}*t^2 * [x_W^{-1/2}*(1 - (2*d + 3*A^2)/x_W^2) - (Y*sig)^{-1/2}]
    outer_integrand = (
        8
        * Y_s ** sp.Rational(3, 2)
        * t_s**2
        * (
            (Y_s * sig) ** (-sp.Rational(1, 2))
            * (1 + (Delta0 + x1_W / Y_s) / (Y_s * sig)) ** (-sp.Rational(1, 2))
            * (1 - (2 * d_s + 3 * A_s**2) / (Y_s * sig) ** 2)
            - (Y_s * sig) ** (-sp.Rational(1, 2))
        )
    )
    outer_ser = sp.series(outer_integrand, Y_s, sp.oo, 2).removeO()
    I0_derived = sp.simplify(outer_ser.subs(Y_s, sp.oo))
    I1_derived = sp.simplify((outer_ser - I0_derived) * Y_s)

    I0_pkg = -4 * t_s**2 * Delta0 / sig ** sp.Rational(3, 2)
    I1_pkg = (8 / sig ** sp.Rational(5, 2)) * (
        t_s**2 * (-2 * d_s - 3 * A_s**2 - A_s * Delta0 / 2 + 3 * Delta0**2 / 8)
        - (d_s * sig / 2) * (1 + t_s**2 * (1 - 1 / r_s**2 - 1 / sig))
    )
    check(
        "symbolic_derivation_of_I0_and_I1",
        sp.simplify(I0_derived - I0_pkg) == 0 and sp.simplify(I1_derived - I1_pkg) == 0,
        "Derived I_0(t) and I_1(t) symbolically from 8*Y^{3/2}*t^2*f(u(x_W))",
    )

    # High-precision verification of c_0(r) and c_1(r) via alternative change of variables u = t/r in [0,1]
    A_mp = 1 / (2 * mp.pi)
    d_mp = mp.mpf(1) / 48 - 1 / (4 * mp.pi**2)
    for r_val in [mp.mpf("0.4"), mp.mpf("0.5"), mp.mpf("0.6")]:
        c0_closed = -(2 / mp.pi) * (mp.polylog(2, r_val) - mp.polylog(2, -r_val) - 2 * r_val)
        c1_closed = -r_val**3 / (6 * (1 - r_val**2)) - 2 * d_mp * (
            ((1 - r_val**2) / r_val**2) * mp.log((1 - r_val) / (1 + r_val)) + 2 / r_val
        )

        def i0_mp(t):
            s_v = 1 - r_val**2 + t**2
            D0 = -A_mp * mp.log(t**2 / (r_val**2 * s_v))
            return -4 * t**2 * D0 / s_v ** mp.mpf("1.5")

        def i1_mp(t):
            s_v = 1 - r_val**2 + t**2
            D0 = -A_mp * mp.log(t**2 / (r_val**2 * s_v))
            return (8 / s_v ** mp.mpf("2.5")) * (
                t**2 * (-2 * d_mp - 3 * A_mp**2 - A_mp * D0 / 2 + 3 * D0**2 / 8)
                - (d_mp * s_v / 2) * (1 + t**2 * (1 - 1 / r_val**2 - 1 / s_v))
            )

        with mp.workdps(42):
            q0 = mp.quad(i0_mp, [0, r_val / 3, r_val])
            q1 = mp.quad(i1_mp, [0, r_val / 3, r_val])
        check(
            f"matched_c0_c1_quadrature_r={r_val}",
            abs(q0 - c0_closed) < mp.mpf("1e-38") and abs(q1 - c1_closed) < mp.mpf("1e-38"),
            f"err0={float(abs(q0 - c0_closed)):.2e}, err1={float(abs(q1 - c1_closed)):.2e}",
        )

    # Verify classical (log Y)^2 / Y Hessian term (Eq. 20)
    y1, y2, d1, d2 = sp.symbols("y1 y2 d1 d2", positive=True)
    Phi2 = 2 * (y1 - y2) * sp.log((sp.sqrt(y1) - sp.sqrt(y2)) / (sp.sqrt(y1) + sp.sqrt(y2))) + 4 * sp.sqrt(y1 * y2)
    hess_term = sp.simplify(
        (
            d1**2 * sp.diff(Phi2, y1, 2)
            + 2 * d1 * d2 * sp.diff(sp.diff(Phi2, y1), y2)
            + d2**2 * sp.diff(Phi2, y2, 2)
        ).subs({y1: Y_s, y2: r_s**2 * Y_s})
        / 2
    )
    check(
        "classical_quadratic_taylor_formula",
        sp.simplify(hess_term - (d2 - r_s**2 * d1) ** 2 / (r_s * (1 - r_s**2) * Y_s)) == 0,
    )
    hess_logY2 = sp.simplify(
        sp.expand(hess_term.subs({d1: -A_ex * Ly + c_ex, d2: -A_ex * (Ly + 2 * sp.log(r_s)) + c_ex})).coeff(Ly, 2)
        * Y_s
    )
    check(
        "classical_logY2_over_Y_nonzero",
        sp.simplify(hess_logY2 - (1 - r_s**2) / (4 * sp.pi**2 * r_s)) == 0,
        f"coeff of (log Y)^2/Y in Phi_cl(u(Y),u(r^2Y)) is (1-r^2)/(4*pi^2*r)",
    )

    return {
        "K_moment": 0,
        "hessian_logY2_coeff": "(1 - r^2)/(4*pi^2*r)",
    }


def audit_independent_wh_convolution():
    """
    Section 5 (Half-line WH convolution):
    Compute u(Y) and Q(u(Y), u(r^2*Y)) using an INDEPENDENT numerical method:
    - Instead of integrating over z in [0, u(r^2*Y)] with B(z) ~ z^{-1/2} (which requires two
      inverse Laplace transforms per node and has a square-root singularity at z=0),
      use the exact measure change z = u(w^2) where B(z) dz = 2*w^2 dw:
      Q(u(Y), u(r^2*Y)) = 4 * int_0^{r*sqrt(Y)} w^2 * f(u(Y) - u(r^2*Y) + u(w^2)) dw.
    - Notice that u(Y) - u(r^2*Y) + u(w^2) >= u(Y) - u(r^2*Y) >= (1-r^2)*Y - O(log Y) >> 1
      for all w in [0, r*sqrt(Y)], so f(W) is evaluated ONLY at large argument W >= (1-r^2)*Y!
    - Cross-check against uniform_limits_wh.json and perform independent model refits.
    """
    wh_cache = json.loads((ROOT / "uniform_limits_wh.json").read_text(encoding="utf-8"))

    # Independent de Hoog-Knight-Stokes Fourier-Pade Laplace inversion in Re(s) > 0
    # (note: a Talbot contour wrapping into Re(s) < -2*pi is invalid for g(s) because
    # Gamma(1 + s/(2*pi)) has poles at s = -2*pi*n on the negative real axis).
    def g_scalar(s):
        z = s / (2 * mp.pi)
        return (s ** (-mp.mpf("0.5"))) * mp.gamma(1 + z) * mp.exp(-z * (mp.log(z) - 1))

    def inv_laplace_dehoog(hat_fn, u_val):
        with mp.workdps(25):
            return mp.invertlaplace(hat_fn, mp.mpf(u_val), method="dehoog")

    F_hat = lambda s: mp.sqrt(mp.pi) * g_scalar(s) / s**2
    B_hat = lambda s: mp.sqrt(mp.pi) * g_scalar(s) / s
    f_hat = lambda s: mp.sqrt(mp.pi) * (g_scalar(s) - 1)

    A_mp = 1 / (2 * mp.pi)
    c_mp = (1 - mp.log(8 * mp.pi)) / (2 * mp.pi)
    d_mp = mp.mpf(1) / 48 - 1 / (4 * mp.pi**2)
    e_mp = (4 + 6 * mp.zeta(3) - mp.pi**2) / (96 * mp.pi**3)

    def u_independent(y_val):
        with mp.workdps(25):
            y_mp = mp.mpf(y_val)
            if y_mp == 0:
                return mp.mpf(0)
            target = (mp.mpf(4) / 3) * (y_mp ** mp.mpf("1.5"))
            if y_mp >= 1:
                u0 = y_mp - A_mp * mp.log(y_mp) + c_mp + d_mp / y_mp + e_mp / y_mp**2
            else:
                u0 = (4 / (3 * mp.sqrt(mp.pi))) * (y_mp ** mp.mpf("1.5"))
            for _ in range(2):
                Fv = inv_laplace_dehoog(F_hat, u0)
                Bv = inv_laplace_dehoog(B_hat, u0)
                du = (Fv - target) / Bv
                u0 -= du
                if abs(du) < mp.mpf("1e-18"):
                    break
            return u0

    # Cross-check u(Y) at Y = 16, 64, 256 against uniform_limits_wh.json
    u_saved = {int(r["Y"]): mp.mpf(str(r["u"])) for r in wh_cache["rows"] if float(r["r"]) == 0.4}
    for Y_test in [16, 64, 256]:
        u_ind = u_independent(Y_test)
        u_diff = abs(u_ind - u_saved[Y_test])
        check(
            f"independent_dehoog_u_Y={Y_test}",
            u_diff < mp.mpf("1e-12"),
            f"diff={float(u_diff):.2e}",
        )

    # Cross-check Q(u(Y), u(r^2*Y)) via the exact dressed-measure integral over w in [0, r*sqrt(Y)]:
    # Since F(u(w^2)) = (4/3)*w^3, B(z) dz = dF = 4*w^2 dw, so
    # Q = 2 * int_0^v f(u-v+z) B(z) dz = 8 * int_0^{r*sqrt(Y)} w^2 * f(u(Y) - u(r^2*Y) + u(w^2)) dw.
    # Discretize with 12-point Gauss-Legendre on [0, r*sqrt(Y)] at Y=16, 64 and r=0.5
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(12)
    saved_pts = {(int(p["Y"]), float(p["r"])): p for p in wh_cache["rows"]}
    for Y_test in [16, 64]:
        r_test = 0.5
        W0 = u_independent(Y_test) - u_independent(r_test**2 * Y_test)
        b_int = r_test * math.sqrt(Y_test)
        q_sum = mp.mpf(0)
        for xi, wi in zip(gl_nodes, gl_weights):
            w_val = mp.mpf("0.5") * b_int * (mp.mpf(str(xi)) + 1)
            weight = mp.mpf("0.5") * b_int * mp.mpf(str(wi))
            u_w2 = u_independent(w_val**2)
            f_val = inv_laplace_dehoog(f_hat, W0 + u_w2)
            q_sum += 8 * (w_val**2) * f_val * weight
        phi_cl = 2 * Y_test * (1 - r_test**2) * mp.log((1 - r_test) / (1 + r_test)) + 4 * r_test * Y_test
        R_ind = q_sum - phi_cl
        R_pkg = mp.mpf(str(saved_pts[(Y_test, r_test)]["residual"]))
        diff_R = abs(R_ind - R_pkg)
        check(
            f"independent_dressed_measure_Q_Y={Y_test}_r=0.5",
            diff_R < mp.mpf("5e-8"),
            f"R_ind={float(R_ind):.10f}, R_pkg={float(R_pkg):.10f}, diff={float(diff_R):.2e}",
        )

    # Independent refit of WH residuals across alternative windows and bases (including Y^{-3/2} log Y)
    c1_formal = lambda r: float(
        -r**3 / (6 * (1 - r**2))
        - 2 * d_mp * (((1 - r**2) / r**2) * mp.log((1 - r) / (1 + r)) + 2 / r)
    )
    wh_refit_summary = {}
    for r_val in [0.4, 0.5, 0.6]:
        pts = sorted([p for p in wh_cache["rows"] if abs(float(p["r"]) - r_val) < 1e-9], key=lambda x: int(x["Y"]))
        Ys = np.array([float(p["Y"]) for p in pts], dtype=float)
        Rs = np.array([float(p["residual"]) for p in pts], dtype=float)
        c1_ests = []
        # Test bases: (1, Y^-1, Y^-2), (1, Y^-1, Y^-3/2), (1, Y^-1, Y^-3/2, Y^-2),
        # plus an additional audit basis (1, Y^-1, Y^-3/2*log(Y), Y^-2) on windows Y>=16, Y>=32, Y>=64
        bases = [
            lambda Y: np.column_stack([np.ones_like(Y), Y**-1, Y**-2]),
            lambda Y: np.column_stack([np.ones_like(Y), Y**-1, Y**-1.5]),
            lambda Y: np.column_stack([np.ones_like(Y), Y**-1, Y**-1.5, Y**-2]),
            lambda Y: np.column_stack([np.ones_like(Y), Y**-1, (Y**-1.5) * np.log(Y), Y**-2]),
        ]
        for start_idx in [0, 1, 2]:
            Yw = Ys[start_idx:]
            Rw = Rs[start_idx:]
            for b_fn in bases:
                X = b_fn(Yw)
                if X.shape[0] >= X.shape[1]:
                    coef, *_ = np.linalg.lstsq(X, Rw, rcond=None)
                    c1_ests.append(float(coef[1]))
        cf = c1_formal(r_val)
        check(
            f"wh_refit_c1_negative_and_brackets_formal_r={r_val}",
            max(c1_ests) < 0 and min(c1_ests) <= cf <= max(c1_ests),
            f"c1_formal={cf:.6f} in [{min(c1_ests):.6f}, {max(c1_ests):.6f}]",
        )
        wh_refit_summary[str(r_val)] = {
            "c1_formal": cf,
            "c1_min": min(c1_ests),
            "c1_max": max(c1_ests),
        }

    return wh_refit_summary


class FejerTodaSea:
    """
    Independent finite-sea Toda Bethe-ansatz solver using Fejer's first rule
    (Chebyshev-Gauss roots x_j = -cos((2j-1)*pi/(2N))) on [-q, q], distinct from
    the Gauss-Legendre Nystrom solver in uniform_limits_scan.py.
    """

    def __init__(self, h, N=360):
        self.h = float(h)
        self.N = int(N)
        self.b = math.exp(-0.5)
        k = np.arange(1, self.N + 1, dtype=float)
        theta = (2 * k - 1) * math.pi / (2 * self.N)
        self.xi = -np.cos(theta)
        w = np.ones(self.N, dtype=float)
        for m in range(1, self.N // 2 + 1):
            w -= (2.0 / (4 * m * m - 1)) * np.cos(2 * m * theta)
        self.wi = (2.0 / self.N) * w
        self.q = self._find_fermi_q()
        self._setup_sea(self.q)

    def _kernel(self, diff):
        z = 1.0 + 1j * (diff / self.h)
        return 2.0 * np.real(scipy_psi(z)) + 2.0 * math.log(self.h)

    def _setup_sea(self, q):
        self.x = q * self.xi
        self.w = q * self.wi
        diff = self.x[:, None] - self.x[None, :]
        K = self._kernel(diff)
        self.A = 2.0 * math.pi * self.h * np.eye(self.N) - K * self.w[None, :]
        sol = np.linalg.solve(self.A, np.column_stack([np.ones(self.N), self.x]))
        self.rho = sol[:, 0]
        self.eta = sol[:, 1]

    def _count_residual(self, q):
        self._setup_sea(q)
        return float(np.dot(self.w, self.rho) - 1.0)

    def _find_fermi_q(self):
        q0 = 2.0 * self.b + self.h
        q1 = 2.0 * self.b + 2.0 * self.h
        f0 = self._count_residual(q0)
        for _ in range(12):
            f1 = self._count_residual(q1)
            if abs(f1) < 1e-14 or abs(q1 - q0) < 1e-14:
                break
            q2 = q1 - f1 * (q1 - q0) / (f1 - f0)
            q0, f0, q1 = q1, f1, q2
        return q1

    def theta_antideriv(self, d):
        from scipy.special import loggamma as scipy_loggamma

        return 2.0 * self.h * np.imag(scipy_loggamma(1.0 + 1j * d / self.h)) + 2.0 * d * math.log(self.h)

    def P_dressed(self, p):
        th_diff = self.theta_antideriv(p - self.x) - self.theta_antideriv(self.q - self.x)
        return (p - self.q) + float(np.dot(self.w * th_diff, self.rho))

    def D_dressed(self, p):
        return 1.0 + float(np.dot(self.w * self._kernel(p - self.x), self.rho))

    def solve_p_for_P(self, P_target):
        p = self.q + max(0.2, P_target)
        for _ in range(15):
            f_val = self.P_dressed(p) - P_target
            df_val = self.D_dressed(p)
            dp = f_val / df_val
            p -= dp
            if abs(dp) < 1e-14:
                break
        return p

    def energy_coeff(self, kappa):
        P0 = 4.0 * self.b * (kappa * math.cosh(kappa) - math.sinh(kappa))
        E0 = 2.0 * (self.b**2) * (math.sinh(kappa) * math.cosh(kappa) - kappa)
        p = self.solve_p_for_P(P0)
        th_diff = self.theta_antideriv(p - self.x) - self.theta_antideriv(self.q - self.x)
        nonlin = (p - self.q) * (0.5 * (p + self.q) - self.b) + float(
            np.dot(self.w * th_diff, self.eta - self.b * self.rho)
        )
        cl_nonlin = E0 - self.b * P0
        return 3.0 * (nonlin - cl_nonlin) / (self.h * P0)


def audit_finite_sea_and_refits():
    """
    Section 5 (Saved-data and fresh-scan refits, independent Fejer Nystrom solver,
    error budgets, and sensitivity checks):
    1. Cross-check finite-sea observables with independent FejerTodaSea solver at h=0.02, kappa1=0.35.
    2. Audit onebody_prec_scan.json refit:
       - Verify package free intercept C_T = 0.15915494136 +/- 8.23e-6 when -3d/(2y) is subtracted.
       - Independent sensitivity test: leave the 1/y coefficient FREE as well (basis (1, y^-1, y^-2, ...))
         and record the widened spread.
    3. Audit Phase 2d (toda_edge_twoterm.json): assert the 3 violations > 0.004 and step ratio 0.500415.
    4. Audit Phase 2f (matching_toda_data.json + matching_toda_fine.json): assert sign reversal of c_1
       both before and after classical lattice phase subtraction.
    5. Audit fresh fixed-kappa_1 and growing-Y scans (uniform_limits_scan.json):
       - Verify why fixed-kappa_1 h-extrapolation has huge spread at kappa_1 = 0.12, 0.18:
         the edge parameter is y = b*kappa_1^2/h, so 1/y = h/(b*kappa_1^2) = 2.29 > 1 at (0.12, 0.02)!
       - Verify the minor table reporting qualification at kappa_1 = 0.35 (energy mesh 4.99e-9 vs phase mesh 4.54e-9).
    """
    # 1. Independent Fejer Nystrom cross-check against uniform_limits_scan.json at h=0.02, kappa1=0.35
    scan_data = json.loads((ROOT / "uniform_limits_scan.json").read_text(encoding="utf-8"))
    ref_row = [r for r in scan_data["rows"] if abs(r["h"] - 0.02) < 1e-9 and r["factor"] == 1][0]
    ref_pt = [p for p in ref_row["fixed"] if abs(p["kappa_fixed"] - 0.35) < 1e-9][0]

    fejer = FejerTodaSea(0.02, N=360)
    check(
        "independent_fejer_fermi_q_h0.02",
        abs(fejer.q - ref_row["q"]) < 2e-7,
        f"q_fejer={fejer.q:.10f}, q_gl={ref_row['q']:.10f}",
    )
    b_val = math.exp(-0.5)
    CE_fejer = fejer.energy_coeff(0.35)
    check(
        "independent_fejer_onebody_energy_h0.02_k0.35",
        abs(CE_fejer - ref_pt["energy"]) < 2e-7,
        f"CE_fejer={CE_fejer:.10f}, CE_gl={ref_pt['energy']:.10f}",
    )

    # 2. One-body saved-data refit and sensitivity to leaving 1/y free
    onebody = json.loads((ROOT / "onebody_prec_scan.json").read_text(encoding="utf-8"))["rows"]
    by_h = {}
    for row in onebody:
        by_h.setdefault(row["h"], {})[row["factor"]] = row
    hs = sorted(by_h.keys(), reverse=True)
    ys = [p["y"] for p in by_h[hs[0]][1]["points"] if p["y"] >= 6.0]

    # Verify package refit numbers in uniform_limits_refit.json
    refit_pkg = json.loads((ROOT / "uniform_limits_refit.json").read_text(encoding="utf-8"))
    ob_pkg = refit_pkg["onebody"]
    ob_center = float(ob_pkg["value"])
    ob_tot = float(ob_pkg["total"])
    ob_amp_err = float(ob_pkg["amplitude_error"])
    check(
        "onebody_refit_matches_saved_json",
        abs(ob_center - 0.15915494136) < 1e-10
        and abs(ob_tot - 8.23e-6) < 1e-7
        and abs(ob_center - 1.0 / (2.0 * math.pi)) < ob_tot,
        f"center={ob_center:.11f}, 1/(2*pi)={1/(2*math.pi):.11f}, unc={ob_tot:.3e}",
    )

    # Independent sensitivity test: using the exact same h -> 0 extrapolated points ob_pkg["points"]
    # and windows (4,32), (8,32), (8,64), (16,64), leave the 1/y coefficient FREE as well
    pts_h0 = [(float(p["y"]), float(p["value"])) for p in ob_pkg["points"]]
    free_1y_intercepts = []
    for lo, hi in [(4, 32), (8, 32), (8, 64), (16, 64)]:
        sub = [(y_v, val) for y_v, val in pts_h0 if lo <= y_v <= hi]
        yw = np.array([s[0] for s in sub], dtype=float)
        cw = np.array([s[1] for s in sub], dtype=float)
        for deg in [2, 3]:
            X_y = np.column_stack([yw ** (-k) for k in range(deg + 1)])
            coef, *_ = np.linalg.lstsq(X_y, cw, rcond=None)
            free_1y_intercepts.append(float(coef[0]))
    spread_free_1y = 0.5 * (max(free_1y_intercepts) - min(free_1y_intercepts))
    center_free_1y = 0.5 * (max(free_1y_intercepts) + min(free_1y_intercepts))
    check(
        "onebody_free_1y_slope_sensitivity",
        abs(center_free_1y - 1.0 / (2.0 * math.pi)) <= spread_free_1y + float(ob_pkg["h_error"])
        and spread_free_1y > ob_amp_err,
        f"free-(1,1/y,...) center={center_free_1y:.10f} +/- {spread_free_1y:.2e} (vs fixed -3d/(2y) y_spread={ob_amp_err:.2e})",
    )

    # 3. Phase 2d step-bound violations (> 0.004)
    p2d = json.loads((ROOT / "toda_edge_twoterm.json").read_text(encoding="utf-8"))
    violations = []
    for key, rows in p2d.items():
        for i in range(len(rows) - 1):
            step = abs(rows[i + 1][1] - rows[i][1])
            if step > 0.004:
                violations.append((key, rows[i][0], rows[i + 1][0], step))
    check(
        "phase2d_three_step_bound_violations",
        len(violations) == 3
        and abs(violations[0][3] - 0.006326022) < 1e-6
        and abs(violations[1][3] - 0.009016413) < 1e-6
        and abs(violations[2][3] - 0.004511945) < 1e-6,
        f"found {len(violations)} violations > 0.004: {[(v[0], round(v[3], 6)) for v in violations]}",
    )

    # 4. Phase 2f classical-subtracted and unsubtracted sign reversals
    p2f_sub_fits = refit_pkg["saved_classical_subtracted_fits"]
    p2f_unsub_list = {row["r"]: row["slopes"] for row in refit_pkg["published_sign_reversal"]}
    for r_key in ["0.4", "0.5", "0.6"]:
        c1_vals = [float(f["coef"][1]) for f in p2f_sub_fits if f["r"] == r_key]
        unsub_s = p2f_unsub_list[r_key]
        check(
            f"phase2f_sign_reversal_r={r_key}",
            min(c1_vals) < 0 < max(c1_vals)
            and float(unsub_s["residual"]) < 0 < float(unsub_s["h_log_h_intercept"]),
            f"subtracted=[{min(c1_vals):.4f}, {max(c1_vals):.4f}], unsubtracted=({float(unsub_s['residual']):.4f} vs {float(unsub_s['h_log_h_intercept']):.4f})",
        )

    # 5. Quantitative explanation of fixed-kappa_1 h-fit spread at small kappa_1 = 0.12, 0.18
    inv_y_k012_h002 = 0.02 / (b_val * 0.12**2)
    inv_y_k012_h00025 = 0.0025 / (b_val * 0.12**2)
    check(
        "small_kappa1_non_asymptotic_1_over_y",
        inv_y_k012_h002 > 2.0 and inv_y_k012_h00025 > 0.25,
        f"At kappa_1=0.12, 1/y = h/(b*kappa_1^2) ranges from {inv_y_k012_h002:.2f} (h=0.02) to {inv_y_k012_h00025:.3f} (h=0.0025)",
    )

    # Check table mesh diagnostic reporting at kappa_1 = 0.35 in uniform_limits_refit.json
    k035_row = [r for r in refit_pkg["fixed_kappa"] if abs(float(r["k"]) - 0.35) < 1e-9][0]
    k035_e_mesh = float(k035_row["energy"]["mesh"])
    k035_p_mesh = float(k035_row["phase"]["mesh"])
    check(
        "table_k035_mesh_diagnostic_comparison",
        k035_e_mesh > k035_p_mesh and abs(k035_p_mesh - 4.536e-9) < 1e-11 and abs(k035_e_mesh - 4.990e-9) < 1e-11,
        f"At kappa_1=0.35, energy.mesh={k035_e_mesh:.3e} vs phase.mesh={k035_p_mesh:.3e} (table prints 4.54e-9)",
    )

    return {
        "fejer_q_h0.02": fejer.q,
        "fejer_CE_h0.02_k0.35": CE_fejer,
        "onebody_free_1y_center": center_free_1y,
        "onebody_free_1y_spread": spread_free_1y,
        "phase2d_violations": len(violations),
        "inv_y_k012_h002": inv_y_k012_h002,
    }


def main():
    t0 = time.time()
    summary = {
        "conventions": audit_conventions_and_scales(),
        "kernel": audit_kernel_boundary_layer(),
        "wh_and_matched": audit_wh_reversion_and_matched_proof(),
        "independent_wh_convolution": audit_independent_wh_convolution(),
        "finite_sea_and_refits": audit_finite_sea_and_refits(),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    print(f"Completed all independent audit checks in {summary['elapsed_seconds']}s")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

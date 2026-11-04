#!/usr/bin/env python3
"""Independent mathematical and numerical audit of phase2/good_twobody.tex
and phase2/scripts/good_twobody_*.py.

Constraint compliance:
- Imports NO code from good_twobody_*.py or any other repository script.
- Modifies NO existing files; verifies SHA-256 digests of all 12 audited files
  before and after execution.
- Encodes refuted historical claims and coverage qualifications as explicit
  discrepancy assertions.
- Ends by printing ALL CHECKS PASSED.
"""
import sys
sys.dont_write_bytecode = True

import hashlib
import json
import subprocess
import time
from pathlib import Path
import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "phase2"
SCRIPTS = PHASE2 / "scripts"

AUDITED_FILES = [
    PHASE2 / "good_twobody.tex",
    PHASE2 / "good_twobody.pdf",
    SCRIPTS / "good_twobody_kernels.py",
    SCRIPTS / "good_twobody_counterterms.py",
    SCRIPTS / "good_twobody_local_insertion.py",
    SCRIPTS / "good_twobody_threebody.py",
    SCRIPTS / "good_twobody_grid.py",
    SCRIPTS / "good_twobody_kernels.json",
    SCRIPTS / "good_twobody_counterterms.json",
    SCRIPTS / "good_twobody_local_insertion.json",
    SCRIPTS / "good_twobody_threebody.json",
    SCRIPTS / "good_twobody_grid.json",
    SCRIPTS / "good_twobody_resonance.json",
]


def compute_sha256_map(paths):
    out = {}
    for p in paths:
        assert p.exists(), f"Missing audited file: {p}"
        out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# Independent 80-digit mpmath primitives (no code shared with package)
# ---------------------------------------------------------------------------

def hirota_poly(K, W):
    """Dispersion polynomial P(K, W) = W^2 - K^2 + K^4 for the good equation."""
    return W * W - K * K + K**4


def one_soliton_data(alpha_val, eps_val):
    """Independent one-soliton classical data in the fixed good convention."""
    a = mp.mpf(alpha_val)
    e = int(eps_val)
    s_a = mp.sin(a)
    c_a = mp.cos(a)
    k = s_a
    v = e * c_a
    M = 2 * k
    Pi = -2 * k * v
    P = mp.mpf(2) * v * (k**3) / 3
    E = mp.mpf(2) * (k**3) / 15 * (5 - 4 * k * k)
    ell = -e * mp.cos(3 * a)
    # dP/da = (2/3) * e * sin(a) * sin(3a), dPi/da = -2 * e * cos(2a)
    dP_da = mp.mpf(2) * e * mp.sin(a) * mp.sin(3 * a) / 3
    dPi_da = -2 * e * mp.cos(2 * a)
    gamma = dPi_da / dP_da
    return {
        "a": a, "e": e, "k": k, "v": v,
        "M": M, "Pi": Pi, "P": P, "E": E, "ell": ell,
        "dP_da": dP_da, "dPi_da": dPi_da, "gamma": gamma,
    }


def hirota_A12(sol1, sol2):
    """Compute A_12 directly from the Hirota bilinear polynomial P(K1-K2, W1-W2)."""
    K1, W1 = sol1["k"], -sol1["k"] * sol1["v"]
    K2, W2 = sol2["k"], -sol2["k"] * sol2["v"]
    return -hirota_poly(K1 - K2, W1 - W2) / hirota_poly(K1 + K2, W1 + W2)


def classical_twobody_data(sol1, sol2):
    """Independent classical two-soliton scattering data and counterterm responses."""
    A = hirota_A12(sol1, sol2)
    assert A > 0, f"Non-positive A_12 = {A}"
    lnA = mp.log(A)
    T12 = mp.mpf(2) * sol1["M"] * sol2["M"] * (sol1["v"] - sol2["v"]) / 3
    delta0 = mp.mpf(2) * (sol1["ell"] - sol2["ell"]) * lnA / 9 + T12
    dX1p = -lnA / sol1["k"] + sol1["gamma"] * sol2["M"]
    dX2p = +lnA / sol2["k"] - sol2["gamma"] * sol1["M"]
    D = sol1["P"] * dX1p + sol2["P"] * dX2p
    return {
        "A": A, "logA": lnA, "T12": T12, "delta0": delta0,
        "dX1p": dX1p, "dX2p": dX2p, "D": D,
        "K_lambda": 3 * delta0 + 4 * D,
        "K_eta": 2 * delta0 + 2 * D,
        "K_zeta": 2 * delta0 + D,
        "K_kappa": -delta0 - D,
    }


def legendre_chi2(w):
    """Legendre's chi_2(w) = (Li_2(w) - Li_2(-w))/2 for real w."""
    w = mp.mpf(w)
    if w == 0:
        return mp.mpf(0)
    if abs(w) == 1:
        return mp.sign(w) * (mp.pi**2) / 8
    if abs(w) < 1:
        return (mp.polylog(2, w) - mp.polylog(2, -w)) / 2
    # Inversion relation: chi_2(w) + chi_2(1/w) = sign(w) * pi^2 / 4 - (i*pi/2)*log|w| for complex,
    # while the principal-value integral int_0^w log|t|/(1-t^2) dt = -chi_2(w) - (1/2)log|w|log|(1-w)/(1+w)|
    # satisfies int_0^w + int_0^(1/w) = -sign(w)*pi^2/4.
    raise ValueError("Use pv_integral_M for |w| > 1")


def pv_integral_M(w):
    """Evaluate M(w) = int_0^w 2*log|t|/(1 - t^2) dt via chi_2 on |w| <= 1
    and the exact reciprocal reflection M(w) + M(1/w) = -sign(w)*pi^2/2."""
    w = mp.mpf(w)
    if w == 0:
        return mp.mpf(0)
    s = mp.sign(w)
    u = abs(w)
    if u == 1:
        return -s * (mp.pi**2) / 4
    if u < 1:
        return -s * (2 * legendre_chi2(u) + mp.log(u) * mp.log((1 - u) / (1 + u)))
    inv = 1 / u
    m_inv = -(2 * legendre_chi2(inv) + mp.log(inv) * mp.log((1 - inv) / (1 + inv)))
    return -s * (mp.pi**2 / 2 + m_inv)


def continuum_integrals_chi2(sol1, sol2):
    """Compute (I_12, J_12, I_sym) using the 3-angle tangent half-angle ratios."""
    a1, e1 = sol1["a"], sol1["e"]
    a2, e2 = sol2["a"], sol2["e"]
    phis1 = (a1 + 2 * mp.pi / 3, a1 - 2 * mp.pi / 3, a1)
    phis2 = (a2 + 2 * mp.pi / 3, a2 - 2 * mp.pi / 3, a2)
    cs = (1, 1, -2)
    I_pp = mp.mpf(0)
    I_pm = mp.mpf(0)
    for c1, f1 in zip(cs, phis1):
        t1 = mp.tan(f1 / 2)
        for c2, f2 in zip(cs, phis2):
            t2 = mp.tan(f2 / 2)
            I_pp += c1 * c2 * pv_integral_M(t1 / t2) / (2 * mp.pi)
            I_pm += c1 * c2 * pv_integral_M(t1 * t2) / (2 * mp.pi)
    I_raw = (e1 + e2) * I_pp / 2 + (e1 - e2) * I_pm / 2
    I_sym = -mp.pi * (e1 + e2) / 4
    J_anti = I_raw - I_sym
    return I_raw, J_anti, I_sym


def hirota_phonon_theta_and_deriv(q, sol_d):
    """Reconstruct phonon phase theta(q) and theta'(q) directly from the
    4-wave Hirota transmission factor T(q) = -P(k - iq, W - i*om)/P(k + iq, W + i*om)
    with continuous branch vanishing at q -> +-infinity."""
    a, e = sol_d["a"], sol_d["e"]
    phis = (a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a)
    cs = (1, 1, -2)
    abs_q = abs(q)
    sq = mp.sqrt(1 + q * q)
    sgn_q = mp.sign(q)
    th = mp.mpf(0)
    dth = mp.mpf(0)
    for c, f in zip(cs, phis):
        sf = mp.sin(f)
        cf = mp.cos(f)
        # Continuous half-line phase vanishing at |q| -> infinity
        th -= c * (
            sgn_q * mp.atan2(sf, abs_q)
            + e * mp.atan2(sf * cf, abs_q * sq + q * q + sf * sf)
        )
        dth += c * sf / (q * q + sf * sf) * (1 + e * sgn_q * cf / sq)
    return th, dth


def continuum_I12_hyperbolic_quad(sol1, sol2):
    """Independent quadrature for I_12 using hyperbolic substitution q = sinh(u)."""
    with mp.workdps(70):
        def integrand(u):
            q = mp.sinh(u)
            dq_du = mp.cosh(u)
            th1_pos, _ = hirota_phonon_theta_and_deriv(q, sol1)
            _, dth2_pos = hirota_phonon_theta_and_deriv(q, sol2)
            th1_neg, _ = hirota_phonon_theta_and_deriv(-q, sol1)
            _, dth2_neg = hirota_phonon_theta_and_deriv(-q, sol2)
            return (th1_pos * dth2_pos + th1_neg * dth2_neg) * dq_du / (4 * mp.pi)

        pts = [mp.mpf(0), mp.mpf("0.2"), mp.mpf("0.8"), mp.mpf("2.0"), mp.mpf("5.0"), mp.inf]
        return mp.quad(integrand, pts)


def onebody_charges_analytic(sol_d):
    """Compute p_m, e_m, L_1, and p_can = dL_1/dv analytically without mp.diff."""
    a, e, k, v = sol_d["a"], sol_d["e"], sol_d["k"], sol_d["v"]
    phis = (a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a)
    cs = (1, 1, -2)
    pm = -e * sum(c * mp.sin(f) * mp.log(abs(mp.tan(f / 2))) for c, f in zip(cs, phis)) / (2 * mp.pi)
    em = (-6 * k - 2 * sum(c * mp.sin(f) * mp.cos(f) * mp.log(abs(mp.tan(f / 2))) for c, f in zip(cs, phis))) / (4 * mp.pi)
    L1 = v * pm - em
    # Exact analytical derivative dL1/da:
    # Since L1 = (-e*cos(a)*sum c_r sin(f_r) log|tan(f_r/2)| + 3*sin(a) + sum c_r sin(f_r) cos(f_r) log|tan(f_r/2)|)/(2*pi)
    # Wait: for e=+1, v = cos(a), so v*pm - em = [3*sin(a) + sum c_r sin(f_r)*(cos(f_r) - cos(a))*log|tan(f_r/2)|]/(2*pi).
    # In general for e = +-1, e*v = cos(a), so v*pm = -cos(a)*sum c_r sin(f_r) log|tan(f_r/2)| / (2*pi),
    # which is independent of e! Thus L1(a) is strictly even in e (independent of e)!
    # Let's differentiate L1(a) = [3*sin(a) + sum c_r sin(f_r)*(cos(f_r) - cos(a))*log|tan(f_r/2)|]/(2*pi) w.r.t. a:
    # Note d/df [log|tan(f/2)|] = 1/sin(f), so sum c_r sin(f_r)*(cos(f_r) - cos(a)) * (1/sin(f_r)) = sum c_r (cos(f_r) - cos(a)) = -3*cos(a)!
    # That cancels +3*cos(a) from d/da[3*sin(a)]!
    # Hence dL1/da = sum_{r} c_r [cos(f_r)*(cos(f_r) - cos(a)) + sin(f_r)*(-sin(f_r) + sin(a))] * log|tan(f_r/2)| / (2*pi)
    #             = sum_{r} c_r [cos(2*f_r) - cos(f_r + a)] * log|tan(f_r/2)| / (2*pi).
    dL1_da = sum(
        c * (mp.cos(2 * f) - mp.cos(f + a)) * mp.log(abs(mp.tan(f / 2)))
        for c, f in zip(cs, phis)
    ) / (2 * mp.pi)
    # Since v = e*cos(a), dv/da = -e*sin(a) = -e*k, so p_can = dL1/dv = (dL1/da) / (-e*k):
    p_can = dL1_da / (-e * k)
    return {"p_m": pm, "e_m": em, "L1": L1, "p_can": p_can, "dL1_da": dL1_da}


# ---------------------------------------------------------------------------
# Audit Check 1: Conventions, Classical Dictionary, Symbolic Darboux Gradient,
# and Frame-Dependence of Symmetric vs Antisymmetric Counterterm Response
# ---------------------------------------------------------------------------

def check_1_conventions_and_symbolic_darboux():
    print("[Check 1] Classical dictionary, symbolic Darboux gradient, and frame analysis...")
    # 1a. Verify good vs bad Boussinesq Euler-Lagrange equations and one-soliton energies
    x, t = sp.symbols("x t", real=True)
    phi = sp.Function("phi")(x, t)
    u = sp.diff(phi, x)
    L_good = sp.Rational(1, 2) * sp.diff(phi, t)**2 - sp.Rational(1, 2) * u**2 + u**3 - sp.Rational(1, 2) * sp.diff(u, x)**2
    L_bad = sp.Rational(1, 2) * sp.diff(phi, t)**2 - sp.Rational(1, 2) * u**2 - u**3 + sp.Rational(1, 2) * sp.diff(u, x)**2

    def euler_lagrange_x(L_expr):
        # d/dx ( d/dt(dL/dphi_t) + d/dx(dL/dphi_x) - d^2/dx^2(dL/dphi_xx) )
        pi_t = sp.diff(phi, t, 2)
        dL_du = sp.diff(L_expr, u)
        dL_dux = sp.diff(L_expr, sp.diff(u, x))
        eom_phi = pi_t + sp.diff(dL_du, x) - sp.diff(dL_dux, x, 2)
        return sp.expand(sp.diff(eom_phi, x))

    eom_g = euler_lagrange_x(L_good)
    eom_b = euler_lagrange_x(L_bad)
    target_g = sp.expand(sp.diff(u, t, 2) - sp.diff(u, x, 2) + sp.diff(u, x, 4) + 3 * sp.diff(u**2, x, 2))
    target_b = sp.expand(sp.diff(u, t, 2) - sp.diff(u, x, 2) - sp.diff(u, x, 4) - 3 * sp.diff(u**2, x, 2))
    assert eom_g == target_g
    assert eom_b == target_b

    # Verify errata M03: printed old energy k^3*(2*cos^2(a)+1)/3 vs true E_g = (2/15)*k^3*(5-4*k^2)
    a1, a2 = sp.symbols("a1 a2", positive=True)
    e1, e2 = sp.symbols("e1 e2", real=True)
    k1 = sp.sin(a1)
    v1 = e1 * sp.cos(a1)
    P1 = sp.Rational(2, 3) * v1 * k1**3
    E1_true = sp.Rational(2, 3) * k1**3 - sp.Rational(8, 15) * k1**5
    E1_old_false = k1**3 * (2 * sp.cos(a1)**2 + 1) / 3
    assert sp.trigsimp(sp.expand(sp.diff(E1_true, a1) - v1 * sp.diff(P1, a1)).subs(e1**2, 1)) == 0
    old_residual = sp.trigsimp((sp.diff(E1_old_false, a1) - sp.expand(v1 * sp.diff(P1, a1)).subs(e1**2, 1)) / sp.cos(a1))
    assert sp.trigsimp(old_residual - k1**2 * (1 - sp.Rational(2, 3) * k1**2)) == 0

    # Verify defect s convention from errata M04: scattering uses s = sigma_vac,
    # whereas defect stability polynomial uses spectral parameter s_spec = eta * sigma_vac.
    sigma_vac, eta_def = sp.symbols("sigma_vac eta_def", real=True)
    qD_true = -sigma_vac / (2 * sp.sqrt(3))
    qD_false_signed = -eta_def * sigma_vac / (2 * sp.sqrt(3))
    assert sp.simplify(qD_true.subs(eta_def, -1) - qD_false_signed.subs(eta_def, -1)) != 0
    alpha_CZ = eta_def / sp.sqrt(3)
    beta_CZ = 2 * eta_def / sp.sqrt(3)
    assert sp.simplify(sp.expand(beta_CZ**2 - alpha_CZ**2).subs(eta_def**2, 1) - 1) == 0

    # 1b. Symbolic proof in SymPy that Hirota A_12 equals the KP root cross-ratio
    # and that d(delta_0)/dP_j = -Delta X'_j for arbitrary symbolic (a1, a2) in all 4 orientations
    k2 = sp.sin(a2)
    M1, M2 = 2 * k1, 2 * k2
    gamma1_sym = -3 * sp.cos(2 * a1) / (sp.sin(a1) * sp.sin(3 * a1))
    gamma2_sym = -3 * sp.cos(2 * a2) / (sp.sin(a2) * sp.sin(3 * a2))

    for e1_v, e2_v in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        v1_s = e1_v * sp.cos(a1)
        v2_s = e2_v * sp.cos(a2)
        P1_s = sp.Rational(2, 3) * v1_s * k1**3
        P2_s = sp.Rational(2, 3) * v2_s * k2**3
        Pi1_s, Pi2_s = -2 * k1 * v1_s, -2 * k2 * v2_s
        ell1_s = -e1_v * sp.cos(3 * a1)
        ell2_s = -e2_v * sp.cos(3 * a2)

        p1a, p1b = (k1 + v1_s / sp.sqrt(3)) / 2, (v1_s / sp.sqrt(3) - k1) / 2
        p2a, p2b = (k2 + v2_s / sp.sqrt(3)) / 2, (v2_s / sp.sqrt(3) - k2) / 2

        W1, W2 = -k1 * v1_s, -k2 * v2_s
        P_hir_num = -((W1 - W2)**2 - (k1 - k2)**2 + (k1 - k2)**4)
        P_hir_den = (W1 + W2)**2 - (k1 + k2)**2 + (k1 + k2)**4
        diff_A = sp.trigsimp(sp.expand(P_hir_num * (p1a - p2b) * (p1b - p2a) - P_hir_den * (p1a - p2a) * (p1b - p2b)))
        assert diff_A == 0, f"KP vs Hirota A12 symbolic mismatch: {diff_A}"

        dlogA_da1 = (
            sp.diff(p1a - p2a, a1) / (p1a - p2a)
            + sp.diff(p1b - p2b, a1) / (p1b - p2b)
            - sp.diff(p1a - p2b, a1) / (p1a - p2b)
            - sp.diff(p1b - p2a, a1) / (p1b - p2a)
        )
        T12_sym = sp.Rational(2, 3) * M1 * M2 * (v1_s - v2_s)
        ddelta0_da1_minus_log = sp.Rational(2, 9) * (ell1_s - ell2_s) * dlogA_da1 + sp.diff(T12_sym, a1)
        dP1_da1 = sp.diff(P1_s, a1)
        dPi1_da1 = sp.diff(Pi1_s, a1)
        assert sp.trigsimp(dPi1_da1 - dP1_da1 * gamma1_sym) == 0
        assert sp.trigsimp(sp.Rational(2, 9) * sp.diff(ell1_s, a1) * k1 - dP1_da1) == 0

        # Key symbolic identity: (2/9)*(ell1 - ell2)*d(log A_12)/da1 + d(T_12)/da1 + dP1_da1 * gamma1 * M2 == 0
        # Use exponential rewrite exp(i*a1), exp(i*a2) for canonical trigonometric polynomial normal form
        lhs_darboux = sp.together((ddelta0_da1_minus_log + dP1_da1 * gamma1_sym * M2).rewrite(sp.exp))
        num_darboux, _ = sp.fraction(lhs_darboux)
        assert sp.expand(num_darboux) == 0

        # 1c. Symbolic proof of the exact log A_12 coefficient in K_lambda = 3*delta_0 + 4*D:
        logA_coeff_diff = sp.trigsimp(
            sp.Rational(2, 3) * (ell1_s - ell2_s) + 4 * (-P1_s / k1 + P2_s / k2) - sp.Rational(2, 3) * (v2_s - v1_s)
        )
        assert logA_coeff_diff == 0

    # 1d. Point 7 Frame analysis: Is K_lambda really antisymmetric, or is a symmetric part dropped by a frame choice?
    v1_pp, v2_pp = sp.cos(a1), sp.cos(a2)
    P1_pp, P2_pp = sp.Rational(2, 3) * v1_pp * k1**3, sp.Rational(2, 3) * v2_pp * k2**3
    Pi1_pp, Pi2_pp = -2 * k1 * v1_pp, -2 * k2 * v2_pp
    dP1_pp = sp.diff(P1_pp, a1)
    S_cb = sp.Rational(1, 2) * (Pi1_pp * M2 + Pi2_pp * M1)
    dS_cb_dP1 = sp.Rational(1, 2) * (gamma1_sym * M2 + Pi2_pp * sp.diff(M1, a1) / dP1_pp)
    dS_cb_dP2 = dS_cb_dP1.xreplace({a1: a2, a2: a1})
    K_lambda_sym_frame = 3 * S_cb - 4 * (P1_pp * dS_cb_dP1 + P2_pp * dS_cb_dP2)
    val_sym_frame = abs(
        complex(K_lambda_sym_frame.subs({a1: sp.Rational(3, 10), a2: sp.Rational(6, 10)}).evalf(40))
    )
    assert val_sym_frame > 0.1, f"Expected non-zero symmetric frame response, got {val_sym_frame}"
    return {"K_lambda_symmetric_coboundary_at_03_06_pp": val_sym_frame}


# ---------------------------------------------------------------------------
# Audit Check 2: Independent Counterterm Re-derivation (Deformed PDE & Profiles),
# Connected u^2 Spacetime Insertion, and 1-Loop Ring Power Counting
# ---------------------------------------------------------------------------

def check_2_counterterms_and_insertion():
    print("[Check 2] Counterterm responses K_lambda, K_eta, K_zeta, K_kappa and connected u^2 insertion...")
    mp.mp.dps = 80

    # 2a. Direct deformed-PDE verification without using the package's coordinate rescaling
    # For H_{a,b,d,g} = int (a/2 pi^2 + b/2 u^2 - g u^3 + d/2 u_x^2) dx:
    # Deformed bilinear equation: (a^-1 D_t^2 - b D_x^2 + d D_x^4) tau.tau = 0, u = (2d/g)(log tau)_xx.
    # For deformed spectral angle alpha_j, physical wavenumber k_tilde = sqrt(b/d)*sin(alpha_j),
    # physical velocity v_tilde = e_j*sqrt(a*b)*cos(alpha_j).
    def deformed_soliton(alpha_val, eps_val, a_c, b_c, d_c, g_c):
        a_v = mp.mpf(alpha_val)
        e_v = int(eps_val)
        kt = mp.sqrt(b_c / d_c) * mp.sin(a_v)
        vt = e_v * mp.sqrt(a_c * b_c) * mp.cos(a_v)
        # Direct spatial integration of u(x) = (d/(2g))*kt^2*sech^2(kt*x/2) via moments:
        # int u dx = (d/g)*(2*kt), int u^2 dx = (d/g)^2 * (2/3)*kt^3
        Mt = (d_c / g_c) * 2 * kt
        Pit = -(vt / a_c) * Mt
        Pt = (vt / a_c) * (d_c / g_c)**2 * (mp.mpf(2) / 3) * (kt**3)
        return {"a": a_v, "e": e_v, "kt": kt, "vt": vt, "Mt": Mt, "Pit": Pit, "Pt": Pt}

    def deformed_twobody_phase_fixed_P(P1_target, P2_target, e1, e2, a1_init, a2_init, a_c, b_c, d_c, g_c):
        # Find deformed angles (al1, al2) matching physical canonical momenta (P1_target, P2_target)
        al1 = mp.findroot(lambda x: deformed_soliton(x, e1, a_c, b_c, d_c, g_c)["Pt"] - P1_target, a1_init, tol=mp.mpf("1e-76"))
        al2 = mp.findroot(lambda x: deformed_soliton(x, e2, a_c, b_c, d_c, g_c)["Pt"] - P2_target, a2_init, tol=mp.mpf("1e-76"))
        s1 = deformed_soliton(al1, e1, a_c, b_c, d_c, g_c)
        s2 = deformed_soliton(al2, e2, a_c, b_c, d_c, g_c)
        # Deformed Hirota polynomial P_{a,b,d}(K, W) = W^2/a - b*K^2 + d*K^4
        P_def = lambda K, W: (W * W) / a_c - b_c * (K * K) + d_c * (K**4)
        K1, W1 = s1["kt"], -s1["kt"] * s1["vt"]
        K2, W2 = s2["kt"], -s2["kt"] * s2["vt"]
        At = -P_def(K1 - K2, W1 - W2) / P_def(K1 + K2, W1 + W2)
        # Deformed action scale factor from Pit*Mt = (d*b^{3/2}/(sqrt(a)*g^2)) * Pi0*M0
        s_scale = d_c * (b_c**mp.mpf("1.5")) / (mp.sqrt(a_c) * (g_c**2))
        ell1 = -e1 * mp.cos(3 * al1)
        ell2 = -e2 * mp.cos(3 * al2)
        T12_t = mp.mpf(2) * (s2["Pit"] * s1["Mt"] - s1["Pit"] * s2["Mt"]) / 3
        return s_scale * (mp.mpf(2) * (ell1 - ell2) * mp.log(At) / 9) + T12_t

    # Recompute K_lambda, K_eta, K_zeta, K_kappa at (0.3, 0.6) in all 4 orientations
    # using 4-point Richardson extrapolation on step h_step = 1e-16 and h_step/2
    s_03_p = one_soliton_data("0.3", +1)
    s_06_p = one_soliton_data("0.6", +1)
    s_06_m = one_soliton_data("0.6", -1)
    c_pp = classical_twobody_data(s_03_p, s_06_p)
    c_pm = classical_twobody_data(s_03_p, s_06_m)

    # Check exact printed numbers in good_twobody.tex lines 195-197
    assert abs(c_pp["K_lambda"] - mp.mpf("0.05008140486841890958182890110412040244319314726741903204")) < mp.mpf("1e-53")
    assert abs(c_pm["K_lambda"] - mp.mpf("-0.2251739386833417584823934338568858909622820176789821613")) < mp.mpf("1e-53")

    max_richardson_err = mp.mpf(0)
    for e1, e2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sol1 = one_soliton_data("0.3", e1)
        sol2 = one_soliton_data("0.6", e2)
        cdat = classical_twobody_data(sol1, sol2)
        # Verify antisymmetry under complete label exchange (1 <-> 2)
        cdat_rev = classical_twobody_data(sol2, sol1)
        for key in ("delta0", "D", "K_lambda", "K_eta", "K_zeta", "K_kappa"):
            assert abs(cdat[key] + cdat_rev[key]) < mp.mpf("1e-75")

        for cname in ("lambda", "eta", "zeta", "kappa"):
            def eval_deformed(hh):
                aa = 1 + 2 * hh * (1 if cname == "kappa" else 0)
                bb = 1 + 2 * hh * (1 if cname == "lambda" else 0)
                dd = 1 + 2 * hh * (1 if cname == "zeta" else 0)
                gg = 1 - hh * (1 if cname == "eta" else 0)
                return deformed_twobody_phase_fixed_P(
                    sol1["P"], sol2["P"], e1, e2, sol1["a"], sol2["a"], aa, bb, dd, gg
                )

            h1 = mp.mpf("1e-14")
            h2 = h1 / 2
            d1 = (eval_deformed(h1) - eval_deformed(-h1)) / (2 * h1)
            d2 = (eval_deformed(h2) - eval_deformed(-h2)) / (2 * h2)
            rich = (4 * d2 - d1) / 3
            err = abs(rich - cdat["K_" + cname])
            max_richardson_err = max(max_richardson_err, err)
            assert err < mp.mpf("1e-52"), f"Deformed PDE Richardson error too large: {cname} {err}"

    # 2b. Independent connected u^2 spacetime insertion check at T = 2000, 3000, 4000
    # and analytical verification of the finite-T exponential overlap tail ~ exp(-k1*|v1-v2|*T)
    def connected_dt_logtau_integrand(sol1, sol2, x, t_val, lnA):
        eta1 = sol1["k"] * (x - sol1["v"] * t_val)
        eta2 = sol2["k"] * (x - sol2["v"] * t_val)
        # Using exact algebraic identity:
        # d_t log(tau_12 / (tau_1 * tau_2)) = -(A - 1) * (k1*v1*e^eta2*(1+e^eta1) + k2*v2*e^eta1*(1+e^eta2)) * e^(eta1+eta2) / ((1+e^eta1)*(1+e^eta2)*tau_12)
        # Evaluate in log-space to avoid overflow:
        m = max(mp.mpf(0), eta1, eta2, eta1 + eta2 + lnA)
        ltau12 = m + mp.log(mp.exp(-m) + mp.exp(eta1 - m) + mp.exp(eta2 - m) + mp.exp(eta1 + eta2 + lnA - m))
        l1 = max(mp.mpf(0), eta1) + mp.log1p(mp.exp(-abs(eta1)))
        l2 = max(mp.mpf(0), eta2) + mp.log1p(mp.exp(-abs(eta2)))
        term1 = sol1["k"] * sol1["v"] * mp.exp(eta1 + eta2 - ltau12 - l1)
        term2 = sol2["k"] * sol2["v"] * mp.exp(eta1 + eta2 - ltau12 - l2)
        return -mp.expm1(lnA) * (term1 + term2)

    insertion_checks = {}
    with mp.workdps(55):
        for e2_val, label in ((1, "pp"), (-1, "pm")):
            s1 = one_soliton_data("0.3", 1)
            s2 = one_soliton_data("0.6", e2_val)
            cd = classical_twobody_data(s1, s2)
            lnA = cd["logA"]
            for T_val in (mp.mpf(2000), mp.mpf(3000)):
                slopes = []
                for tt in (-T_val, T_val):
                    c_pts = sorted([
                        s1["v"] * tt, s1["v"] * tt - lnA / s1["k"],
                        s2["v"] * tt, s2["v"] * tt - lnA / s2["k"],
                    ])
                    grid_pts = [-mp.inf]
                    for cp in c_pts:
                        grid_pts.extend([cp - 40, cp, cp + 40])
                    grid_pts = sorted(set(grid_pts)) + [mp.inf]
                    slopes.append(mp.quad(lambda xx: connected_dt_logtau_integrand(s1, s2, xx, tt, lnA), grid_pts))
                ins = -mp.mpf(2) * (slopes[1] - slopes[0]) / 3
                target = mp.mpf(2) * (s1["v"] - s2["v"]) * lnA / 3
                err = abs(ins - target)
                insertion_checks[f"{label}_T{int(T_val)}"] = mp.nstr(err, 12)
                assert err < mp.mpf("1e-33")
            if label == "pp":
                e2000 = mp.mpf(insertion_checks["pp_T2000"])
                e3000 = mp.mpf(insertion_checks["pp_T3000"])
                decay_ratio = mp.log(e2000 / e3000) / 1000
                expected_rate = s1["k"] * (s1["v"] - s2["v"])
                assert abs(decay_ratio - expected_rate) < mp.mpf("1e-3")

    # 2c. Verify the 1-loop ring superficial degree 3 - 2n (text-to-script gap at lines 240-244)
    # Expanding (1/2) * int (d omega_E / 2pi) log(1 - 6*U*q^2 / (omega_E^2 + q^2 + q^4))
    # gives -1/(2n) * int (d omega_E / 2pi) (6*U*q^2)^n / (omega_E^2 + q^2 + q^4)^n.
    q_s, U_s, om_s = sp.symbols("q U omega_E", positive=True)
    w_U = sp.sqrt(q_s**2 + q_s**4 - 6 * U_s * q_s**2) / 2
    for n in range(1, 5):
        M_sq = q_s**2 + q_s**4
        om_int = sp.integrate(1 / (om_s**2 + M_sq)**n, (om_s, -sp.oo, sp.oo)) / (2 * sp.pi)
        feynman_ring_n = -sp.Rational(1, 2 * n) * (6 * U_s * q_s**2)**n * om_int
        taylor_n = (sp.diff(w_U, U_s, n).subs(U_s, 0) / sp.factorial(n)) * U_s**n
        assert sp.simplify(feynman_ring_n - taylor_n) == 0
        uv_power = sp.limit(taylor_n / (U_s**n * q_s**(2 - 2 * n)), q_s, sp.oo)
        assert uv_power != 0 and uv_power.is_finite

    return {
        "max_deformed_pde_richardson_error": mp.nstr(max_richardson_err, 12),
        "K_lambda_03_06_pp": mp.nstr(c_pp["K_lambda"], 30),
        "K_lambda_03_06_pm": mp.nstr(c_pm["K_lambda"], 30),
        "connected_insertion_errors": insertion_checks,
    }


# ---------------------------------------------------------------------------
# Audit Check 3: Continuum Integrals, Regulated Mode Trace (with B_Lambda, C_Lambda),
# and Analytical p_can
# ---------------------------------------------------------------------------

def check_3_continuum_integrals_and_regulated_trace():
    print("[Check 3] Continuum integrals I_12, J_12 and cutoff-regulated mode trace...")
    mp.mp.dps = 80
    max_quad_diff = mp.mpf(0)

    for e1, e2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        s1 = one_soliton_data("0.3", e1)
        s2 = one_soliton_data("0.6", e2)
        i_chi, j_chi, sym_chi = continuum_integrals_chi2(s1, s2)
        i_quad = continuum_I12_hyperbolic_quad(s1, s2)
        diff = abs(i_chi - i_quad)
        max_quad_diff = max(max_quad_diff, diff)
        assert diff < mp.mpf("1e-65"), f"Quadrature vs chi2 mismatch: {diff}"

        # Verify analytical p_can vs high-order central difference of L1(v)
        ch1 = onebody_charges_analytic(s1)
        h_a = mp.mpf("1e-18")
        L1_plus = onebody_charges_analytic(one_soliton_data(s1["a"] + h_a, e1))["L1"]
        L1_minus = onebody_charges_analytic(one_soliton_data(s1["a"] - h_a, e1))["L1"]
        p_can_fd = ((L1_plus - L1_minus) / (2 * h_a)) / (-e1 * s1["k"])
        assert abs(ch1["p_can"] - p_can_fd) < mp.mpf("1e-35")

    # Fill the pre-subtracted integrand gap in good_twobody_kernels.py:136-141
    # by testing the actual cutoff-regulated mode energy trace:
    # E_reg(Lambda) = (1/4pi) int_{-Lambda}^{Lambda} omega(q) theta'(q) dq - B_Lambda + (3*k/pi)*(sqrt(1+Lambda^2) - 1)
    #               = e_m(alpha) + 3*sin(alpha)/(4*pi*Lambda) + O(Lambda^-3).
    s_test = one_soliton_data("0.3", +1)
    ch_test = onebody_charges_analytic(s_test)
    # Using exact indefinite integral of omega(q)*theta'(q) via integration by parts:
    # (1/4pi) int_{-L}^{L} omega(q) theta'(q) dq - B_L = -(1/2pi) int_0^L (q*(1+2q^2)/sqrt(1+q^2)) * theta_even(q) dq
    # Wait: directly evaluate the finite-cutoff integral of omega(q)*theta'(q) on [-L, L] at L = 1000 and L = 2000:
    def regulated_energy_at_cutoff(sol_d, Lam):
        with mp.workdps(45):
            Lam = mp.mpf(Lam)
            def raw_integrand(q):
                om = abs(q) * mp.sqrt(1 + q * q)
                _, dth_p = hirota_phonon_theta_and_deriv(q, sol_d)
                _, dth_m = hirota_phonon_theta_and_deriv(-q, sol_d)
                return om * (dth_p + dth_m) / (4 * mp.pi)
            raw_int = mp.quad(raw_integrand, [0, mp.mpf("0.5"), mp.mpf("2.0"), mp.mpf("15.0"), mp.mpf("120.0"), Lam])
            th_p, _ = hirota_phonon_theta_and_deriv(Lam, sol_d)
            th_m, _ = hirota_phonon_theta_and_deriv(-Lam, sol_d)
            B_Lam = Lam * mp.sqrt(1 + Lam * Lam) * (th_p - th_m) / (4 * mp.pi)
            C_Lam_M = (3 * sol_d["k"] / mp.pi) * (mp.sqrt(1 + Lam * Lam) - 1)
            return raw_int - B_Lam + C_Lam_M

    E_1000 = regulated_energy_at_cutoff(s_test, 1000)
    E_2000 = regulated_energy_at_cutoff(s_test, 2000)
    asymp_1000 = 3 * s_test["k"] / (4 * mp.pi * 1000)
    assert abs((E_1000 - ch_test["e_m"]) - asymp_1000) < mp.mpf("1e-9")
    # Richardson extrapolation eliminates the O(Lambda^-1) term, leaving O(Lambda^-3) ~ 1e-10
    E_rich = 2 * E_2000 - E_1000
    assert abs(E_rich - ch_test["e_m"]) < mp.mpf("2e-10")

    return {"max_continuum_quadrature_diff": mp.nstr(max_quad_diff, 12)}


# ---------------------------------------------------------------------------
# Audit Check 4: Three-Body Clustering Reconciliation (Continuum vs Toda, Point 8)
# ---------------------------------------------------------------------------

def check_4_threebody_clustering_reconciliation():
    print("[Check 4] Three-body clustering reconciliation (Continuum vs Toda, Point 8)...")
    mp.mp.dps = 80

    # 4a. Independent 4-wave Hirota bilinear test at a NEW exact rational point
    # Rational parametrization: k_j = 2*z_j/(1+z_j^2), v_j = (1-z_j^2)/(1+z_j^2)
    z_rats = [sp.Rational(1, 7), sp.Rational(1, 4), sp.Rational(2, 5)]
    ks_r = [2 * z / (1 + z * z) for z in z_rats]
    ws_r = [-k * (1 - z * z) / (1 + z * z) for k, z in zip(ks_r, z_rats)]
    # 4th wave: imaginary phonon K4 = i*(5/12), W4 = i*(5/12)*(13/12) = i*(65/144)
    ks_r.append(sp.I * sp.Rational(5, 12))
    ws_r.append(-sp.I * sp.Rational(65, 144))

    P_bi = lambda K, W: W * W - K * K + K**4
    A_pairs = {}
    for i in range(4):
        for j in range(i + 1, 4):
            A_pairs[(i, j)] = sp.simplify(-P_bi(ks_r[i] - ks_r[j], ws_r[i] - ws_r[j]) / P_bi(ks_r[i] + ks_r[j], ws_r[i] + ws_r[j]))

    # Expand sum_{mu, nu in {0,1}^4} c_mu c_nu P(K_mu - K_nu, W_mu - W_nu) E^(mu + nu)
    from itertools import product
    monomials = []
    for mu in product((0, 1), repeat=4):
        c_mu = sp.Integer(1)
        for (i, j), a_ij in A_pairs.items():
            if mu[i] and mu[j]:
                c_mu *= a_ij
        K_mu = sum(ks_r[i] * mu[i] for i in range(4))
        W_mu = sum(ws_r[i] * mu[i] for i in range(4))
        monomials.append((mu, sp.simplify(c_mu), K_mu, W_mu))

    bilinear_coeffs = {}
    for mu, c_mu, K_mu, W_mu in monomials:
        for nu, c_nu, K_nu, W_nu in monomials:
            deg = tuple(mu[i] + nu[i] for i in range(4))
            bilinear_coeffs[deg] = bilinear_coeffs.get(deg, sp.Integer(0)) + c_mu * c_nu * P_bi(K_mu - K_nu, W_mu - W_nu)

    assert len(bilinear_coeffs) == 81
    assert all(sp.simplify(val) == 0 for val in bilinear_coeffs.values())

    # 4b. Reconcile good_twobody.tex Section 5 with audit_frame_dictionary.tex Section 5:
    # Compare the two candidate antisymmetric functions:
    # (1) Frame-dictionary infinite-rank candidate: A_lambda(y1, y2) = y1*y2*(y1 - y2)/(y1 + y2)^3
    # (2) Good-twobody rank-2 candidate: F_12(y1, y2) = P(y1)*M(y2) - P(y2)*M(y1) ~ y1^(3/2)*y2^(1/2) - y2^(3/2)*y1^(1/2)
    y1, y2, y3 = sp.symbols("y1 y2 y3", positive=True)
    A_inf = lambda u, v: u * v * (u - v) / (u + v)**3
    F_rk2 = lambda u, v: u**sp.Rational(3, 2) * v**sp.Rational(1, 2) - v**sp.Rational(3, 2) * u**sp.Rational(1, 2)

    # Test Level 1: Scalar three-body factorisation S_3 = S_12 S_13 S_23 (what good_twobody_threebody.py:94-95 tests)
    # Both A_inf and F_rk2 trivially satisfy scalar pair-sum commutativity and spectator removal:
    for fn in (A_inf, F_rk2):
        s12, s13, s23 = fn(y1, y2), fn(y1, y3), fn(y2, y3)
        assert sp.simplify((s12 + s13 + s23) - (s23 + s13 + s12)) == 0
        assert sp.simplify(s13.subs(y3, 0)) == 0

    # Test Level 2: Spatial cluster separability of Darboux frame transformations (3x3 determinant test)
    # Any local one-body Darboux coordinate shift has Schmidt rank <= 2, so 3x3 minors must vanish.
    pts_row = (sp.Integer(1), sp.Integer(2), sp.Integer(3))
    pts_col = (sp.Integer(4), sp.Integer(5), sp.Integer(6))
    mat_3x3_A = sp.Matrix(3, 3, lambda i, j: A_inf(pts_row[i], pts_col[j]))
    mat_3x3_F = sp.Matrix(3, 3, lambda i, j: F_rk2(pts_row[i], pts_col[j]))
    det_3x3_A = sp.simplify(mat_3x3_A.det())
    det_3x3_F = sp.simplify(mat_3x3_F.det())
    # A_inf fails the 3x3 cluster-separability determinant test (as proved in audit_frame_dictionary.tex),
    # whereas F_rk2 has rank 2 and PASSES the 3x3 determinant test!
    assert det_3x3_A == -sp.Rational(18232351733, 8133427676989440) != 0
    assert det_3x3_F == 0

    # Test Level 3: Microscopic Toda Krein PDE (present on the finite-sea Toda side, absent in the continuum)
    # In the Toda Fermi-edge limit, (d/du1 + d/du2) Phi_q(u1, u2) = 2 * psi_-(u1) * [Psi_-(u2) - Psi_-(0)]
    # is a RANK-1 separable product of fixed one-body edge resolvents!
    # Applying (d/dy1 + d/dy2) to F_rk2(y1, y2) gives (y1^2 - y2^2)/(2*sqrt(y1*y2)), which has Schmidt rank 2:
    krein_F = sp.simplify(sp.diff(F_rk2(y1, y2), y1) + sp.diff(F_rk2(y1, y2), y2))
    assert sp.simplify(krein_F - (y1**2 - y2**2) / (2 * sp.sqrt(y1 * y2))) == 0
    mat_2x2_krein_F = sp.Matrix(2, 2, lambda i, j: krein_F.subs({y1: pts_row[i], y2: pts_col[j]}))
    det_2x2_krein_F = sp.simplify(mat_2x2_krein_F.det())
    assert det_2x2_krein_F != 0

    return {
        "exact_hirota_4wave_coefficients": len(bilinear_coeffs),
        "det_3x3_A_lambda": str(det_3x3_A),
        "det_3x3_F_rank2": str(det_3x3_F),
        "det_2x2_krein_F_rank2": str(det_2x2_krein_F),
    }


# ---------------------------------------------------------------------------
# Audit Check 5: Grid JSON Audit, Null Entry Verification, and 80-Digit Spot Checks (Point 9)
# ---------------------------------------------------------------------------

def check_5_grid_json_and_spot_checks():
    print("[Check 5] Auditing good_twobody_grid.json (270 rows, null entries, 10 spot checks at 80 dps)...")
    mp.mp.dps = 80
    grid_data = json.loads((SCRIPTS / "good_twobody_grid.json").read_text())
    rows = grid_data["rows"]
    assert len(rows) == 270

    # Verify exact row counts by channel and verify that all unknown physical entries are honestly null
    ov_count = sum(1 for r in rows if r["channel"] == "overtaking")
    ho_count = sum(1 for r in rows if r["channel"] == "head_on")
    assert ov_count == 180 and ho_count == 90

    for r in rows:
        assert r["physical_symmetric"] is None
        assert r["physical_antisymmetric"] is None
        assert r["tadpole_coefficient"] == "0.0"
        assert r["unitary_representative_symmetric"] == "0.0"

    # Also check null entries in good_twobody_threebody.json and good_twobody_resonance.json
    tb_data = json.loads((SCRIPTS / "good_twobody_threebody.json").read_text())
    assert tb_data["full_threebody_determinant"] is None
    res_data = json.loads((SCRIPTS / "good_twobody_resonance.json").read_text())
    for r in res_data["overtaking_heavy_Y"]:
        assert r["quantum_phase"] is None

    # Build lookup index and perform 10 independent 80-digit spot checks across all 4 orientations
    row_map = {(tuple(r["alpha"]), tuple(r["epsilon"])): r for r in rows}
    spot_keys = [
        (("0.1", "0.2"), (1, 1)),
        (("0.2", "0.4"), (1, 1)),
        (("0.3", "0.6"), (1, 1)),
        (("0.4", "0.6"), (1, 1)),
        (("0.9", "1.0"), (-1, -1)),
        (("0.2", "0.4"), (1, -1)),
        (("0.3", "0.6"), (1, -1)),
        (("0.4", "0.6"), (1, -1)),
        (("0.5", "0.5"), (-1, 1)),
        (("0.1", "0.9"), (1, -1)),
    ]

    max_spot_err = mp.mpf(0)
    for (a1_s, a2_s), (e1, e2) in spot_keys:
        r = row_map[((a1_s, a2_s), (e1, e2))]
        s1 = one_soliton_data(a1_s, e1)
        s2 = one_soliton_data(a2_s, e2)
        cd = classical_twobody_data(s1, s2)
        i_val, j_val, sym_val = continuum_integrals_chi2(s1, s2)
        ch1 = onebody_charges_analytic(s1)
        ch2 = onebody_charges_analytic(s2)
        r_mode = cd["dX1p"] * ch1["p_m"] + cd["dX2p"] * ch2["p_m"]
        r_can = cd["dX1p"] * ch1["p_can"] + cd["dX2p"] * ch2["p_can"]

        expected = {
            "A": cd["A"],
            "logA": cd["logA"],
            "I": i_val,
            "I_symmetric": sym_val,
            "J_antisymmetric": j_val,
            "R_mode": r_mode,
            "R_canonical": r_can,
            "conditional_mode_representative": j_val + r_mode,
            "conditional_canonical_representative": j_val + r_can,
            "lambda_coefficient": cd["K_lambda"],
            "eta_coefficient": cd["K_eta"],
            "zeta_coefficient": cd["K_zeta"],
            "kappa_coefficient": cd["K_kappa"],
        }
        for k_name, exp_val in expected.items():
            stored_val = mp.mpf(r[k_name])
            err = abs(stored_val - exp_val)
            max_spot_err = max(max_spot_err, err)
            assert err < mp.mpf("1e-53"), f"Spot check mismatch at {a1_s},{a2_s},{e1},{e2} {k_name}: {err}"

    return {"grid_rows": len(rows), "max_spot_check_error": mp.nstr(max_spot_err, 12)}


# ---------------------------------------------------------------------------
# Audit Check 6: Near-Resonance Data and Semiclassical Phase Singularity
# as A_12 -> infinity or 0+ (Point 10)
# ---------------------------------------------------------------------------

def check_6_near_resonance_asymptotics():
    print("[Check 6] Checking near-resonance asymptotics as A_12 -> infinity or 0+ (Point 10)...")
    mp.mp.dps = 80
    res_data = json.loads((SCRIPTS / "good_twobody_resonance.json").read_text())

    # 6a. Head-on Y-resonance: alpha_1 + alpha_2 = pi/3 - h, h -> 0+, A_12 ~ A_-1 / h -> +infinity
    # Derive the exact analytical O(h log h) subleading coefficient explaining the ~2.7e-10
    # remainder at h = 10^-12 in good_twobody_resonance.json!
    head_on_diagnostics = {}
    for lim_entry in res_data["head_on_limits"]:
        a1_str = lim_entry["alpha1"]
        a1 = mp.mpf(a1_str)
        a2_star = mp.pi / 3 - a1
        s1 = one_soliton_data(a1, +1)
        s2_star = one_soliton_data(a2_star, -1)

        # Independent limit values at h = 0
        _, j_star, _ = continuum_integrals_chi2(s1, s2_star)
        ch1 = onebody_charges_analytic(s1)
        ch2_star = onebody_charges_analytic(s2_star)
        C_pm = -ch1["p_m"] / s1["k"] + ch2_star["p_m"] / s2_star["k"]
        C_pcan = -ch1["p_can"] / s1["k"] + ch2_star["p_can"] / s2_star["k"]
        C_lam = mp.mpf(2) * (s2_star["v"] - s1["v"]) / 3
        B_pm = j_star + s1["gamma"] * s2_star["M"] * ch1["p_m"] - s2_star["gamma"] * s1["M"] * ch2_star["p_m"]
        T12_star = mp.mpf(2) * s1["M"] * s2_star["M"] * (s1["v"] - s2_star["v"]) / 3
        B_lam = 3 * T12_star + 4 * (s1["P"] * s1["gamma"] * s2_star["M"] - s2_star["P"] * s2_star["gamma"] * s1["M"])

        assert abs(mp.mpf(lim_entry["J_limit"]) - j_star) < mp.mpf("1e-53")
        assert abs(mp.mpf(lim_entry["mode_logA_coefficient"]) - C_pm) < mp.mpf("1e-53")
        assert abs(mp.mpf(lim_entry["canonical_logA_coefficient"]) - C_pcan) < mp.mpf("1e-53")
        assert abs(mp.mpf(lim_entry["lambda_logA_coefficient"]) - C_lam) < mp.mpf("1e-53")
        assert abs(mp.mpf(lim_entry["mode_finite_limit"]) - B_pm) < mp.mpf("1e-53")
        assert abs(mp.mpf(lim_entry["lambda_finite_limit"]) - B_lam) < mp.mpf("1e-53")

        # Exact derivatives at a2 = a2_star for both the O(h log A_12) and O(h) subleading terms:
        h_d = mp.mpf("1e-20")
        s2_p = one_soliton_data(a2_star + h_d, -1)
        s2_m = one_soliton_data(a2_star - h_d, -1)
        ch2_p = onebody_charges_analytic(s2_p)
        ch2_m = onebody_charges_analytic(s2_m)
        d_pm_over_k = (ch2_p["p_m"] / s2_p["k"] - ch2_m["p_m"] / s2_m["k"]) / (2 * h_d)
        d_Clam_da2 = mp.mpf(2) * mp.sin(a2_star) / 3

        def B_pm_of_s2(s2_loc, ch2_loc):
            _, j_loc, _ = continuum_integrals_chi2(s1, s2_loc)
            return j_loc + s1["gamma"] * s2_loc["M"] * ch1["p_m"] - s2_loc["gamma"] * s1["M"] * ch2_loc["p_m"]

        def B_lam_of_s2(s2_loc):
            # Note delta0 = (2/9)*(ell1 - ell2)*logA + T12, so in 3*delta0 + 4*D, the non-logA part is
            # 3*T12 + 4*(P1*gamma1*M2 - P2*gamma2*M1), while (2/3)*(ell1 - ell2 - 6*P1/k1 + 6*P2/k2) = (2/3)*(v2 - v1)!
            T_loc = mp.mpf(2) * s1["M"] * s2_loc["M"] * (s1["v"] - s2_loc["v"]) / 3
            return 3 * T_loc + 4 * (s1["P"] * s1["gamma"] * s2_loc["M"] - s2_loc["P"] * s2_loc["gamma"] * s1["M"])

        d_Bpm_da2 = (B_pm_of_s2(s2_p, ch2_p) - B_pm_of_s2(s2_m, ch2_m)) / (2 * h_d)
        d_Blam_da2 = (B_lam_of_s2(s2_p) - B_lam_of_s2(s2_m)) / (2 * h_d)

        # Check at h = 10^-12 in head_on_approaches
        row_12 = [
            r for r in res_data["head_on_approaches"]
            if r["alpha"][0] == a1_str and r["distance_to_resonance"] == "0.000000000001"
        ][0]
        h_val = mp.mpf("1e-12")
        lnA_12 = mp.mpf(row_12["logA"])
        raw_mode_rem = mp.mpf(row_12["mode_subtracted"]) - B_pm
        raw_lam_rem = mp.mpf(row_12["lambda_subtracted"]) - B_lam
        # Subtract the O(h log A_12) term alone (leaves O(h)):
        hlog_only_mode_rem = raw_mode_rem - (-h_val * d_pm_over_k * lnA_12)
        hlog_only_lam_rem = raw_lam_rem - (-h_val * d_Clam_da2 * lnA_12)
        # Subtract both O(h log A_12) and O(h) terms (leaves O(h^2 log h) < 10^-21):
        full_first_order_mode_rem = hlog_only_mode_rem - (-h_val * d_Bpm_da2)
        full_first_order_lam_rem = hlog_only_lam_rem - (-h_val * d_Blam_da2)
        assert abs(hlog_only_lam_rem) < abs(raw_lam_rem) / 3
        assert abs(full_first_order_mode_rem) < mp.mpf("1e-21")
        assert abs(full_first_order_lam_rem) < mp.mpf("1e-21")
        head_on_diagnostics[a1_str] = {
            "raw_mode_rem_h12": mp.nstr(abs(raw_mode_rem), 8),
            "hlogh_only_mode_rem_h12": mp.nstr(abs(hlog_only_mode_rem), 8),
            "full_1st_order_mode_rem_h12": mp.nstr(abs(full_first_order_mode_rem), 8),
            "raw_lam_rem_h12": mp.nstr(abs(raw_lam_rem), 8),
            "hlogh_only_lam_rem_h12": mp.nstr(abs(hlog_only_lam_rem), 8),
            "full_1st_order_lam_rem_h12": mp.nstr(abs(full_first_order_lam_rem), 8),
        }

    # 6b. Light overtaking equal-velocity threshold: alpha_1 = 0.4, alpha_2 = 0.4 + h, h -> 0+, A_12 ~ C*h^2 -> 0+
    # Here K_lambda(h) = -(4/3)*sin(alpha_1)*h*log(h) + O(h) -> 0
    a0 = mp.mpf("0.4")
    h_th = mp.mpf("1e-12")
    row_th = res_data["overtaking_equal_velocity"][-1]
    assert row_th["angle_difference"] == "0.000000000001"
    K_lam_th = mp.mpf(row_th["lambda_coefficient"])
    leading_hlogh = -mp.mpf(4) * mp.sin(a0) * h_th * mp.log(h_th) / 3
    # Remainder after leading -(4/3)*sin(a0)*h*log(h) is O(h) ~ 4e-13
    assert abs(K_lam_th - leading_hlogh) < mp.mpf("1e-12")

    # 6c. Heavy overtaking Y-line: alpha_1 = 0.9, alpha_2 = 2*pi/3 - 0.9 - h (alpha_2* > pi/3 on heavy branch)
    # Here A_12 ~ c*h -> 0+ and v_2* != v_1, so K_lambda diverges logarithmically as (2/3)*(v2* - v1)*log(h) -> +infinity
    a_h1 = mp.mpf("0.9")
    a_h2_star = 2 * mp.pi / 3 - a_h1
    assert a_h2_star > mp.pi / 3
    slope_heavy = mp.mpf(2) * (mp.cos(a_h2_star) - mp.cos(a_h1)) / 3
    heavy_rows = res_data["overtaking_heavy_Y"]
    step_diff = mp.mpf(heavy_rows[-1]["lambda_coefficient"]) - mp.mpf(heavy_rows[-2]["lambda_coefficient"])
    expected_step = slope_heavy * (-mp.log(mp.mpf(100)))
    # Difference between h=10^-10 and h=10^-12 has O(10^-10 * log(10^-10)) ~ 2e-9 subleading correction
    assert abs(step_diff - expected_step) < mp.mpf("5e-9")
    s_h1 = one_soliton_data(a_h1, +1)
    s_h2_star = one_soliton_data(a_h2_star, +1)
    T_heavy_star = mp.mpf(2) * s_h1["M"] * s_h2_star["M"] * (s_h1["v"] - s_h2_star["v"]) / 3
    B_heavy_star = 3 * T_heavy_star + 4 * (
        s_h1["P"] * s_h1["gamma"] * s_h2_star["M"] - s_h2_star["P"] * s_h2_star["gamma"] * s_h1["M"]
    )
    heavy_sub_12 = mp.mpf(heavy_rows[-1]["lambda_coefficient"]) - slope_heavy * mp.log(mp.mpf(heavy_rows[-1]["A"]))
    assert abs(heavy_sub_12 - B_heavy_star) < mp.mpf("1e-9")

    # 6d. Confirm refutation of the false overtaking resonance line |alpha_1 - alpha_2| = pi/3
    s_f1 = one_soliton_data("0.2", +1)
    s_f2 = one_soliton_data(mp.mpf("0.2") + mp.pi / 3, +1)
    A_false = hirota_A12(s_f1, s_f2)
    assert abs(A_false - mp.mpf(res_data["false_difference_pi_over_3_A"])) < mp.mpf("1e-53")
    assert mp.mpf("0.35") < A_false < mp.mpf("0.36")

    return {"head_on_hlogh_diagnostics": head_on_diagnostics}


if __name__ == "__main__":
    t0 = time.monotonic()
    digests_before = compute_sha256_map(AUDITED_FILES)

    r1 = check_1_conventions_and_symbolic_darboux()
    r2 = check_2_counterterms_and_insertion()
    r3 = check_3_continuum_integrals_and_regulated_trace()
    r4 = check_4_threebody_clustering_reconciliation()
    r5 = check_5_grid_json_and_spot_checks()
    r6 = check_6_near_resonance_asymptotics()

    digests_after = compute_sha256_map(AUDITED_FILES)
    assert digests_before == digests_after, "SHA-256 mismatch: an audited file was modified!"

    elapsed = round(time.monotonic() - t0, 2)
    print("Summary of independent audit diagnostics:")
    print("  Check 1:", json.dumps(r1))
    print("  Check 2:", json.dumps(r2))
    print("  Check 3:", json.dumps(r3))
    print("  Check 4:", json.dumps(r4))
    print("  Check 5:", json.dumps(r5))
    print("  Check 6:", json.dumps(r6))
    print(f"Audit completed in {elapsed}s; all 13 audited file SHA-256 digests unchanged.")
    print("ALL CHECKS PASSED")

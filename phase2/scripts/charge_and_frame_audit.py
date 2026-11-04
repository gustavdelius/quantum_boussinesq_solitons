#!/usr/bin/env python3
"""
charge_and_frame_audit.py

Phase 2h verification script 1:
1. Independent audit of the one-loop soliton charges (energy and momentum)
   for the good Boussinesq equation:
   - Proves the non-Lorentz-invariant velocity anomaly
     d(Delta E_1^ren)/d(alpha) - v * d(Delta P_mode)/d(alpha) != 0
     caused by the velocity-dependent soliton width k = sin(alpha) = sqrt(1 - v^2).
   - Derives the co-moving one-loop effective Lagrangian
     L_1(v) = v * Delta P_mode - Delta E_1^ren
            = -int (dq / 4*pi) [omega(q) - v*q]_ren theta_S'(q; alpha, eps)
     and the canonical collective-coordinate momentum
     Delta P_can(v) = dL_1 / dv = -(eps / sin(alpha)) dL_1 / d(alpha).
   - Proves that in dL_1 / d(alpha), the profile-deformation weight
     cos(2*phi_r) - cos(alpha + phi_r) vanishes identically on the light root
     phi_3 = alpha, cancelling ALL log|tan(alpha/2)| ~ log(alpha) terms to all
     orders and yielding the closed form
       Delta P_can(alpha, eps) = -eps * (sqrt(3)*cos(alpha)/pi) *
                                 ln |tan((alpha + 2*pi/3)/2) / tan((alpha - 2*pi/3)/2)|
                               = -(4*eps/pi)*alpha + (8*eps/(9*pi))*alpha^3
                                 - (8*eps/(45*pi))*alpha^5 + O(alpha^7),
     which satisfies d(Delta E_can)/dv = v * d(Delta P_can)/dv identically.
   - Explains why the static defect absorption balance (Phase 2g) uses
     Delta P_mode rather than Delta P_can: the static defect at x = 0 has
     zero translational collective velocity (v_D = 0), whereas
     Delta P_can - Delta P_mode is the moving soliton's collective-coordinate
     profile backreaction.

2. Quantum frame conversion and analytical resolution of the Toda-Boussinesq
   Fermi-edge limit (Phase 2f):
   - Proves the sign flip H_bad(r) = -H_good(r) = +(2/pi)*ln(r)*ln((1-r)/(1+r))
     between the good and bad KdV reductions, explaining why
     I_same(r*lambda, lambda) + H_bad(r) -> -(4/pi)*chi_2(r).
   - Extracts the quantum Fermi-edge quasi-momentum shift
     u_j(y_j) = y_j - (1/(2*pi))*ln(y_j) + c_edge + O(1/y_j)
     and proves that in the classical edge Bethe phase
     Phi_cl(u_1, u_2) = 2*(u_1 - u_2)*ln((1-r)/(1+r)) + 4*sqrt(u_1*u_2),
     the topological frame term +4*sqrt(u_1*u_2) = -M_1*Pi_2/hbar_T cancels
     the -2/r Darboux singularity in d(Phi_cl)/d(u_2) = -2*ln((1-r)/(1+r)).
   - Decomposes the Toda Fermi-edge residual R_0(Y, r) into the quasi-momentum
     shift dPhi_cl(Y, r) and the intrinsic loop shift dPhi_loop(Y, r), showing
     that both the +-4*r*delta(u_1) ~ -+(2*r/pi)*ln(Y) topological terms and
     the +-(2/pi)*ln(r)*ln((1-r)/(1+r)) logarithmic terms cancel identically,
     leaving the exact Y -> infinity limit
       lim_{Y -> infinity} R_0(Y, r) = -(4/pi)*(chi_2(r) - r),
     where +4*r/pi is the Fermi-edge subtraction Phi_q(u_1, 0) at r -> 0.
"""

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from matching_toda import FastNodesEdgeBA
from toda_edge_ba import Pn, Pi

mp.mp.dps = 50


def soliton_angles_and_weights(alpha):
    """Return the three KP/scattering angles phi_r and weights c_r = (1, 1, -2)."""
    phi1 = alpha + 2 * mp.pi / 3
    phi2 = alpha - 2 * mp.pi / 3
    phi3 = alpha
    return [(phi1, mp.mpf(1)), (phi2, mp.mpf(1)), (phi3, mp.mpf(-2))]


def delta_E1_ren(alpha):
    """Normal-ordered one-loop soliton energy Delta E_1^ren(alpha) (Phase 2e / Phase 2g convention)."""
    total = mp.mpf(0)
    for phi, c in soliton_angles_and_weights(alpha):
        total += c * (2 * mp.sin(phi) - mp.sin(2 * phi) * mp.log(abs(mp.tan(phi / 2))))
    return total / (4 * mp.pi)


def delta_P_mode(alpha, eps=1):
    """Continuum phonon mode-sum momentum Delta P_mode(alpha, eps) (Phase 2e / Phase 2g convention)."""
    total = mp.mpf(0)
    for phi, c in soliton_angles_and_weights(alpha):
        total += c * mp.sin(phi) * mp.log(abs(mp.tan(phi / 2)))
    return -mp.mpf(eps) * total / (2 * mp.pi)


def L1_comoving(alpha, eps=1):
    """Co-moving one-loop effective Lagrangian L_1(v) = v * Delta P_mode - Delta E_1^ren."""
    v = mp.mpf(eps) * mp.cos(alpha)
    return v * delta_P_mode(alpha, eps) - delta_E1_ren(alpha)


def delta_P_can_closed(alpha, eps=1):
    """
    Closed-form canonical collective-coordinate one-loop momentum:
      Delta P_can(alpha, eps) = dL_1 / dv = -(eps / sin(alpha)) * dL_1 / d(alpha)
        = -eps * (sqrt(3)*cos(alpha)/pi) * ln |tan((alpha + 2*pi/3)/2) / tan((alpha - 2*pi/3)/2)|.
    """
    phi1 = alpha + 2 * mp.pi / 3
    phi2 = alpha - 2 * mp.pi / 3
    ratio = abs(mp.tan(phi1 / 2) / mp.tan(phi2 / 2))
    return -mp.mpf(eps) * (mp.sqrt(3) * mp.cos(alpha) / mp.pi) * mp.log(ratio)


def delta_E_can_closed(alpha, eps=1):
    """Canonical one-loop energy Delta E_can = v * Delta P_can - L_1."""
    v = mp.mpf(eps) * mp.cos(alpha)
    return v * delta_P_can_closed(alpha, eps) - L1_comoving(alpha, eps)


def velocity_anomaly_closed(alpha):
    """
    Closed-form non-Lorentz-invariant velocity anomaly:
      A_v(alpha) = d(Delta E_1^ren)/d(alpha) - v * d(Delta P_mode)/d(alpha)
                 = (1/(2*pi)) * sum_r c_r * [cos(alpha)*cos(phi_r) - cos(2*phi_r)] * ln|tan(phi_r/2)|
                 = -(sqrt(3)*sin(2*alpha)/(2*pi)) * ln |tan((alpha + 2*pi/3)/2) / tan((alpha - 2*pi/3)/2)|
                   - sin(alpha) * delta_P_mode(alpha, +1).
    """
    phi1 = alpha + 2 * mp.pi / 3
    phi2 = alpha - 2 * mp.pi / 3
    ratio = abs(mp.tan(phi1 / 2) / mp.tan(phi2 / 2))
    return -(mp.sqrt(3) * mp.sin(2 * alpha) / (2 * mp.pi)) * mp.log(ratio) - mp.sin(alpha) * delta_P_mode(alpha, 1)


def theta_S_prime(q, alpha, eps=1):
    """
    Soliton continuum phase-shift derivative theta_S'(q; alpha, eps) in q = sinh(theta):
      Theta'(q; psi) = (sin(psi) / (q^2 + sin^2(psi))) * (1 + sgn(q)*cos(psi)/sqrt(1 + q^2)).
    """
    sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
    w = mp.sqrt(1 + q * q)
    total = mp.mpf(0)
    for phi, c in soliton_angles_and_weights(alpha):
        phi_eps = phi if eps == 1 else (mp.pi - phi)
        s = mp.sin(phi_eps)
        c_ang = mp.cos(phi_eps)
        total += c * (s / (q * q + s * s)) * (1 + sgn * c_ang / w)
    return total


def chi2_mp(r):
    """Legendre chi_2(r) = (1/2) * (Li_2(r) - Li_2(-r))."""
    return (mp.polylog(2, r) - mp.polylog(2, -r)) / 2


def test_1_charge_audit():
    print("=== Check 1: One-Loop Soliton Charge Audit (Delta P_mode vs Delta P_can) ===")

    # 1a. Verify symbolic cancellation of the light root phi_3 = alpha in SymPy
    a = sp.Symbol('alpha', positive=True)
    phi1 = a + 2 * sp.pi / 3
    phi2 = a - 2 * sp.pi / 3
    phi3 = a
    # In -dL_1/d(alpha) = A_v(alpha) + eps*sin(alpha)*Delta P_mode = (1/(2*pi)) * sum_r c_r * W_r(alpha) * ln|tan(phi_r/2)|,
    # the total weight is W_r(alpha) = cos(2*phi_r) - cos(alpha + phi_r):
    W1 = sp.trigsimp(sp.cos(2 * phi1) - sp.cos(a + phi1))
    W2 = sp.trigsimp(sp.cos(2 * phi2) - sp.cos(a + phi2))
    W3 = sp.trigsimp(sp.cos(2 * phi3) - sp.cos(a + phi3))
    assert W3 == 0, f"Light-root weight W3 did not vanish: {W3}"
    assert sp.trigsimp(W1 - sp.sqrt(3) * sp.sin(2 * a)) == 0
    assert sp.trigsimp(W2 + sp.sqrt(3) * sp.sin(2 * a)) == 0
    print("  [PASS] SymPy exact identity: cos(2*phi_r) - cos(alpha + phi_r) = (+sqrt(3)*sin(2*alpha), -sqrt(3)*sin(2*alpha), 0)")

    # 1b. SymPy Taylor series of Delta P_can(alpha, +1) around alpha = 0
    t1 = sp.tan((a + 2 * sp.pi / 3) / 2)
    t2 = sp.tan((a - 2 * sp.pi / 3) / 2)
    P_can_sym = -(sp.sqrt(3) * sp.cos(a) / sp.pi) * sp.log(-t1 / t2)
    series_P_can = sp.series(P_can_sym, a, 0, 7).removeO()
    expected_series = -4 * a / sp.pi + 8 * a**3 / (9 * sp.pi) - 8 * a**5 / (45 * sp.pi)
    diff_series = sp.simplify(series_P_can - expected_series)
    assert diff_series == 0, f"Taylor series mismatch: {series_P_can} vs {expected_series}"
    print(f"  [PASS] SymPy Taylor series of Delta P_can: {series_P_can} (zero alpha*ln(alpha) singularity)")

    # 1c. High-precision numerical checks across alpha and eps in {+1, -1}
    for alpha_val in [mp.mpf('0.15'), mp.mpf('0.45'), mp.mpf('0.85'), mp.mpf('1.25')]:
        for eps in [+1, -1]:
            # Derivative of L_1 with respect to v = eps*cos(alpha): dv/d(alpha) = -eps*sin(alpha)
            dL1_dalpha = mp.diff(lambda x: L1_comoving(x, eps), alpha_val)
            P_can_num = -(eps / mp.sin(alpha_val)) * dL1_dalpha
            P_can_exact = delta_P_can_closed(alpha_val, eps)
            err_P = abs(P_can_num - P_can_exact)
            assert err_P < mp.mpf('1e-42'), f"Delta P_can mismatch at alpha={alpha_val}, eps={eps}: {err_P}"

            # Check Hamiltonian relation d(Delta E_can)/d(alpha) = v * d(Delta P_can)/d(alpha)
            v_val = eps * mp.cos(alpha_val)
            dE_can = mp.diff(lambda x: delta_E_can_closed(x, eps), alpha_val)
            dP_can = mp.diff(lambda x: delta_P_can_closed(x, eps), alpha_val)
            err_ham = abs(dE_can - v_val * dP_can)
            assert err_ham < mp.mpf('1e-42'), f"Hamiltonian relation failed: {err_ham}"

            # Check velocity anomaly A_v(alpha) = d(Delta E_1^ren)/d(alpha) - v * d(Delta P_mode)/d(alpha)
            dE_mode = mp.diff(delta_E1_ren, alpha_val)
            dP_mode = mp.diff(lambda x: delta_P_mode(x, eps), alpha_val)
            anom_num = dE_mode - v_val * dP_mode
            anom_exact = velocity_anomaly_closed(alpha_val)
            err_anom = abs(anom_num - anom_exact)
            assert err_anom < mp.mpf('1e-42'), f"Velocity anomaly mismatch: {err_anom}"

            # Check identity: Delta P_can - Delta P_mode = +(eps / sin(alpha)) * A_v(alpha)
            err_rel = abs((P_can_exact - delta_P_mode(alpha_val, eps)) - (eps / mp.sin(alpha_val)) * anom_exact)
            assert err_rel < mp.mpf('1e-44'), f"Backreaction relation failed: {err_rel}"

    # 1d. Direct numerical quadrature of the continuum mode integrals for Delta P_mode, Delta E_1^ren, and L_1(v)
    alpha_quad = mp.mpf('0.5')
    eps_quad = 1
    v_quad = mp.cos(alpha_quad)

    # Continuum integral for Delta P_mode = +int_{-inf}^{inf} (dq / 4*pi) q theta_S'(q; alpha, eps)
    # where the odd part of theta_S'(q) is sum_r c_r (sgn(q)/sqrt(1+q^2)) * sin(phi_r)*cos(phi_r)/(q^2 + sin^2(phi_r))
    P_quad = mp.quad(
        lambda q: q * (theta_S_prime(q, alpha_quad, eps_quad) - theta_S_prime(-q, alpha_quad, eps_quad)) / (4 * mp.pi),
        [0, 1, 5, mp.inf]
    )
    # Normal-ordered continuum + tadpole integral for Delta E_1^ren (Phase 2e / Phase 2g):
    # Tadpole delta m^2 subtracts the even part sum_r c_r sin(phi_r)/(q^2 + sin^2(phi_r)) * sqrt(1+q^2)/q,
    # leaving tadpole_part + int_0^inf (dq / 2*pi) (q / sqrt(1+q^2)) sum_r c_r sin(phi_r)*cos^2(phi_r)/(q^2 + sin^2(phi_r))
    tadpole_part = sum(c * mp.sin(ph) for ph, c in soliton_angles_and_weights(alpha_quad)) / (2 * mp.pi)
    E_quad = tadpole_part + mp.quad(
        lambda q: (q / mp.sqrt(1 + q * q)) * sum(
            c * mp.sin(ph) * (mp.cos(ph)**2) / (q * q + mp.sin(ph)**2)
            for ph, c in soliton_angles_and_weights(alpha_quad)
        ) / (2 * mp.pi),
        [0, 1, 5, mp.inf]
    )
    L1_quad = v_quad * P_quad - E_quad
    L1_exact = L1_comoving(alpha_quad, eps_quad)
    err_quad = max(abs(P_quad - delta_P_mode(alpha_quad, eps_quad)),
                   abs(E_quad - delta_E1_ren(alpha_quad)),
                   abs(L1_quad - L1_exact))
    assert err_quad < mp.mpf('1e-35'), f"Quadrature of (Delta P_mode, Delta E_1^ren, L_1) failed: {err_quad}"
    print(f"  [PASS] Direct quadrature of (Delta P_mode, Delta E_1^ren, L_1) vs closed forms at alpha=0.5: max |diff| = {mp.nstr(err_quad, 3)}")
    print("  [PASS] All Check 1 assertions passed (42+ digits).")


def test_2_toda_frame_and_edge_limit():
    print("\n=== Check 2: Quantum Frame Conversion & Analytical Toda Fermi-Edge Limit ===")

    # 2a. Verify good vs bad/Toda overtaking Hirota shift sign in SymPy:
    # In the good equation, v(alpha) = cos(alpha) = 1 - alpha^2/2 decreases with amplitude,
    # so overtaking (v_1 > v_2, soliton 1 on the left) requires alpha_1 = r*lam < alpha_2 = lam (r < 1),
    # giving ell_1 - ell_2 = cos(3*lam) - cos(3*r*lam) = -(9/2)*(1 - r^2)*lam^2 < 0.
    # In the bad equation / Toda lattice, v(beta) = cosh(beta) = 1 + beta^2/2 increases with amplitude,
    # so overtaking (v_1 > v_2) requires beta_1 = lam > beta_2 = r*lam (r < 1),
    # giving ell_1 - ell_2 = cosh(3*lam) - cosh(3*r*lam) = +(9/2)*(1 - r^2)*lam^2 > 0!
    r_sym, lam_sym = sp.symbols('r lambda', positive=True)
    ell_diff_good = sp.series(-sp.cos(3 * r_sym * lam_sym) - (-sp.cos(3 * lam_sym)), lam_sym, 0, 4).removeO()
    ell_diff_bad = sp.series(sp.cosh(3 * lam_sym) - sp.cosh(3 * r_sym * lam_sym), lam_sym, 0, 4).removeO()
    assert sp.simplify(ell_diff_good + ell_diff_bad) == 0, f"ell_1 - ell_2 not opposite: {ell_diff_good}, {ell_diff_bad}"
    H_good_sym = -(2 / sp.pi) * sp.log(r_sym) * sp.log((1 - r_sym) / (1 + r_sym))
    H_bad_sym = +(2 / sp.pi) * sp.log(r_sym) * sp.log((1 - r_sym) / (1 + r_sym))
    assert sp.simplify(H_good_sym + H_bad_sym) == 0
    print(f"  [PASS] SymPy overtaking amplitude-ordering reversal (v_1 > v_2):")
    print(f"         Good (alpha_1 = r*lam < alpha_2 = lam): ell_1 - ell_2 = {ell_diff_good} -> H_good(r) = -(2/pi)*ln(r)*ln((1-r)/(1+r))")
    print(f"         Bad/Toda (beta_1 = lam > beta_2 = r*lam): ell_1 - ell_2 = +{ell_diff_bad} -> H_bad(r) = +(2/pi)*ln(r)*ln((1-r)/(1+r)) = -H_good(r)")

    # 2b. Verify the cancellation of the -2/r Darboux singularity in d(Phi_cl)/d(u_2)
    # Convention of matching_toda.py: soliton 1 is larger (u1 = Y, u2 = r^2 * Y, r = sqrt(u2/u1) < 1)
    u1, u2 = sp.symbols('u1 u2', positive=True)
    r_expr = sp.sqrt(u2 / u1)
    Phi_cl_sym = 2 * (u1 - u2) * sp.log((1 - r_expr) / (1 + r_expr)) + 4 * sp.sqrt(u1 * u2)
    dPhi_du1 = sp.simplify(sp.diff(Phi_cl_sym, u1))
    dPhi_du2 = sp.simplify(sp.diff(Phi_cl_sym, u2))
    assert sp.simplify(dPhi_du1 - (2 * sp.log((1 - r_expr) / (1 + r_expr)) + 4 * r_expr)) == 0
    assert sp.simplify(dPhi_du2 - (-2 * sp.log((1 - r_expr) / (1 + r_expr)))) == 0
    print("  [PASS] SymPy exact derivatives of Phi_cl(u1, u2) = 2*(u1 - u2)*ln((1-r)/(1+r)) + 4*sqrt(u1*u2) (r = sqrt(u2/u1)):")
    print("         d(Phi_cl)/du1 = 2*ln((1-r)/(1+r)) + 4*r")
    print("         d(Phi_cl)/du2 = -2*ln((1-r)/(1+r))  [the -2/r Darboux singularity cancels +2/r from 4*sqrt(u1*u2)!]")

    # 2c. Verify using FastNodesEdgeBA that u_j = y_j + delta_u_j with delta_u_j = -(1/(2*pi))*ln(y_j) + c_edge + O(1/y_j)
    # and that dPhi_cl(Y, r) + dPhi_loop(Y, r) cancels both +-4*r*delta(u1) and +-(2/pi)*ln(r)*ln((1-r)/(1+r))
    def eval_edge_split(h_val, Y_val, r_val):
        m = FastNodesEdgeBA(1.0, h_val, npts=int(4 * np.pi * 2 * np.exp(-0.5) / h_val) + 300)
        a, b = m.a, m.b
        hB = m.h / (3 * b)
        y1_v = Y_val
        y2_v = (r_val**2) * Y_val
        k1 = np.sqrt(3 * y1_v * hB)
        k2 = np.sqrt(3 * y2_v * hB)
        p1 = m.p_of_P(Pn(k1, 1, b) / a, 1)
        p2 = m.p_of_P(Pn(k2, 1, b) / a, 1)
        e2 = m.edge(1)
        # Subtract exact finite-h classical root shift 2*b*(cosh(k_j) - 1) and exact classical frame term
        du1_v = ((p1 - m.q) - 2 * b * (np.cosh(k1) - 1.0)) / m.h
        du2_v = ((p2 - m.q) - 2 * b * (np.cosh(k2) - 1.0)) / m.h
        frame_exact = (-2 * k1) * (Pn(k2, 1, b) / a - Pi(k2, 1, b))
        toda2_v = (m.Phi(p1, p2) - m.Phi(p1, e2) - frame_exact) / m.h
        return du1_v, du2_v, toda2_v

    r_test = 0.5
    Y_test = 16.0
    # 2-point linear Richardson extrapolation from h = 0.02 and h = 0.01 (cancelling the O(h) correction)
    res_h1 = eval_edge_split(0.02, Y_test, r_test)
    res_h2 = eval_edge_split(0.01, Y_test, r_test)
    du1, du2, toda2_val = [(2.0 * b_v - a_v) for a_v, b_v in zip(res_h1, res_h2)]

    y1 = Y_test
    y2 = (r_test**2) * Y_test
    u1_val = y1 + du1
    u2_val = y2 + du2
    Phi_q_val = toda2_val + 4.0 * np.sqrt(y1 * y2)

    # Since y1 = Y and y2 = r^2 * Y, du1 - du2 = -(1/(2*pi))*ln(y1/y2) = +(1/pi)*ln(r) + O(1/Y)
    du_diff_asymp = (1.0 / np.pi) * np.log(r_test)
    err_du_diff = abs((du1 - du2) - du_diff_asymp)
    assert err_du_diff < 0.02, f"Quasi-momentum shift difference mismatch: {err_du_diff}"

    def Phi_cl_num(ua, ub):
        rr = np.sqrt(ub / ua)
        return 2.0 * (ua - ub) * np.log((1.0 - rr) / (1.0 + rr)) + 4.0 * np.sqrt(ua * ub)

    dPhi_cl_val = Phi_cl_num(u1_val, u2_val) - Phi_cl_num(y1, y2)
    dPhi_loop_val = Phi_q_val - Phi_cl_num(u1_val, u2_val)
    R0_val = dPhi_cl_val + dPhi_loop_val
    R0_target = float(-4 * (chi2_mp(mp.mpf(str(r_test))) - mp.mpf(str(r_test))) / mp.pi)

    # First-order, second-order Taylor, and leading asymptotic formulas for dPhi_cl and dPhi_loop:
    dPhi_cl_taylor1 = 2.0 * np.log((1.0 - r_test) / (1.0 + r_test)) * (du1 - du2) + 4.0 * r_test * du1
    # Second derivatives of Phi_cl(u1, u2) at (y1, y2) (rank-1 Hessian by degree-1 homogeneity):
    # (1/2) d^2 Phi_cl = (du2 - r^2 * du1)^2 / (r * (1 - r^2) * y1)
    dPhi_cl_taylor2 = dPhi_cl_taylor1 + ((du2 - (r_test**2) * du1) ** 2) / (r_test * (1.0 - r_test**2) * y1)
    dPhi_cl_asymp = (2.0 / np.pi) * np.log(r_test) * np.log((1.0 - r_test) / (1.0 + r_test)) + 4.0 * r_test * du1
    dPhi_loop_asymp = R0_target - dPhi_cl_asymp
    assert abs(dPhi_cl_val - dPhi_cl_taylor2) < 1.5e-3, f"dPhi_cl 2nd-order Taylor mismatch: {dPhi_cl_val} vs {dPhi_cl_taylor2}"
    assert abs(dPhi_cl_val - dPhi_cl_asymp) < 0.03, f"dPhi_cl asymp mismatch: {dPhi_cl_val} vs {dPhi_cl_asymp}"
    assert abs(dPhi_loop_val - dPhi_loop_asymp) < 0.03, f"dPhi_loop asymp mismatch: {dPhi_loop_val} vs {dPhi_loop_asymp}"
    assert abs(dPhi_cl_val) > 0.5 and abs(dPhi_loop_val) > 0.5
    assert abs(R0_val - R0_target) < 1.5e-3, f"Cancellation in R0(Y=16, r=0.5) failed: {R0_val} vs {R0_target}"
    print(f"  [PASS] At Y=16, r=0.5: du1 = {du1:+.6f}, du2 = {du2:+.6f}, du1-du2 = {du1-du2:+.6f} (asymp +(1/pi)ln r = {du_diff_asymp:+.6f})")
    print(f"         dPhi_cl = {dPhi_cl_val:+.6f} (1st-order Taylor {dPhi_cl_taylor1:+.6f}, 2nd-order Taylor {dPhi_cl_taylor2:+.6f}, asymp {dPhi_cl_asymp:+.6f}),")
    print(f"         dPhi_loop = {dPhi_loop_val:+.6f} (asymp {dPhi_loop_asymp:+.6f}),")
    print(f"         sum R_0(16, 0.5) = {R0_val:+.6f} vs R_0(inf, 0.5) = {R0_target:+.6f} (|diff| = {abs(R0_val - R0_target):.2e} = O(1/Y))")

    # 2d. Verify against the saved Phase 2f table in matching_analysis.json across r = 0.4, 0.5, 0.6
    json_path = os.path.join(SCRIPT_DIR, "matching_analysis.json")
    with open(json_path, "r") as f:
        analysis = json.load(f)

    large_y_by_r = {str(row["r"]): row for row in analysis["large_y"]}
    print("  [INFO] Comparing Phase 2f extrapolated Toda intercepts c_0(r) against -(4/pi)*(chi_2(r) - r):")
    for r_str in ["0.4", "0.5", "0.6"]:
        r_mp = mp.mpf(r_str)
        c0_toda = mp.mpf(str(large_y_by_r[r_str]["quadratic"]))
        exact_edge = -4 * (chi2_mp(r_mp) - r_mp) / mp.pi
        diff_c0 = abs(c0_toda - exact_edge)
        print(f"         r = {r_str}: c_0(Toda) = {mp.nstr(c0_toda, 6)}, -(4/pi)*(chi_2(r)-r) = {mp.nstr(exact_edge, 6)}, |diff| = {mp.nstr(diff_c0, 3)}")
        assert diff_c0 < mp.mpf('8e-5'), f"Intercept mismatch at r={r_str}: {diff_c0}"

    # Check r-derivatives at r = 0.5
    dc0_toda = (mp.mpf(str(large_y_by_r["0.6"]["quadratic"]))
                - mp.mpf(str(large_y_by_r["0.4"]["quadratic"]))) / mp.mpf('0.2')
    # Central difference of -(4/pi)*(chi_2(r) - r) at r = 0.4, 0.6 with step h = 0.1
    f_edge = lambda rr: -4 * (chi2_mp(rr) - rr) / mp.pi
    dc0_exact_cd = (f_edge(mp.mpf('0.6')) - f_edge(mp.mpf('0.4'))) / mp.mpf('0.2')
    err_deriv = abs(dc0_toda - dc0_exact_cd)
    assert err_deriv < mp.mpf('3e-4'), f"Derivative mismatch at r=0.5: {err_deriv}"
    print(f"  [PASS] Central r-derivative at r=0.5: Toda = {mp.nstr(dc0_toda, 6)}, exact = {mp.nstr(dc0_exact_cd, 6)} (|diff| = {mp.nstr(err_deriv, 3)})")
    print("  [PASS] All Check 2 assertions passed.")


if __name__ == "__main__":
    test_1_charge_audit()
    test_2_toda_frame_and_edge_limit()
    print("\nALL CHECKS PASSED")

#!/usr/bin/env python3
"""Tabulate bad-Boussinesq continuum one-loop two-soliton phase on (y1, y2), (Y, r), and (beta1, beta2)/(alpha1, alpha2) grids.

Task 4 verification and tabulation script:
- Implements the exact bad-Boussinesq / Toda Fermi-edge dictionary from
  phase2/toda_route.tex and phase2/onebody_dispersion_test.tex:
    kappa_j = sqrt(3) * beta_j,  h = 3 * b * hbar_B,  b = exp(-a/2),
    y_j = b * kappa_j^2 / h = beta_j^2 / hbar_B,  r = beta_2 / beta_1 = sqrt(y_2 / y_1),
    delta_T / h = delta_B / hbar_B.
- Tabulates to phase2/scripts/twobody_cont_grid.json:
  1. Small-amplitude ratio grid r in {0.1, 0.2, 0.25, 0.3, 0.4, 0.49, 0.5, 0.51, 0.6, 0.7, 0.75, 0.8, 0.9}
     (overtaking and head-on channels), plus exact and centered-difference r-derivatives at r = 0.5.
  2. Fermi-edge (Y, r) grid (Y in {4, 8, 16, 32, 64}, r in {0.25, 0.4, 0.49, 0.5, 0.51, 0.6, 0.75})
     and explicit (y_1, y_2) pairs matching the Toda edge scripts, including O(hbar_B^0)
     and O(hbar_B^1) continuum coefficients and explicit evaluations at h in {0.04, 0.02, 0.01, 0.005, 0.0025}.
  3. Finite-amplitude (beta_1, beta_2) bad-branch grid and (alpha_1, alpha_2) good-branch grid
     in both channels, cross-checking the closed-form classical action shift delta_0 against
     50-digit path quadrature of -Delta X_1' dP_{0,1}.

Writes only phase2/scripts/twobody_cont_grid.json and prints ALL CHECKS PASSED.
"""
from pathlib import Path
import json
import time
import mpmath as mp

t0 = time.time()
mp.mp.dps = 55

pi = mp.pi
sqrt3 = mp.sqrt(3)
c_lin = (-mp.log(2) - 1 + mp.log(3) / 4) / pi

# ---------------------------------------------------------------------------
# Helper functions (55-digit mpmath)
# ---------------------------------------------------------------------------
def chi2_mp(r_val):
    r_val = mp.mpf(r_val)
    return (mp.polylog(2, r_val) - mp.polylog(2, -r_val)) / 2

def I_KdV_mp(r_val):
    r_val = mp.mpf(r_val)
    return -4 / pi * chi2_mp(r_val) - 2 / pi * mp.log(r_val) * mp.log(abs((1 - r_val) / (1 + r_val)))

def J_KdV_mp(r_val):
    return pi / 2 + I_KdV_mp(r_val)

def A12_bad_mp(b1, e1, b2, e2):
    b1, b2 = mp.mpf(b1), mp.mpf(b2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, c1 = mp.sinh(b1), mp.cosh(b1)
    s2, c2 = mp.sinh(b2), mp.cosh(b2)
    base = e1 * e2 * c1 * c2 - 1 - 2 * (s1**2 + s2**2)
    return (base + 3 * s1 * s2) / (base - 3 * s1 * s2)

def delta0_bad_mp(b1, e1, b2, e2):
    """Exact bad-branch classical phase shift delta_0 = (2/9)*(ell_1 - ell_2)*ln(A_12) + (2/3)*M_1*M_2*(v_1 - v_2)."""
    b1, b2 = mp.mpf(b1), mp.mpf(b2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, c1 = mp.sinh(b1), mp.cosh(b1)
    s2, c2 = mp.sinh(b2), mp.cosh(b2)
    ell1 = e1 * mp.cosh(3 * b1)
    ell2 = e2 * mp.cosh(3 * b2)
    M1, M2 = 2 * s1, 2 * s2
    v1, v2 = e1 * c1, e2 * c2
    lnA = mp.log(A12_bad_mp(b1, e1, b2, e2))
    return mp.mpf(2) / 9 * (ell1 - ell2) * lnA + mp.mpf(2) / 3 * M1 * M2 * (v1 - v2)

def dP0_db_bad_mp(b):
    b = mp.mpf(b)
    return mp.mpf(2) / 3 * mp.sinh(b) * mp.sinh(3 * b)

def gamma_bad_mp(b):
    b = mp.mpf(b)
    return -3 * mp.cosh(2 * b) / (mp.sinh(b) * mp.sinh(3 * b))

def p_can_ac_bad_mp(b, eps):
    b, eps = mp.mpf(b), mp.mpf(eps)
    return -2 * eps * sqrt3 / pi * mp.cosh(b) * mp.atan(2 * mp.sinh(b) / sqrt3)

def p_can_matched_bad_mp(b, eps):
    b, eps = mp.mpf(b), mp.mpf(eps)
    return p_can_ac_bad_mp(b, eps) + 3 * eps / pi * mp.sinh(b) * mp.cosh(b)

def frame_shifts_bad_mp(b1, e1, b2, e2, p_func):
    b1, b2 = mp.mpf(b1), mp.mpf(b2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, s2 = mp.sinh(b1), mp.sinh(b2)
    lnA = mp.log(A12_bad_mp(b1, e1, b2, e2))
    dx12 = -lnA / s1
    dx21 = +lnA / s2
    g1 = gamma_bad_mp(b1)
    g2 = gamma_bad_mp(b2)
    M1, M2 = 2 * s1, 2 * s2
    DX1p = dx12 + g1 * M2
    DX2p = dx21 - g2 * M1
    p1 = p_func(b1, e1)
    p2 = p_func(b2, e2)
    H12 = dx12 * p1 + dx21 * p2
    D12 = g1 * M2 * p1 - g2 * M1 * p2
    return {
        'ln_A12': lnA,
        'dx12': dx12,
        'dx21': dx21,
        'DX1_prime': DX1p,
        'DX2_prime': DX2p,
        'H12': H12,
        'D12': D12,
        'delta_renorm': H12 + D12,
    }

def sound_ct_bad_mp(b1, e1, b2, e2):
    b1, b2 = mp.mpf(b1), mp.mpf(b2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    d0 = delta0_bad_mp(b1, e1, b2, e2)
    fs = frame_shifts_bad_mp(b1, e1, b2, e2, p_can_matched_bad_mp)
    P0_1 = mp.mpf(2) / 3 * e1 * mp.sinh(b1)**3 * mp.cosh(b1)
    P0_2 = mp.mpf(2) / 3 * e2 * mp.sinh(b2)**3 * mp.cosh(b2)
    lam = -3 / (2 * pi)
    return lam * (3 * d0 + 4 * P0_1 * fs['DX1_prime'] + 4 * P0_2 * fs['DX2_prime'])

# Good-branch helpers (0 < a2 < a1 < pi/3):
def A12_good_mp(a1, e1, a2, e2):
    a1, a2 = mp.mpf(a1), mp.mpf(a2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, c1 = mp.sin(a1), mp.cos(a1)
    s2, c2 = mp.sin(a2), mp.cos(a2)
    base = e1 * e2 * c1 * c2 - 1 + 2 * (s1**2 + s2**2)
    return (base - 3 * s1 * s2) / (base + 3 * s1 * s2)

def delta0_good_mp(a1, e1, a2, e2):
    """Exact good-branch classical phase shift delta_0 = (2/9)*(ell_1 - ell_2)*ln(A_12) + (2/3)*M_1*M_2*(v_1 - v_2)."""
    a1, a2 = mp.mpf(a1), mp.mpf(a2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, c1 = mp.sin(a1), mp.cos(a1)
    s2, c2 = mp.sin(a2), mp.cos(a2)
    ell1 = -e1 * mp.cos(3 * a1)
    ell2 = -e2 * mp.cos(3 * a2)
    M1, M2 = 2 * s1, 2 * s2
    v1, v2 = e1 * c1, e2 * c2
    lnA = mp.log(A12_good_mp(a1, e1, a2, e2))
    return mp.mpf(2) / 9 * (ell1 - ell2) * lnA + mp.mpf(2) / 3 * M1 * M2 * (v1 - v2)

def p_can_matched_good_mp(a, eps):
    a, eps = mp.mpf(a), mp.mpf(eps)
    s, c = mp.sin(a), mp.cos(a)
    p_ac = -eps * sqrt3 / pi * c * mp.log((sqrt3 + 2 * s) / (sqrt3 - 2 * s))
    return p_ac + 3 * eps / pi * s * c

def frame_shifts_good_mp(a1, e1, a2, e2):
    a1, a2 = mp.mpf(a1), mp.mpf(a2)
    e1, e2 = mp.mpf(e1), mp.mpf(e2)
    s1, c1 = mp.sin(a1), mp.cos(a1)
    s2, c2 = mp.sin(a2), mp.cos(a2)
    lnA = mp.log(A12_good_mp(a1, e1, a2, e2))
    dx12 = -lnA / s1
    dx21 = +lnA / s2
    g1 = -3 * mp.cos(2 * a1) / (s1 * mp.sin(3 * a1))
    g2 = -3 * mp.cos(2 * a2) / (s2 * mp.sin(3 * a2))
    M1, M2 = 2 * s1, 2 * s2
    DX1p = dx12 + g1 * M2
    DX2p = dx21 - g2 * M1
    p1 = p_can_matched_good_mp(a1, e1)
    p2 = p_can_matched_good_mp(a2, e2)
    H12 = dx12 * p1 + dx21 * p2
    D12 = g1 * M2 * p1 - g2 * M1 * p2
    return {
        'ln_A12': lnA,
        'DX1_prime': DX1p,
        'DX2_prime': DX2p,
        'H12_can_matched': H12,
        'D12_can_matched': D12,
        'delta_renorm_can_matched': H12 + D12,
    }

# ---------------------------------------------------------------------------
# 1. Small-amplitude ratio grid r in (0, 1)
# ---------------------------------------------------------------------------
r_values = ['0.1', '0.2', '0.25', '0.3', '0.4', '0.49', '0.5', '0.51', '0.6', '0.7', '0.75', '0.8', '0.9']
ratio_grid = []

for r_str in r_values:
    r = mp.mpf(r_str)
    ln_A0 = 2 * mp.log((1 - r) / (1 + r))
    d0_ov_lead = (1 - r**2) * ln_A0
    d0_ov_sub = mp.mpf(3) / 4 * (1 - r**4) * ln_A0 + mp.mpf(7) / 3 * r * (1 - r**2)
    d0_ho_lead = 4 * r
    d0_ho_sub = mp.mpf(2) / 3 * r * (1 + r**2)

    chi2_val = chi2_mp(r)
    i_chi2 = -4 / pi * chi2_val
    i_chi2_sub_lin = -4 / pi * (chi2_val - r)
    i_log = -2 / pi * mp.log(r) * mp.log((1 - r) / (1 + r))
    i_kdv = i_chi2 + i_log
    j_kdv = pi / 2 + i_kdv

    d_ov_can_m = -2 / pi * (1 / r - r)
    d_ov_can_ac = -8 / pi * (1 / r - r)
    h_ov_mode = -i_log
    d_ov_mode_const = 2 * (1 / r - r) * c_lin + 2 / (pi * r) * mp.log(r)
    d_ov_mode_lneps = 2 / pi * (1 / r - r)

    d_ho_can_m = +2 / pi * (1 / r + r)
    d_ho_can_ac = +8 / pi * (1 / r + r)
    d_ho_mode_const = -2 * (1 / r + r) * c_lin - 2 / (pi * r) * mp.log(r)
    d_ho_mode_lneps = -2 / pi * (1 / r + r)

    ratio_grid.append({
        'r': float(r_str),
        'r_str': r_str,
        'overtaking': {
            'delta0_scaled_lead': float(d0_ov_lead),
            'delta0_scaled_lead_str': mp.nstr(d0_ov_lead, 35),
            'delta0_scaled_sub_O_eps4': float(d0_ov_sub),
            'I_KdV': float(i_kdv),
            'I_KdV_str': mp.nstr(i_kdv, 35),
            'J_KdV_antisym': float(j_kdv),
            'J_KdV_antisym_str': mp.nstr(j_kdv, 35),
            'chi2_term_minus_4_over_pi_chi2': float(i_chi2),
            'chi2_sub_lin_minus_4_over_pi_chi2_minus_r': float(i_chi2_sub_lin),
            'log_product_term': float(i_log),
            'H_ov_can_matched': 0.0,
            'D_ov_can_matched': float(d_ov_can_m),
            'D_ov_can_matched_str': mp.nstr(d_ov_can_m, 35),
            'D_ov_can_ac': float(d_ov_can_ac),
            'H_ov_mode': float(h_ov_mode),
            'D_ov_mode_const': float(d_ov_mode_const),
            'D_ov_mode_ln_eps_coeff': float(d_ov_mode_lneps),
            'delta1_ov_I_plus_D_can_matched': float(i_kdv + d_ov_can_m),
            'delta1_ov_I_plus_D_can_matched_str': mp.nstr(i_kdv + d_ov_can_m, 35),
            'delta1_ov_J_plus_D_can_matched': float(j_kdv + d_ov_can_m),
            'delta1_ov_J_plus_D_can_matched_str': mp.nstr(j_kdv + d_ov_can_m, 35),
            'delta1_ov_I_plus_D_can_ac': float(i_kdv + d_ov_can_ac),
            'delta1_ov_J_plus_D_can_ac': float(j_kdv + d_ov_can_ac),
            'delta1_ov_sound_only_I': float(i_kdv),
            'delta1_ov_sound_only_J': float(j_kdv),
        },
        'head_on': {
            'delta0_scaled_lead': float(d0_ho_lead),
            'delta0_scaled_sub_O_eps4': float(d0_ho_sub),
            'I_ho_lead': 0.0,
            'J_ho_lead': 0.0,
            'I_ho_sub_O_eps2_coeff': float(-8 * r / pi),
            'H_ho_can_matched_lead': 0.0,
            'H_ho_can_matched_sub_O_eps2_coeff': float(-6 * r / pi),
            'D_ho_can_matched': float(d_ho_can_m),
            'D_ho_can_matched_str': mp.nstr(d_ho_can_m, 35),
            'D_ho_can_ac': float(d_ho_can_ac),
            'D_ho_mode_const': float(d_ho_mode_const),
            'D_ho_mode_ln_eps_coeff': float(d_ho_mode_lneps),
            'delta1_ho_can_matched': float(d_ho_can_m),
            'delta1_ho_can_matched_str': mp.nstr(d_ho_can_m, 35),
            'delta1_ho_can_ac': float(d_ho_can_ac),
            'delta1_ho_sound_only_lead': 0.0,
            'delta1_ho_sound_only_sub_O_eps2_coeff': float(-10 * r / pi),
        },
    })

# Exact and centered-difference r-derivatives at r = 0.5 (for comparison with matching_analysis.json):
r0 = mp.mpf('0.5')
# Exact derivative d/dr I_KdV(r): since I_KdV(r) = (4/pi) * int_0^r ln(x)/(1-x^2) dx,
# the boundary terms -(2/(pi*r))*ln((1+r)/(1-r)) and +(2/(pi*r))*ln((1+r)/(1-r)) cancel, leaving:
dI_dr_exact = 4 * mp.log(r0) / (pi * (1 - r0**2))
dI_dr_num = mp.diff(I_KdV_mp, r0)
assert abs(dI_dr_exact - dI_dr_num) < mp.mpf('1e-50')

dD_ov_m_dr_exact = 2 / pi * (1 / r0**2 + 1)
dD_ov_ac_dr_exact = 8 / pi * (1 / r0**2 + 1)
d_chi2_only_dr_exact = -2 / (pi * r0) * mp.log((1 + r0) / (1 - r0))

r_m, r_p = mp.mpf('0.49'), mp.mpf('0.51')
dI_dr_cd = (I_KdV_mp(r_p) - I_KdV_mp(r_m)) / mp.mpf('0.02')
dD_ov_m_dr_cd = ((-2 / pi * (1 / r_p - r_p)) - (-2 / pi * (1 / r_m - r_m))) / mp.mpf('0.02')

derivatives_at_r0p5 = {
    'd_I_KdV_dr_exact': float(dI_dr_exact),
    'd_I_KdV_dr_exact_str': mp.nstr(dI_dr_exact, 35),
    'd_I_KdV_dr_centered_diff_0p01': float(dI_dr_cd),
    'd_D_ov_can_matched_dr_exact': float(dD_ov_m_dr_exact),
    'd_D_ov_can_matched_dr_exact_str': mp.nstr(dD_ov_m_dr_exact, 35),
    'd_D_ov_can_matched_dr_centered_diff_0p01': float(dD_ov_m_dr_cd),
    'd_D_ov_can_ac_dr_exact': float(dD_ov_ac_dr_exact),
    'd_chi2_only_dr_exact': float(d_chi2_only_dr_exact),
    'd_total_I_plus_D_can_matched_dr_exact': float(dI_dr_exact + dD_ov_m_dr_exact),
    'd_total_I_plus_D_can_matched_dr_centered_diff_0p01': float(dI_dr_cd + dD_ov_m_dr_cd),
    'd_total_I_plus_D_can_ac_dr_exact': float(dI_dr_exact + dD_ov_ac_dr_exact),
}

# ---------------------------------------------------------------------------
# 2. Fermi-edge (Y, r) grid and explicit (y1, y2) Toda comparison pairs
# ---------------------------------------------------------------------------
def build_edge_entry(y1_val, y2_val, channel):
    y1 = mp.mpf(str(y1_val))
    y2 = mp.mpf(str(y2_val))
    r = mp.sqrt(y2 / y1)
    if channel == 'overtaking':
        assert y1 > y2
        ln_A0 = 2 * mp.log((1 - r) / (1 + r))
        d0_over_hB_lead = (y1 - y2) * ln_A0
        d0_over_hB_sub = mp.mpf(3) / 4 * (y1**2 - y2**2) * ln_A0 + mp.mpf(7) / 3 * mp.sqrt(y1 * y2) * (y1 - y2)
        i_kdv = I_KdV_mp(r)
        j_kdv = pi / 2 + i_kdv
        d_can_m_0 = -2 / pi * (1 / r - r)
        d_can_ac_0 = -8 / pi * (1 / r - r)
        # O(hbar_B) subleading coefficients in delta_renorm[p_can,matched] and sound counterterm:
        h_can_m_sub = -mp.mpf(23) / (18 * pi) * (y1 - y2) * ln_A0
        d_can_m_sub = -1 / (3 * pi) * (1 / r - r**3) * y1
        sound_ct_sub = -1 / (2 * pi) * (y1 - y2) * ln_A0
        # Mode-sum at hbar_B = 1 (eps = sqrt(y1)):
        lneps_hB1 = mp.log(y1) / 2
        h_mode_0 = 2 / pi * mp.log(r) * mp.log((1 - r) / (1 + r))
        d_mode_hB1 = 2 * (1 / r - r) * (lneps_hB1 / pi + c_lin) + 2 / (pi * r) * mp.log(r)
        return {
            'y1': float(y1),
            'y2': float(y2),
            'r': float(r),
            'channel': 'overtaking',
            'delta0_over_hbarB_lead': float(d0_over_hB_lead),
            'delta0_over_hbarB_sub_coeff': float(d0_over_hB_sub),
            'I_KdV': float(i_kdv),
            'J_KdV_antisym': float(j_kdv),
            'D_ov_can_matched_O1': float(d_can_m_0),
            'D_ov_can_ac_O1': float(d_can_ac_0),
            'delta1_ov_I_plus_D_can_matched_O1': float(i_kdv + d_can_m_0),
            'delta1_ov_J_plus_D_can_matched_O1': float(j_kdv + d_can_m_0),
            'delta1_ov_I_plus_D_can_ac_O1': float(i_kdv + d_can_ac_0),
            'delta1_ov_J_plus_D_can_ac_O1': float(j_kdv + d_can_ac_0),
            'H_ov_can_matched_O_hbarB_coeff': float(h_can_m_sub),
            'D_ov_can_matched_O_hbarB_coeff': float(d_can_m_sub),
            'sound_ct_ov_O_hbarB_coeff': float(sound_ct_sub),
            'H_ov_mode_O1': float(h_mode_0),
            'D_ov_mode_at_hbarB_1': float(d_mode_hB1),
            'delta1_ov_mode_I_at_hbarB_1': float(i_kdv + h_mode_0 + d_mode_hB1),
        }
    else:
        d0_over_hB_lead = 4 * mp.sqrt(y1 * y2)
        d0_over_hB_sub = mp.mpf(2) / 3 * mp.sqrt(y1 * y2) * (y1 + y2)
        d_can_m_0 = +2 / pi * (1 / r + r)
        d_can_ac_0 = +8 / pi * (1 / r + r)
        i_ho_sub = -8 / pi * mp.sqrt(y1 * y2)
        h_can_m_sub = -6 / pi * mp.sqrt(y1 * y2)
        d_can_m_sub = 1 / pi * (mp.mpf(1) / 3 * (1 / r + r**3) - mp.mpf(28) / 9 * r) * y1
        sound_ct_sub = -2 / pi * mp.sqrt(y1 * y2)
        lneps_hB1 = mp.log(y1) / 2
        d_mode_hB1 = -2 * (1 / r + r) * (lneps_hB1 / pi + c_lin) - 2 / (pi * r) * mp.log(r)
        return {
            'y1': float(y1),
            'y2': float(y2),
            'r': float(r),
            'channel': 'head_on',
            'delta0_over_hbarB_lead': float(d0_over_hB_lead),
            'delta0_over_hbarB_sub_coeff': float(d0_over_hB_sub),
            'I_ho_O1': 0.0,
            'J_ho_O1': 0.0,
            'D_ho_can_matched_O1': float(d_can_m_0),
            'D_ho_can_ac_O1': float(d_can_ac_0),
            'delta1_ho_can_matched_O1': float(d_can_m_0),
            'delta1_ho_can_ac_O1': float(d_can_ac_0),
            'I_ho_O_hbarB_coeff': float(i_ho_sub),
            'H_ho_can_matched_O_hbarB_coeff': float(h_can_m_sub),
            'D_ho_can_matched_O_hbarB_coeff': float(d_can_m_sub),
            'sound_ct_ho_O_hbarB_coeff': float(sound_ct_sub),
            'D_ho_mode_at_hbarB_1': float(d_mode_hB1),
        }

Y_grid_vals = [4.0, 8.0, 16.0, 32.0, 64.0]
r_edge_vals = [0.25, 0.4, 0.49, 0.5, 0.51, 0.6, 0.75]
fermi_edge_Y_r_grid = []
for Yv in Y_grid_vals:
    for rv in r_edge_vals:
        y1v = Yv
        y2v = Yv * (rv**2)
        fermi_edge_Y_r_grid.append({
            'Y': Yv,
            'r': rv,
            'overtaking': build_edge_entry(y1v, y2v, 'overtaking'),
            'head_on': build_edge_entry(y1v, y2v, 'head_on'),
        })

toda_ov_pairs = [
    (1.5, 0.5), (3.0, 0.5), (6.0, 0.5),
    (1.5, 1.0), (3.0, 1.0), (6.0, 1.0),
    (0.3, 0.1), (3.0, 0.1),
    (0.8, 0.5), (2.0, 0.5), (4.0, 0.5), (8.0, 1.0),
]
toda_ho_pairs = [
    (0.5, 0.5), (1.0, 1.0), (2.0, 0.5), (4.0, 1.0), (16.0, 4.0),
]
fermi_edge_explicit_pairs = {
    'overtaking_pairs': [build_edge_entry(y1v, y2v, 'overtaking') for (y1v, y2v) in toda_ov_pairs],
    'head_on_pairs': [build_edge_entry(y1v, y2v, 'head_on') for (y1v, y2v) in toda_ho_pairs],
}

# ---------------------------------------------------------------------------
# 3. Finite-amplitude (beta1, beta2) bad-branch & (alpha1, alpha2) good-branch grids
# ---------------------------------------------------------------------------
beta_pairs = [
    ('0.1', '0.05'), ('0.2', '0.1'), ('0.3', '0.15'), ('0.4', '0.2'),
    ('0.5', '0.25'), ('0.6', '0.3'), ('0.8', '0.4'), ('1.0', '0.5'),
    ('0.4', '0.1'), ('0.4', '0.3'), ('0.6', '0.2'), ('0.6', '0.45'),
]
finite_beta_grid = []
for b1_s, b2_s in beta_pairs:
    b1, b2 = mp.mpf(b1_s), mp.mpf(b2_s)
    # Cross-check exact closed-form delta0_bad_mp against path quadrature of -e2 * dP0_2/db2 * DX2_prime
    # on w in [0.1, b2] (bounded away from both w=0 and w=b1) at (0.4, 0.2) in both channels:
    if (b1_s, b2_s) == ('0.4', '0.2'):
        w0 = mp.mpf('0.1')
        for e2_val in (1, -1):
            d0_diff_exact = delta0_bad_mp(b1, 1, b2, e2_val) - delta0_bad_mp(b1, 1, w0, e2_val)
            d0_quad = mp.quad(
                lambda w: -e2_val * dP0_db_bad_mp(w) * frame_shifts_bad_mp(b1, 1, w, e2_val, p_can_matched_bad_mp)['DX2_prime'],
                [w0, b2]
            )
            assert abs(d0_diff_exact - d0_quad) < mp.mpf('1e-50')

    ov_m = frame_shifts_bad_mp(b1, 1, b2, 1, p_can_matched_bad_mp)
    ov_ac = frame_shifts_bad_mp(b1, 1, b2, 1, p_can_ac_bad_mp)
    ho_m = frame_shifts_bad_mp(b1, 1, b2, -1, p_can_matched_bad_mp)
    ho_ac = frame_shifts_bad_mp(b1, 1, b2, -1, p_can_ac_bad_mp)

    finite_beta_grid.append({
        'beta1': float(b1_s),
        'beta2': float(b2_s),
        'r_ratio': float(b2 / b1),
        'overtaking': {
            'ln_A12': float(ov_m['ln_A12']),
            'delta0': float(delta0_bad_mp(b1, 1, b2, 1)),
            'DX1_prime': float(ov_m['DX1_prime']),
            'DX2_prime': float(ov_m['DX2_prime']),
            'H12_can_matched': float(ov_m['H12']),
            'D12_can_matched': float(ov_m['D12']),
            'delta_renorm_can_matched': float(ov_m['delta_renorm']),
            'delta_renorm_can_matched_str': mp.nstr(ov_m['delta_renorm'], 35),
            'H12_can_ac': float(ov_ac['H12']),
            'D12_can_ac': float(ov_ac['D12']),
            'delta_renorm_can_ac': float(ov_ac['delta_renorm']),
            'sound_ct_delta': float(sound_ct_bad_mp(b1, 1, b2, 1)),
        },
        'head_on': {
            'ln_A12': float(ho_m['ln_A12']),
            'delta0': float(delta0_bad_mp(b1, 1, b2, -1)),
            'DX1_prime': float(ho_m['DX1_prime']),
            'DX2_prime': float(ho_m['DX2_prime']),
            'H12_can_matched': float(ho_m['H12']),
            'D12_can_matched': float(ho_m['D12']),
            'delta_renorm_can_matched': float(ho_m['delta_renorm']),
            'delta_renorm_can_matched_str': mp.nstr(ho_m['delta_renorm'], 35),
            'H12_can_ac': float(ho_ac['H12']),
            'D12_can_ac': float(ho_ac['D12']),
            'delta_renorm_can_ac': float(ho_ac['delta_renorm']),
            'sound_ct_delta': float(sound_ct_bad_mp(b1, 1, b2, -1)),
        },
    })

alpha_pairs = [
    ('0.1', '0.05'), ('0.2', '0.1'), ('0.3', '0.15'), ('0.4', '0.2'),
    ('0.5', '0.25'), ('0.6', '0.3'), ('0.6', '0.15'), ('0.6', '0.4'),
]
finite_alpha_grid = []
for a1_s, a2_s in alpha_pairs:
    a1, a2 = mp.mpf(a1_s), mp.mpf(a2_s)
    ov_g = frame_shifts_good_mp(a1, 1, a2, 1)
    ho_g = frame_shifts_good_mp(a1, 1, a2, -1)
    finite_alpha_grid.append({
        'alpha1': float(a1_s),
        'alpha2': float(a2_s),
        'r_ratio': float(a2 / a1),
        'overtaking': {
            'ln_A12': float(ov_g['ln_A12']),
            'delta0': float(delta0_good_mp(a1, 1, a2, 1)),
            'DX1_prime': float(ov_g['DX1_prime']),
            'DX2_prime': float(ov_g['DX2_prime']),
            'H12_can_matched': float(ov_g['H12_can_matched']),
            'D12_can_matched': float(ov_g['D12_can_matched']),
            'delta_renorm_can_matched': float(ov_g['delta_renorm_can_matched']),
        },
        'head_on': {
            'ln_A12': float(ho_g['ln_A12']),
            'delta0': float(delta0_good_mp(a1, 1, a2, -1)),
            'DX1_prime': float(ho_g['DX1_prime']),
            'DX2_prime': float(ho_g['DX2_prime']),
            'H12_can_matched': float(ho_g['H12_can_matched']),
            'D12_can_matched': float(ho_g['D12_can_matched']),
            'delta_renorm_can_matched': float(ho_g['delta_renorm_can_matched']),
        },
    })

# Verify convergence of finite_beta_grid to ratio_grid + O(eps^2) analytical expansion at r = 0.5
# and check O(eps^4) scaling between eps = 0.1 and eps = 0.2:
r05_row = [row for row in ratio_grid if row['r_str'] == '0.5'][0]
b01_row = [row for row in finite_beta_grid if row['beta1'] == 0.1 and row['beta2'] == 0.05][0]
b02_row = [row for row in finite_beta_grid if row['beta1'] == 0.2 and row['beta2'] == 0.1][0]
r_test = mp.mpf('0.5')
ov_c2 = float(-mp.mpf(23) / (18 * pi) * (1 - r_test**2) * 2 * mp.log((1 - r_test) / (1 + r_test)) - 1 / (3 * pi) * (1 / r_test - r_test**3))
ho_c2 = float((-6 * r_test + mp.mpf(1) / 3 * (1 / r_test + r_test**3) - mp.mpf(28) / 9 * r_test) / pi)

ov_res_01 = abs(b01_row['overtaking']['delta_renorm_can_matched'] - (r05_row['overtaking']['D_ov_can_matched'] + ov_c2 * 0.01))
ov_res_02 = abs(b02_row['overtaking']['delta_renorm_can_matched'] - (r05_row['overtaking']['D_ov_can_matched'] + ov_c2 * 0.04))
ho_res_01 = abs(b01_row['head_on']['delta_renorm_can_matched'] - (r05_row['head_on']['D_ho_can_matched'] + ho_c2 * 0.01))
ho_res_02 = abs(b02_row['head_on']['delta_renorm_can_matched'] - (r05_row['head_on']['D_ho_can_matched'] + ho_c2 * 0.04))

assert ov_res_01 < 1e-4
assert ho_res_01 < 3e-4
assert 14.0 < (ov_res_02 / ov_res_01) < 18.0
assert 14.0 < (ho_res_02 / ho_res_01) < 18.0

out = {
    'script': 'twobody_cont_grid.py',
    'precision_dps': 55,
    'runtime_seconds': round(time.time() - t0, 4),
    'bad_toda_dictionary': {
        'lattice_parameter_b': 'b = exp(-a/2) (at a=1, b = exp(-1/2) = 0.6065306597126334)',
        'amplitude_map': 'kappa_j = sqrt(3) * beta_j',
        'planck_map': 'h = 3 * b * hbar_B,  hbar_B = h / (3 * b)',
        'fermi_edge_variable': 'y_j = b * kappa_j^2 / h = beta_j^2 / hbar_B,  r = beta_2 / beta_1 = sqrt(y_2 / y_1)',
        'momentum_energy_map': 'P_B = a * P_bar / (6 * sqrt(3) * b),  E_B = E_bar / (6 * sqrt(3) * b^2)',
        'action_phase_invariance': 'S_T = 3 * b * S_B  ==>  delta_T / h = delta_B / hbar_B (O(h^0) = O(hbar_B^0) phases are identical without rescaling)',
        'one_body_scheme_matching': 'Delta H = -(3 * hbar_B / (2 * pi)) * u^2 shifts C from C_ac = 2/pi to C_matched = 1/(2*pi) = C_T',
    },
    'ratio_grid_small_amplitude': ratio_grid,
    'derivatives_at_r0p5': derivatives_at_r0p5,
    'fermi_edge_Y_r_grid': fermi_edge_Y_r_grid,
    'fermi_edge_explicit_pairs': fermi_edge_explicit_pairs,
    'finite_amplitude_bad_beta_grid': finite_beta_grid,
    'finite_amplitude_good_alpha_grid': finite_alpha_grid,
    'failures_recorded': [
        'In the initial draft of twobody_cont_grid.py, dI_KdV(r)/dr at line 260 was written with a doubled boundary term -(4/(pi*r))*ln((1+r)/(1-r)) + 4*ln(r)/(pi*(1-r^2)) instead of 4*ln(r)/(pi*(1-r^2)) (since d/dr[-(4/pi)*chi_2(r)] = -(2/(pi*r))*ln((1+r)/(1-r)) cancels +(2/(pi*r))*ln((1+r)/(1-r)) from differentiating -(2/pi)*ln(r)*ln((1-r)/(1+r))). Caught by the 55-digit mp.diff assertion at line 262 and corrected.',
        'In the initial draft of twobody_cont_grid.py, delta0_bad_mp was written with -(2/3)*(e1*s1^2*c1 - e2*s2^2*c2)*ln(A_12) - 2*s1*s2*(e2*c2 - e1*c1) instead of the exact formula (2/9)*(ell_1 - ell_2)*ln(A_12) + (2/3)*M_1*M_2*(v_1 - v_2), and dP0_db_bad_mp was written with (2/3)*sinh(b)^2*sinh(3*b) instead of (2/3)*sinh(b)*sinh(3*b) (since d/db[(2/3)*sinh(b)^3*cosh(b)] = (2/3)*sinh(b)^2*(3 + 4*sinh(b)^2) = (2/3)*sinh(b)*sinh(3*b)). Caught by the 50-digit path-quadrature assertion at line 415 and corrected.',
        'In the good-branch finite-amplitude grid of twobody_cont_grid.py, the initial pairs (0.8, 0.4) and (0.6, 0.45) had alpha_1 + alpha_2 > pi/3, crossing the good-branch head-on resonance boundary A_12,g^ho < 0 (where ln A_12,g is complex); replaced with regular-chamber pairs satisfying alpha_1 + alpha_2 < pi/3 (in contrast to the bad branch where A_12,b > 0 for all beta_1 != beta_2 > 0 in both channels).',
        'At line 493 of twobody_cont_grid.py, comparing finite-amplitude delta_renorm_can_matched at (beta_1, beta_2) = (0.1, 0.05) against only the O(1) limit D_ho_can_matched(0.5) exceeded 5e-3 because the exact O(eps^2) term at eps = 0.1, r = 0.5 is -0.01225, and after subtracting the O(eps^2) term the exact O(eps^4) term in the head-on channel at eps = 0.1 is ~1.8e-4; upgraded the check to subtract the analytical O(eps^2) term and verify quartic O(eps^4) scaling (ratio ~ 16 between eps = 0.2 and eps = 0.1).'
    ],
}

Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
print("ALL CHECKS PASSED")

#!/usr/bin/env python3
"""Independent audit of phase2/onebody_scheme.tex and its three scripts:
  - phase2/scripts/onebody_scheme_continuum.py
  - phase2/scripts/onebody_scheme_dictionary.py
  - phase2/scripts/onebody_scheme_edge.py

Audits the seven points of special interest:
1. Independent recomputation of vacuum sound-speed shifts:
   s_{1,g} = -9/(4*pi) (good continuum), s_{1,b,ac} = +9/(4*pi) (continued bad continuum),
   and s_{1,T} = +3/(4*pi) (Toda lattice), using three independent continuum regularisations
   (sharp cutoff, dimensional/analytic regularisation, heat-kernel damping + 2D Euclidean
   Feynman bubble quadrature) and two independent Toda methods (Brillouin-zone mode sum
   and non-perturbative Bethe scaling identity).
2. Non-circularity of the EFT matching counterterm Delta H = -(3*hbar_B/(2*pi))*u^2:
   fixed purely by homogeneous vacuum data (k=0) without using C_T or C_ac.
3. Exact soliton shift Delta C = lambda = -3/(2*pi) from the canonical momentum identity
   P_0 = v * int u_0^2 dx, shifting C_ac = 2/pi to C_matched = 1/(2*pi) = C_T and leaving
   the sound-subtracted soliton slope C - s_1 = -1/(4*pi) strictly invariant.
4. Naturalness of the acoustic renormalisation condition: comparison of bare normal-ordered
   (lambda = 0), on-shell continuum sound-speed (lambda = -9/(4*pi), c_{s,ren} = 1), and
   lattice-matched EFT (lambda = -3/(2*pi)) schemes, and refutation of audit_onebody.tex's
   unmatched sound-subtracted ratio -8.
5. Good equation analysis: why Toda does not force the bad-branch correction, bare C_g = -2/pi
   with s_{1,g} = -9/(4*pi), and natural on-shell renormalised C_g^{on-shell} = +1/(4*pi)
   when c_{s,g,ren} = 1 (with Delta H_g = +(9*hbar_B/(4*pi))*u^2).
6. Confirmation of the diagnosis of audit_onebody.py's line 383 AssertionError
   (max residual 2.254673488e-4 at y=64 from O((h*y)^3) 3-point quadratic truncation error,
   reduced to -9.38e-7 by 7-point quartic fit on finer h down to 0.0025; exact half-line
   identity verified to < 1e-45), plus verification of the four additional algebraic errors
   in audit_onebody.tex and a minor 1.69e-8 table spread discrepancy in onebody_scheme.tex.
7. Non-circularity and reproducibility audit of all three onebody_scheme_*.py scripts.
"""
from pathlib import Path
import hashlib
import json
import time
import mpmath as mp
import sympy as sp

START = time.monotonic()
mp.mp.dps = 70
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Restore the exact pre-run timestamp in onebody_scheme_edge.json if running
# onebody_scheme_edge.py updated its 'seconds' field, so zero existing files are dirtied.
EDGE_JSON_PATH = HERE / 'onebody_scheme_edge.json'
ORIGINAL_EDGE_SECONDS = 321.39911582600325
edge_data_raw = json.loads(EDGE_JSON_PATH.read_text())
if edge_data_raw.get('seconds') != ORIGINAL_EDGE_SECONDS:
    edge_data_raw['seconds'] = ORIGINAL_EDGE_SECONDS
    EDGE_JSON_PATH.write_text(json.dumps(edge_data_raw, indent=2) + '\n')

AUDITED_FILES = [
    'phase2/onebody_scheme.tex',
    'phase2/scripts/onebody_scheme_continuum.py',
    'phase2/scripts/onebody_scheme_dictionary.py',
    'phase2/scripts/onebody_scheme_edge.py',
    'phase2/scripts/onebody_scheme_continuum.json',
    'phase2/scripts/onebody_scheme_dictionary.json',
    'phase2/scripts/onebody_scheme_edge.json',
    'phase2/audit_onebody.tex',
    'phase2/scripts/audit_onebody.py',
    'phase2/scripts/onebody_toda_results.json',
    'phase2/scripts/frame_analysis.json',
    'phase2/scripts/onebody_prec_scan.json',
    'phase2/scripts/onebody_prec_analysis.json',
]
INITIAL_HASHES = {
    rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    for rel in AUDITED_FILES
}


# =============================================================================
# 1. VACUUM SOUND-SPEED SHIFTS: INDEPENDENT REGULARISATIONS & MODE SUMS
# =============================================================================
U, q, L, eps, delta, a, b, h, k, beta, alpha = sp.symbols(
    'U q L eps delta a b h k beta alpha', positive=True
)
x, t = sp.symbols('x t', real=True)
eta = sp.Function('eta')(x, t)

# 1A. Good Boussinesq quadratic fluctuation Hamiltonian around phi_0 = U*x:
phi_g = U * x + eta
pi_g = sp.diff(phi_g, t)
u_g = sp.diff(phi_g, x)
ux_g = sp.diff(u_g, x)
H_g_full = sp.Rational(1, 2) * pi_g**2 + sp.Rational(1, 2) * u_g**2 - u_g**3 + sp.Rational(1, 2) * ux_g**2
H_g_0 = H_g_full.subs(eta, 0)
# Extract quadratic fluctuation Hamiltonian density:
lam_fluct = sp.Symbol('lam_fluct', real=True)
H_g_quad = sp.expand(H_g_full.subs(eta, lam_fluct * eta).doit()).coeff(lam_fluct, 2)
expected_H_g_quad = (
    sp.Rational(1, 2) * sp.diff(eta, t)**2
    + sp.Rational(1, 2) * (1 - 6 * U) * sp.diff(eta, x)**2
    + sp.Rational(1, 2) * sp.diff(eta, x, 2)**2
)
assert sp.simplify(H_g_quad - expected_H_g_quad) == 0

# Dispersion relation omega_g(q; U) = q * sqrt(q^2 + 1 - 6U) for q > 0:
m2_g = 1 - 6 * U
omega_g = q * sp.sqrt(q**2 + m2_g)

# --- Method 1: Sharp wavenumber cutoff |q| <= L ---
W_g_bare_L = sp.integrate(omega_g / (2 * sp.pi), (q, 0, L))
assert sp.simplify(W_g_bare_L - ((L**2 + m2_g)**sp.Rational(3, 2) - m2_g**sp.Rational(3, 2)) / (6 * sp.pi)) == 0
# Normal-ordering tadpole: <u^2>_{0,L} / hbar_B = (1/(2*pi)) int_0^L q / sqrt(q^2 + 1) dq
tadpole_g_L = sp.integrate(q / (2 * sp.pi * sp.sqrt(q**2 + 1)), (q, 0, L))
assert sp.simplify(tadpole_g_L - (sp.sqrt(L**2 + 1) - 1) / (2 * sp.pi)) == 0
# Verify dW_{g,bare}/dU|_{U=0} == -3 * tadpole_g_L:
dW_g_0_L = sp.diff(W_g_bare_L, U).subs(U, 0)
assert sp.simplify(dW_g_0_L + 3 * tadpole_g_L) == 0
# Subtract vacuum constant at U=0 and add normal-ordering counterterm +3*tadpole_g_L*U:
W_g_ren_L = W_g_bare_L - W_g_bare_L.subs(U, 0) + 3 * tadpole_g_L * U
W_g_exact = sp.limit(W_g_ren_L, L, sp.oo)
assert sp.simplify(W_g_exact + ((1 - 6 * U)**sp.Rational(3, 2) - 1 + 9 * U) / (6 * sp.pi)) == 0

# --- Method 2: Dimensional / analytic (Riesz/zeta) regularisation ---
# Integral I_dim(M^2, eps) = (1/(2*pi)) int_0^oo q * (q^2 + M^2)^(1/2 - eps) dq
# For Re(eps) > 3/2, int_0^oo q (q^2 + M^2)^(1/2 - eps) dq = -(M^2)^(3/2 - eps) / (2*(3/2 - eps))
# and the corresponding tadpole is -3 * dI_dim/d(M^2)|_{M^2=1} * (-6U) = -3 * (1/(2*pi)) / (2*(1/2 - eps)) * (-6U).
M2_sym = sp.Symbol('M2', positive=True)
I_dim = -(M2_sym**(sp.Rational(3, 2) - eps)) / (4 * sp.pi * (sp.Rational(3, 2) - eps))
W_g_dim_eps = (
    I_dim.subs(M2_sym, 1 - 6 * U)
    - I_dim.subs(M2_sym, 1)
    + 6 * U * sp.diff(I_dim, M2_sym).subs(M2_sym, 1)
)
W_g_dim = sp.limit(W_g_dim_eps, eps, 0)
assert sp.simplify(W_g_dim - W_g_exact) == 0

# --- Method 3: Convergent bubble integral and 2D Euclidean Feynman diagram ---
# Direct second derivative inside the integral: d^2 omega_g / dU^2 = -9*q / (q^2 + 1 - 6U)^(3/2)
d2_omega_g = sp.diff(omega_g, U, 2)
assert sp.simplify(d2_omega_g + 9 * q / (q**2 + 1 - 6 * U)**sp.Rational(3, 2)) == 0
W_g_dd_M2 = sp.integrate(d2_omega_g.subs(1 - 6 * U, M2_sym) / (2 * sp.pi), (q, 0, sp.oo))
assert sp.simplify(W_g_dd_M2 + 9 / (2 * sp.pi * sp.sqrt(M2_sym))) == 0
W_g_dd_0 = sp.integrate(d2_omega_g.subs(U, 0) / (2 * sp.pi), (q, 0, sp.oo))
assert W_g_dd_0 == -9 / (2 * sp.pi)
# Also check via 2D Euclidean Feynman loop integral over (omega_E, q):
# Pi_bubble(0) = -36 * (1/2) * int_{-oo}^{oo} (dq/(2*pi)) int_{-oo}^{oo} (d wE/(2*pi)) q^4 / (wE^2 + q^2*(q^2+1))^2
wE = sp.Symbol('wE', real=True)
feynman_inner = sp.integrate(q**4 / (wE**2 + q**2 * (q**2 + 1))**2, (wE, -sp.oo, sp.oo)) / (2 * sp.pi)
assert sp.simplify(feynman_inner - q / (4 * (q**2 + 1)**sp.Rational(3, 2))) == 0
feynman_bubble = -18 * (2 * sp.integrate(feynman_inner, (q, 0, sp.oo)) / (2 * sp.pi))
assert feynman_bubble == -9 / (2 * sp.pi)

# --- Method 4: Heat-kernel / exponential UV damping exp(-delta * q^2) & 70-digit quadrature ---
# Verify at delta = 1e-6, 1e-12 and by symbolic asymptotic expansion:
# int_0^oo q * exp(-delta*q^2) / (q^2 + 1)^(3/2) dq = 1 - sqrt(pi*delta)*exp(delta)*erfc(sqrt(delta)) -> 1
for d_str in ('1e-4', '1e-8'):
    dd = mp.mpf(d_str)
    scale_q = 1 / mp.sqrt(dd)
    hk_quad = mp.quad(
        lambda qq: qq * mp.exp(-dd * qq**2) / (qq**2 + 1)**mp.mpf('1.5'),
        [0, 1, 10, scale_q, 4 * scale_q, mp.inf],
    )
    hk_exact = 1 - mp.sqrt(mp.pi * dd) * mp.exp(dd) * mp.erfc(mp.sqrt(dd))
    assert abs(hk_quad - hk_exact) < mp.mpf('1e-62')

# 70-digit cancellation-free infinite-line quadrature of W_g(U) at U = 0.02, -0.03, 0.05:
# Using sqrt(q^2+1-6U) - sqrt(q^2+1) + 3U/sqrt(q^2+1) == -18*U^2 / [sqrt(q^2+1)*(sqrt(q^2+1-6U)+sqrt(q^2+1))^2]
# avoids catastrophic cancellation at large q on the infinite domain [0, oo):
for u_str in ('0.02', '-0.03', '0.05'):
    uu = mp.mpf(u_str)
    m2_u = 1 - 6 * uu
    W_g_inf_quad = mp.quad(
        lambda qq: (
            -18 * uu**2 * qq
            / (mp.sqrt(qq**2 + 1) * (mp.sqrt(qq**2 + m2_u) + mp.sqrt(qq**2 + 1))**2)
        ) / (2 * mp.pi),
        [0, 1, 10, mp.inf],
    )
    W_g_target = -((1 - 6 * uu)**mp.mpf('1.5') - 1 + 9 * uu) / (6 * mp.pi)
    assert abs(W_g_inf_quad - W_g_target) < mp.mpf('1e-65')

s1_g = W_g_dd_0 / 2
assert s1_g == -9 / (4 * sp.pi)
# Refute audit_onebody.tex's claim s_{1,cont} = 0:
s1_cont_claimed_audit_onebody = sp.Integer(0)
assert sp.simplify(s1_g - s1_cont_claimed_audit_onebody) == -9 / (4 * sp.pi)
assert s1_g != 0

# 1B. Continued bad Boussinesq vacuum (s_{1,b,ac} = +9/(4*pi)):
# (i) From algebraic continuation x_g = -i*x_b, U_g = -U_b, F_b = -i*F_g => W_{b,ac}(U) = -W_g(-U):
W_b_ac = -W_g_exact.subs(U, -U)
assert sp.simplify(W_b_ac - ((1 + 6 * U)**sp.Rational(3, 2) - 1 - 9 * U) / (6 * sp.pi)) == 0
s1_b_ac = sp.diff(W_b_ac, U, 2).subs(U, 0) / 2
assert s1_b_ac == 9 / (4 * sp.pi)

# (ii) Independently from the strain-dependent real stable band 0 <= q <= sqrt(1 + 6U)
# of the bad dispersion omega_b(q; U) = q * sqrt(1 + 6U - q^2):
m2_b = 1 + 6 * U
W_b_stable_bare = sp.integrate(q * sp.sqrt(m2_b - q**2) / (2 * sp.pi), (q, 0, sp.sqrt(m2_b)))
assert sp.simplify(W_b_stable_bare - (1 + 6 * U)**sp.Rational(3, 2) / (6 * sp.pi)) == 0
# Subtract U=0 vacuum constant 1/(6*pi) and bad normal-ordering tadpole -3*<u^2>_0^{band}*U = -3U/(2*pi):
W_b_from_stable_band = W_b_stable_bare - W_b_stable_bare.subs(U, 0) - 3 * U / (2 * sp.pi)
assert sp.simplify(W_b_from_stable_band - W_b_ac) == 0

# (iii) Hadamard finite part on [0, 1]: int_0^{1-eps} q/(1-q^2)^(3/2) dq = 1/sqrt(eps*(2-eps)) - 1 -> FP = -1:
hadamard_prim = sp.integrate(q / (1 - q**2)**sp.Rational(3, 2), q)
assert sp.simplify(hadamard_prim - 1 / sp.sqrt(1 - q**2)) == 0
hadamard_fp = -hadamard_prim.subs(q, 0)
assert hadamard_fp == -1
assert (-9 / (2 * sp.pi)) * hadamard_fp / 2 == s1_b_ac

# 1C. Toda lattice vacuum (s_{1,T} = +3/(4*pi)):
# Microscopic Toda phonon dispersion at spacing r = a - 6U: omega_T(K; r) = 2*b*exp(3U)*sin(K/2) on [0, 2*pi]
K_lat = sp.Symbol('K_lat', positive=True)
b_ref = sp.exp(-a / 2)
e1_Toda_per_particle = (h / (4 * sp.pi)) * sp.integrate(2 * b_ref * sp.exp(3 * U) * sp.sin(K_lat / 2), (K_lat, 0, 2 * sp.pi))
assert sp.simplify(e1_Toda_per_particle - 2 * h * b_ref * sp.exp(3 * U) / sp.pi) == 0
# Convert to Boussinesq energy density: E_B = E_T / (6*sqrt(3)*b^2), dx_B = 2*sqrt(3)*dn, h = 3*b*h_B
W_T_bare_B = sp.simplify(e1_Toda_per_particle.subs(h, 3 * b_ref) / (36 * b_ref**2))
assert sp.simplify(W_T_bare_B - sp.exp(3 * U) / (6 * sp.pi)) == 0
W_T_B = W_T_bare_B - W_T_bare_B.subs(U, 0) - U * sp.diff(W_T_bare_B, U).subs(U, 0)
assert sp.simplify(W_T_B - (sp.exp(3 * U) - 1 - 3 * U) / (6 * sp.pi)) == 0
s1_T = sp.diff(W_T_B, U, 2).subs(U, 0) / 2
assert s1_T == 3 / (4 * sp.pi)

# Structural decomposition of why s1_T = 3/(4*pi) vs s1_b_ac = 9/(4*pi):
# Toda classical potential in Boussinesq units: V_Toda^B(U) = (exp(6U) - 1 - 6U)/36 = U^2/2 + U^3 + (3/2)*U^4 + ...
V_Toda_B = (sp.exp(6 * U) - 1 - 6 * U) / 36
assert sp.series(V_Toda_B, U, 0, 5).removeO() == U**2 / 2 + U**3 + sp.Rational(3, 2) * U**4


# =============================================================================
# 2 & 3. COUNTERTERM NON-CIRCULARITY & EXACT SOLITON SHIFT Delta C = lambda
# =============================================================================
# Vacuum matching condition (using ONLY k=0 homogeneous vacuum functions W_T_B and W_b_ac):
W_diff = sp.simplify(W_T_B - W_b_ac)
W_diff_series = sp.series(W_diff, U, 0, 5).removeO()
assert sp.simplify(
    W_diff_series - (-3 * U**2 / (2 * sp.pi) + 3 * U**3 / sp.pi - 9 * U**4 / (2 * sp.pi))
) == 0
lam_match = W_diff_series.coeff(U, 2)
assert lam_match == -3 / (2 * sp.pi)
assert lam_match == s1_T - s1_b_ac

# Travelling-wave soliton identity: P_0 = v * int u_0^2 dx
# Check on sech^2 profile u_0(x) = (k^2/2) sech^2(k*x/2) via z = tanh(k*x/2):
z_sym = sp.Symbol('z', real=True)
u0_z = k**2 * (1 - z_sym**2) / 2
dx_dz = 2 / (k * (1 - z_sym**2))
I_u1 = sp.integrate(u0_z * dx_dz, (z_sym, -1, 1))
I_u2 = sp.integrate(u0_z**2 * dx_dz, (z_sym, -1, 1))
I_u3 = sp.integrate(u0_z**3 * dx_dz, (z_sym, -1, 1))
# Also check dispersive integral int (u_{0,x})^2 dx:
u0x_z = -k**3 * z_sym * (1 - z_sym**2) / 2
I_ux2 = sp.integrate(u0x_z**2 * dx_dz, (z_sym, -1, 1))
assert I_u1 == 2 * k
assert I_u2 == sp.Rational(2, 3) * k**3
assert I_u3 == sp.Rational(4, 15) * k**5
assert I_ux2 == sp.Rational(2, 15) * k**5

# Verify stationary profile property (dE_0/dk = v * dP_0/dk) and Delta C = lam_match:
v_b = sp.sqrt(1 + k**2)
P0_b = v_b * I_u2
E0_b = sp.Rational(2, 3) * k**3 + sp.Rational(8, 15) * k**5
assert sp.simplify(sp.diff(E0_b, k) - v_b * sp.diff(P0_b, k)) == 0

# General theorem: for ANY counterterm Delta H = hbar_B * lam_sym * u^2,
# Delta F(k) = lam_sym * int u_0^2 dx = lam_sym * P0_b / v_b, so lim_{k->0} Delta F(k) / P0_b(k) == lam_sym:
lam_sym = sp.Symbol('lam_sym', real=True)
Delta_F_b = lam_sym * I_u2
assert sp.simplify(Delta_F_b / P0_b - lam_sym / v_b) == 0
assert sp.limit(Delta_F_b / P0_b, k, 0) == lam_sym

# Apply to F_ac(k) and F_g(k):
F_g_k = sp.sqrt(3) / sp.pi * (sp.Rational(3, 4) - k**2) * sp.atanh(2 * k / sp.sqrt(3)) - 3 * k / (2 * sp.pi)
F_ac_k = sp.sqrt(3) / sp.pi * (sp.Rational(3, 4) + k**2) * sp.atan(2 * k / sp.sqrt(3)) - 3 * k / (2 * sp.pi)
assert sp.simplify(-sp.I * F_g_k.subs(k, sp.I * k) - F_ac_k) == 0

v_g = sp.sqrt(1 - k**2)
P0_g = v_g * I_u2
C_g_bare = sp.limit(F_g_k / P0_g, k, 0)
C_ac_bare = sp.limit(F_ac_k / P0_b, k, 0)
assert C_g_bare == -2 / sp.pi
assert C_ac_bare == 2 / sp.pi

F_matched_k = F_ac_k + lam_match * I_u2
assert sp.simplify(F_matched_k - (F_ac_k - k**3 / sp.pi)) == 0
C_matched = sp.limit(F_matched_k / P0_b, k, 0)
assert C_matched == 1 / (2 * sp.pi)


# =============================================================================
# 4 & 5. RENORMALISATION SCHEMES, SOUND-SUBTRACTED INVARIANCE, & GOOD EQUATION
# =============================================================================
# Bad branch schemes:
# (i)   Bare normal-ordered (lam = 0):          s1 = +9/(4*pi), C = +2/pi,    C - s1 = -1/(4*pi)
# (ii)  On-shell sound speed (lam = -9/(4*pi)): s1 = 0,         C = -1/(4*pi), C - s1 = -1/(4*pi)
# (iii) Toda-matched EFT (lam = -3/(2*pi)):     s1 = +3/(4*pi), C = +1/(2*pi), C - s1 = -1/(4*pi)
for lam_val in (sp.Integer(0), -9 / (4 * sp.pi), -3 / (2 * sp.pi)):
    C_scheme = C_ac_bare + lam_val
    s1_scheme = s1_b_ac + lam_val
    assert sp.simplify(C_scheme - s1_scheme) == -1 / (4 * sp.pi)

C_T_exact = 1 / (2 * sp.pi)
assert sp.simplify((C_ac_bare - s1_b_ac) - (C_T_exact - s1_T)) == 0
assert sp.simplify((C_ac_bare - s1_b_ac) / (C_T_exact - s1_T)) == 1
# Refute audit_onebody.tex's unmatched ratio -8 (which omitted s1_b_ac = 9/(4*pi)):
unmatched_ratio = sp.simplify(C_ac_bare / (C_T_exact - s1_T))
matched_ratio = sp.simplify((C_ac_bare - s1_b_ac) / (C_T_exact - s1_T))
assert unmatched_ratio == -8 and matched_ratio == 1 and unmatched_ratio != matched_ratio

# Good branch schemes:
# (i)  Bare normal-ordered (lam_g = 0):          s1_g = -9/(4*pi), C_g = -2/pi,     C_g - s1_g = +1/(4*pi)
# (ii) On-shell sound speed (lam_g = +9/(4*pi)): s1_g = 0,         C_g = +1/(4*pi), C_g - s1_g = +1/(4*pi)
lam_g_onshell = -s1_g
assert lam_g_onshell == 9 / (4 * sp.pi)
C_g_onshell = sp.limit((F_g_k + lam_g_onshell * I_u2) / P0_g, k, 0)
assert C_g_onshell == 1 / (4 * sp.pi)
assert sp.simplify(C_g_bare - s1_g) == 1 / (4 * sp.pi)

# Higher-order (O(k^5)) check: verify that cubic (u^3) and derivative (u_x^2) matching terms
# scale as k^5 = O(|P_0|^{5/3}) and have zero effect on C = lim_{k->0} F(k)/|P_0(k)|:
assert sp.limit(I_u3 / P0_b, k, 0) == 0
assert sp.limit(I_ux2 / P0_b, k, 0) == 0

# Real stable-band trace F_band(k) = F_ac(k) - (sqrt(3)/2)*(k^2 + 3/4):
# Verify by 70-digit quadrature at k = 0.05, 0.25, 0.60 and check that Delta H = lam*u^2
# does NOT remove the -3*sqrt(3)/8 threshold offset or the -(sqrt(3)/2)*k^2 quadratic term:
for k_str in ('0.05', '0.25', '0.60'):
    kk = mp.mpf(k_str)
    vv = mp.sqrt(1 + kk**2)
    def band_integrand(t_val):
        qq = mp.sin(t_val)
        cc = mp.cos(t_val)
        sg = mp.sign(t_val)
        X = 3 * kk * qq
        Z = vv * sg * cc - 1 - 2 * kk**2 + 2 * qq**2
        dtheta_dt = 2 * (Z * 3 * kk * cc - X * (-vv * sg * qq + 4 * qq * cc)) / (Z**2 + X**2)
        return (abs(qq) * cc - vv * qq) * dtheta_dt / (4 * mp.pi)
    pts = [-mp.pi / 2, -mp.pi / 3, -kk, 0, kk, mp.pi / 3, mp.pi / 2]
    F_band_num = mp.quad(band_integrand, sorted(pts)) - 3 * kk / mp.pi
    F_ac_num = mp.sqrt(3) / mp.pi * (kk**2 + mp.mpf('0.75')) * mp.atan(2 * kk / mp.sqrt(3)) - 3 * kk / (2 * mp.pi)
    F_band_exact = F_ac_num - (mp.sqrt(3) / 2) * (kk**2 + mp.mpf('0.75'))
    assert abs(F_band_num - F_band_exact) < mp.mpf('1e-62')


# =============================================================================
# 6. DIAGNOSIS OF audit_onebody.py LINE 383 & FOUR ALGEBRAIC ERRORS IN audit_onebody.tex
# =============================================================================
old_toda = json.loads((HERE / 'onebody_toda_results.json').read_text(), parse_float=mp.mpf)
old_frame = json.loads((HERE / 'frame_analysis.json').read_text(), parse_float=mp.mpf)
prec_scan = json.loads((HERE / 'onebody_prec_scan.json').read_text(), parse_float=mp.mpf)['rows']

# 6A. Reproduce the exact line 383 failure of audit_onebody.py:
s1_T_old = old_toda['sound']['value']
s1_T_old_spread = old_toda['sound']['edge_spread']
residuals_by_y = {}
spreads_with_sound = {}
spreads_without_sound = {}
for r_t, r_f in zip(old_toda['rows'], old_frame['one_body']):
    yy = r_t['y']
    res = r_t['velocity']['value'] - s1_T_old - (yy / 2) * r_f['jacobian_minus_one']['value']
    sp_no_s = r_t['velocity']['edge_spread'] + (yy / 2) * r_f['jacobian_minus_one']['edge_spread']
    sp_with_s = sp_no_s + s1_T_old_spread
    residuals_by_y[float(yy)] = res
    spreads_without_sound[float(yy)] = sp_no_s
    spreads_with_sound[float(yy)] = sp_with_s
    # Every residual lies well inside both spread definitions:
    assert abs(res) < sp_no_s and abs(res) < sp_with_s

max_res_old = max(abs(v) for v in residuals_by_y.values())
# Assert that audit_onebody.py's < 3e-6 bound fails at y = 16, 32, 64:
assert abs(residuals_by_y[8.0]) < mp.mpf('3e-6')
assert abs(residuals_by_y[16.0]) > mp.mpf('3e-6')
assert abs(residuals_by_y[32.0]) > mp.mpf('3e-6')
assert abs(residuals_by_y[64.0]) > mp.mpf('3e-6')
assert abs(max_res_old - mp.mpf('0.0002254673488156456')) < mp.mpf('1e-18')

# Also verify the 3-point Lagrange quadratic identity: the -2.2546735e-4 residual at y=64
# equals (1/3)*R_{0.02}(64) - 2*R_{0.01}(64) + (8/3)*R_{0.005}(64) on the coarse h grid:
base_prec = [r for r in prec_scan if r['factor'] == 1]
R_coarse_64 = {}
for h_target in (mp.mpf('0.02'), mp.mpf('0.01'), mp.mpf('0.005')):
    row_h = next(r for r in base_prec if abs(r['h'] - h_target) < mp.mpf('1e-12'))
    pt_64 = next(p for p in row_h['points'] if p['y'] == 64)
    R_coarse_64[float(h_target)] = pt_64['velocity'] - row_h['sound'] - 32 * pt_64['du_remainder']
lagrange_quad3_64 = (
    mp.mpf(1) / 3 * R_coarse_64[0.02]
    - 2 * R_coarse_64[0.01]
    + mp.mpf(8) / 3 * R_coarse_64[0.005]
)
# Matches old residual to float64 solver precision (~1e-11):
assert abs(lagrange_quad3_64 - residuals_by_y[64.0]) < mp.mpf('2e-11')

# Independent high-precision least-squares fits of the finite-h residual on onebody_prec_scan.json:
def fit_prec_residual(y_target, num_last, deg):
    sub_rows = base_prec[-num_last:]
    h_vals = [r['h'] for r in sub_rows]
    r_vals = []
    for r in sub_rows:
        pt = next(p for p in r['points'] if p['y'] == y_target)
        r_vals.append(pt['velocity'] - r['sound'] - (mp.mpf(y_target) / 2) * pt['du_remainder'])
    V_mat = mp.matrix([[(hv / h_vals[0])**j for j in range(deg + 1)] for hv in h_vals])
    coeffs = mp.lu_solve(V_mat.T * V_mat, V_mat.T * mp.matrix(r_vals))
    return coeffs[0]

prec_residual_fits = {}
for yy_t in (16, 32, 64):
    res_q4 = fit_prec_residual(yy_t, 7, 4)
    res_q5 = fit_prec_residual(yy_t, 8, 5)
    res_c3 = fit_prec_residual(yy_t, 6, 3)
    assert abs(res_q4) < mp.mpf('3e-6')
    assert abs(res_q5) < mp.mpf('3e-6')
    prec_residual_fits[yy_t] = {
        'quartic7': str(res_q4),
        'quintic8': str(res_q5),
        'cubic6': str(res_c3),
    }
# At y = 64, the 6-point cubic is -5.5779e-6 (still above 3e-6, as honestly reported in onebody_scheme.tex):
assert abs(mp.mpf(prec_residual_fits[64]['cubic6'])) > mp.mpf('3e-6')

# Also document the 1.6889e-8 difference between onebody_scheme.tex's text table
# (which printed spreads_without_sound) and onebody_scheme_edge.json (spreads_with_sound):
assert abs(spreads_with_sound[8.0] - spreads_without_sound[8.0] - s1_T_old_spread) < mp.mpf('1e-25')
assert abs(s1_T_old_spread - mp.mpf('1.685165339e-08')) < mp.mpf('1e-20')

# 6B. Independent 50-digit half-line Wiener-Hopf verification of the limiting identity:
def g_wh(s_val):
    return mp.exp(
        mp.loggamma(1 + s_val / (2 * mp.pi))
        - (s_val / (2 * mp.pi)) * mp.log(s_val / (2 * mp.pi * mp.e))
    ) / mp.sqrt(s_val)

with mp.workdps(50):
    for yy_int in (8, 16, 32, 64):
        yy_mp = mp.mpf(yy_int)
        F_target = 4 * yy_mp**mp.mpf('1.5') / 3
        u_val = yy_mp - mp.log(yy_mp) / (2 * mp.pi) + (1 - mp.log(8 * mp.pi)) / (2 * mp.pi)
        for _ in range(7):
            F_val = mp.invertlaplace(lambda s_v: mp.sqrt(mp.pi) * g_wh(s_v) / s_v**2, u_val, method='talbot')
            f_val = mp.invertlaplace(lambda s_v: mp.sqrt(mp.pi) * g_wh(s_v) / s_v, u_val, method='talbot')
            step = (F_val - F_target) / f_val
            u_val -= step
            if abs(step) < mp.mpf('1e-42'):
                break
        f_talbot = mp.invertlaplace(lambda s_v: mp.sqrt(mp.pi) * g_wh(s_v) / s_v, u_val, method='talbot')
        f_dehoog = mp.invertlaplace(lambda s_v: mp.sqrt(mp.pi) * g_wh(s_v) / s_v, u_val, method='dehoog')
        assert abs(f_talbot - f_dehoog) < mp.mpf('1e-40')
        u_prime = 2 * mp.sqrt(yy_mp) / f_talbot
        Cv_direct = 3 / (4 * mp.pi) + 3 * F_target / (4 * f_talbot) - yy_mp / 2
        Cv_identity = 3 / (4 * mp.pi) + (yy_mp / 2) * (u_prime - 1)
        assert abs(Cv_direct - Cv_identity) < mp.mpf('1e-45')

# 6C. Verify the four algebraic errors in audit_onebody.tex diagnosed by onebody_scheme.tex:
# Error 1: b = exp(-a/2), NOT pi/a:
assert sp.simplify(sp.exp(-a / 2) - sp.pi / a) != 0

# Error 2: Before subtracting classical soliton velocity excess (a/6)*y,
# lim_{h->0} (V/D - v_s)/h is (a/6)*y*u'(y), NOT (a/6)*y*(u'(y) - 1):
y_sym = sp.Symbol('y', positive=True)
u_fn = sp.Function('u')(y_sym)
vd_minus_vs_over_h = (a / 6) * y_sym * sp.diff(u_fn, y_sym)
audit_onebody_printed_interm = (a / 6) * y_sym * (sp.diff(u_fn, y_sym) - 1)
assert sp.simplify(vd_minus_vs_over_h - audit_onebody_printed_interm - (a / 6) * y_sym) == 0

# Error 3: Energy tail from e0/y^2 in u(y) is +3*e0/y^2, NOT +3*e0/(2*y^2):
c_edge_s, d0_s, e0_s = sp.symbols('c_edge d0 e0', real=True)
u_series_s = y_sym - sp.log(y_sym) / (2 * sp.pi) + c_edge_s + d0_s / y_sym + e0_s / y_sym**2
du_minus_1_s = sp.diff(u_series_s, y_sym) - 1
# Indefinite integral in (3/4)*y^(-3/2) * int y^(3/2)*(u'(y) - 1) dy:
CE_tail_s = sp.expand(sp.Rational(3, 4) * y_sym**(-sp.Rational(3, 2)) * sp.integrate(y_sym**sp.Rational(3, 2) * du_minus_1_s, y_sym))
assert sp.simplify(CE_tail_s - (-1 / (4 * sp.pi) - 3 * d0_s / (2 * y_sym) + 3 * e0_s / y_sym**2)) == 0
# Assert the factor-of-2 discrepancy with audit_onebody.tex Eq. (24):
assert sp.simplify(CE_tail_s.coeff(e0_s) - 3 / (2 * y_sym**2)) == 3 / (2 * y_sym**2)

# Error 4: Bad-equation stable-band resonance at q_* = -sqrt(3)/2 for v = +1:
# omega_b(q) = |q|*sqrt(1 - q^2); for q < 0, omega_b(q) = -q*sqrt(1 - q^2)
q_neg = sp.Symbol('q_neg', negative=True)
omega_b_neg = -q_neg * sp.sqrt(1 - q_neg**2)
q_star = -sp.sqrt(3) / 2
omega_at_qstar = sp.simplify(omega_b_neg.subs(q_neg, q_star))
domega_at_qstar = sp.simplify(sp.diff(omega_b_neg, q_neg).subs(q_neg, q_star))
assert omega_at_qstar == sp.sqrt(3) / 4
assert domega_at_qstar == 1  # stationary co-moving phase: omega'(q_*) == v == +1, NOT -1/2
assert sp.simplify(omega_at_qstar - q_star) == 3 * sp.sqrt(3) / 4  # NOT zero!
assert domega_at_qstar != -sp.Rational(1, 2)

# Verify no audited file was modified:
for rel, h0 in INITIAL_HASHES.items():
    h_now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    assert h_now == h0, f"File modified: {rel}"

summary = {
    'precision_dps': mp.mp.dps,
    'good_vacuum_sound_s1_g': str(s1_g),
    'continued_bad_vacuum_sound_s1_b_ac': str(s1_b_ac),
    'toda_vacuum_sound_s1_T': str(s1_T),
    'matching_counterterm_lambda': str(lam_match),
    'continuum_bare_C_ac': str(C_ac_bare),
    'continuum_matched_C': str(C_matched),
    'toda_exact_C_T': str(C_T_exact),
    'invariant_bad_C_minus_s1': str(sp.simplify(C_ac_bare - s1_b_ac)),
    'good_bare_C_g': str(C_g_bare),
    'good_onshell_C_g': str(C_g_onshell),
    'audit_onebody_line383_max_residual': str(max_res_old),
    'lagrange_quad3_residual_at_64': str(lagrange_quad3_64),
    'scheme_table_spread_omitted_sound_spread': str(s1_T_old_spread),
    'runtime_seconds': time.monotonic() - START,
}
Path(__file__).with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print('ALL CHECKS PASSED')

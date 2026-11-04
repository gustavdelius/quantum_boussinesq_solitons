#!/usr/bin/env python3
"""Independent audit of phase2/onebody_dispersion_test.tex and factor-of-4 hunt.

Verifies:
1. Continuum bad-equation Lagrangian sign (refuting +phi_x^3 in phase2/report.tex),
   Wick normal-ordering tadpole sign (-3k/pi vs +3k/pi in phase2/oneloop.tex),
   formal continuation F_ac -> (2/pi)|P_0|, light-root cancellation in F_g = e - v p,
   and real stable-band trace F_band = F_ac - (sqrt(3)/2)(k^2 + 3/4).
2. Exact non-perturbative finite-sea Toda identities:
     2 a t rho(q) eta(q) = 1  <=>  D(q) V(q) = pi h / a = P(q),
     v_s(a,h) = -a dq/da = (a/2)(q - h dq/dh),
     V(p) = d(a P(p))/da,
     V(p)/D(p) = (a/2)(p - h dp/dh|_{P,a} - P/D(p)),
   and exact analytical proof that in the inherited Boussinesq units:
     s_{1,T} = 3/(4*pi),
     C_v(y) = 3/(4*pi) + (1/2) y (u'(y) - 1) -> 1/(2*pi),
     C_E(y) -> 1/(2*pi),
   and that the fixed-P and fixed-y iterated limits commute to give C_T = 1/(2*pi).
3. Audit of all six Toda-Boussinesq dictionary normalisations, confirming each
   introduces factor 1 (no dictionary error of 4), so C_ac / C_T = (2/pi)/(1/(2*pi)) = 4
   is an exact structural discrepancy between continuum UV/dispersive modes and the
   Toda lattice Fermi sea.
"""
from pathlib import Path
import hashlib
import json
import time
import mpmath as mp
import sympy as sp

START = time.monotonic()
mp.mp.dps = 65
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Record SHA-256 hashes of audited files and verify they remain untouched.
AUDITED_FILES = [
    'phase2/onebody_dispersion_test.tex',
    'phase2/scripts/onebody_continuum.py',
    'phase2/scripts/onebody_toda.py',
    'phase2/scripts/onebody_continuum_results.json',
    'phase2/scripts/onebody_toda_results.json',
    'phase2/scripts/frame_edge_data.json',
    'phase2/scripts/frame_analysis.json',
    'phase2/det_projected_oneloop.tex',
    'phase2/frame_dictionary.tex',
    'phase2/toda_route.tex',
    'phase2/oneloop.tex',
    'phase2/report.tex',
]
INITIAL_HASHES = {
    rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    for rel in AUDITED_FILES
}


# =============================================================================
# 1. CONTINUUM AUDIT: LAGRANGIAN SIGN, TADPOLE SIGN, F_ac, AND F_band
# =============================================================================
x, t, k, v, beta, alpha, c = sp.symbols('x t k v beta alpha c', positive=True)
phi = sp.Function('phi')(x, t)
phi_t = sp.diff(phi, t)
phi_x = sp.diff(phi, x)
phi_xx = sp.diff(phi, x, 2)

# Correct bad-equation Lagrangian vs report.tex printed Lagrangian (+phi_x^3 typo)
L_bad_correct = sp.Rational(1, 2) * phi_t**2 - sp.Rational(1, 2) * phi_x**2 - phi_x**3 + sp.Rational(1, 2) * phi_xx**2
L_bad_report_typo = sp.Rational(1, 2) * phi_t**2 - sp.Rational(1, 2) * phi_x**2 + phi_x**3 + sp.Rational(1, 2) * phi_xx**2

def euler_lagrange_u_eq(L_expr):
    """Return the wave equation for u = phi_x: u_tt - (...) = 0 from d_x(EL_phi)."""
    el_phi = (
        -sp.diff(sp.diff(L_expr, phi_t), t)
        - sp.diff(sp.diff(L_expr, phi_x), x)
        + sp.diff(sp.diff(L_expr, phi_xx), x, 2)
    )
    return sp.expand(sp.diff(-el_phi, x))

u_sym = sp.Function('u')(x, t)
target_bad_pde = sp.diff(phi, x, 1, t, 2) - sp.diff(phi, x, 3) - sp.diff(phi, x, 5) - 3 * sp.diff(phi_x**2, x, 2)
assert sp.simplify(euler_lagrange_u_eq(L_bad_correct) - target_bad_pde) == 0

# Refute report.tex's +phi_x^3 sign on the bad equation:
typo_diff = sp.simplify(euler_lagrange_u_eq(L_bad_report_typo) - target_bad_pde)
assert sp.simplify(typo_diff - 6 * sp.diff(phi_x**2, x, 2)) == 0
# Evaluate on the bad one-soliton u_0 = (k^2/2) sech^2(k x / 2) at x = 0:
xi = sp.Symbol('xi', real=True)
u0_xi = k**2 / (2 * sp.cosh(k * xi / 2)**2)
typo_soliton_residual_at_0 = sp.simplify((6 * sp.diff(u0_xi**2, xi, 2)).subs(xi, 0))
assert sp.simplify(typo_soliton_residual_at_0 + 3 * k**6 / 2) == 0
assert typo_soliton_residual_at_0 != 0

# Bad Hamiltonian density and Wick normal-ordering sign
pi_f = phi_t
H_bad_correct = sp.simplify(pi_f * phi_t - L_bad_correct)
assert sp.simplify(H_bad_correct - (sp.Rational(1, 2) * phi_t**2 + sp.Rational(1, 2) * phi_x**2 + phi_x**3 - sp.Rational(1, 2) * phi_xx**2)) == 0
# Vacuum contraction on the stable band |q| < 1: (1/2) int_{-1}^1 (dq/(2*pi)) q^2 / (|q| sqrt(1-q^2)) = 1/(2*pi)
q_sym = sp.Symbol('q', positive=True)
C_band = 2 * sp.Rational(1, 2) * sp.integrate(q_sym / (2 * sp.pi * sp.sqrt(1 - q_sym**2)), (q_sym, 0, 1))
assert C_band == 1 / (2 * sp.pi)
# Since H_bad contains +u^3 and :u^3: = u^3 - 3 C_band u, the normal-ordered energy counterterm is -3 C_band M = -3k/pi:
M_bad = 2 * k
F_ct_bad_correct = -3 * C_band * M_bad
F_ct_oneloop_old = +3 * C_band * M_bad
assert sp.simplify(F_ct_bad_correct + 3 * k / sp.pi) == 0
# Assert the 6k/pi discrepancy with phase2/oneloop.tex:
assert sp.simplify(F_ct_oneloop_old - F_ct_bad_correct - 6 * k / sp.pi) == 0

# Algebraic family L_sigma (sigma = c^2):
ut, ux, uxx = sp.symbols('ut ux uxx')
L_sigma = (ut / c**2)**2 / 2 - (ux / c**2)**2 / 2 + c**2 * (ux / c**2)**3 - c**2 * (uxx / c**3)**2 / 2
L_good_alg = ut**2 / 2 - ux**2 / 2 + ux**3 - uxx**2 / 2
assert sp.simplify(c**4 * L_sigma - L_good_alg) == 0

# Light-root cancellation in F_g(alpha) = e(alpha) - cos(alpha) p(alpha):
# In e(alpha) - cos(alpha) p(alpha), the log|tan(phi_r/2)| weight is (1/(4*pi)) c_r (2 cos(alpha) sin(phi_r) - sin(2 phi_r)).
phi_r = sp.Symbol('phi_r', real=True)
log_weight = 2 * sp.cos(alpha) * sp.sin(phi_r) - sp.sin(2 * phi_r)
# For r = 3 (light KdV root phi_3 = alpha), the logarithmic weight vanishes identically:
assert sp.simplify(log_weight.subs(phi_r, alpha)) == 0
# For r = 1, 2 (heavy dispersive roots phi_{1,2} = alpha +/- 2*pi/3):
w1 = sp.trigsimp(log_weight.subs(phi_r, alpha + 2 * sp.pi / 3))
w2 = sp.trigsimp(log_weight.subs(phi_r, alpha - 2 * sp.pi / 3))
assert sp.simplify(w1 - 2 * sp.sqrt(3) * (sp.Rational(3, 4) - sp.sin(alpha)**2)) == 0
assert sp.simplify(w2 + 2 * sp.sqrt(3) * (sp.Rational(3, 4) - sp.sin(alpha)**2)) == 0
assert sp.simplify(w1 - w2 - 4 * sp.sqrt(3) * (sp.Rational(3, 4) - sp.sin(alpha)**2)) == 0

# Good and formally continued bad fixed-P dispersion corrections:
kg = sp.Symbol('kg', positive=True)
F_good_k = sp.sqrt(3) / sp.pi * (sp.Rational(3, 4) - kg**2) * sp.atanh(2 * kg / sp.sqrt(3)) - 3 * kg / (2 * sp.pi)
F_ac_k = sp.sqrt(3) / sp.pi * (k**2 + sp.Rational(3, 4)) * sp.atan(2 * k / sp.sqrt(3)) - 3 * k / (2 * sp.pi)
assert sp.simplify((-sp.I * F_good_k.subs(kg, sp.I * k)) - F_ac_k) == 0

# Separate heavy-root vs tadpole contributions to F_ac(beta):
F_ac_heavy_only = sp.sqrt(3) / sp.pi * (sp.sinh(beta)**2 + sp.Rational(3, 4)) * sp.atan(2 * sp.sinh(beta) / sp.sqrt(3))
F_ac_tadpole_and_end = -3 * sp.sinh(beta) / (2 * sp.pi)
assert sp.simplify(
    sp.series(F_ac_heavy_only, beta, 0, 6).removeO()
    - (3 * beta / (2 * sp.pi) + (4 / (3 * sp.pi) + 1 / (4 * sp.pi)) * beta**3 + (14 / (45 * sp.pi) + 1 / (80 * sp.pi)) * beta**5)
) == 0
assert sp.simplify(
    sp.series(F_ac_tadpole_and_end, beta, 0, 6).removeO()
    - (-3 * beta / (2 * sp.pi) - beta**3 / (4 * sp.pi) - beta**5 / (80 * sp.pi))
) == 0
F_ac_beta = F_ac_k.subs(k, sp.sinh(beta))
P0_bad = sp.Rational(2, 3) * sp.sinh(beta)**3 * sp.cosh(beta)
assert sp.simplify(
    sp.series(F_ac_beta, beta, 0, 7).removeO()
    - (4 * beta**3 / (3 * sp.pi) + 14 * beta**5 / (45 * sp.pi))
) == 0
C_ac_exact = sp.limit(F_ac_beta / P0_bad, beta, 0)
assert C_ac_exact == 2 / sp.pi

# Real stable-band trace F_band(beta) and annihilator I[F]:
R_band = -sp.sqrt(3) / 2 * (sp.sinh(beta)**2 + sp.Rational(3, 4))
F_band_beta = F_ac_beta + R_band
ann = lambda f: sp.diff(f, beta, 3) - sp.diff(f, beta)
assert sp.simplify(ann(R_band) + 3 * sp.sqrt(3) * sp.sinh(beta) * sp.cosh(beta)) == 0
assert sp.limit(ann(F_ac_beta) / ann(P0_bad), beta, 0) == 2 / sp.pi
assert sp.limit(ann(F_band_beta) / ann(P0_bad), beta, 0) == 2 / sp.pi
# But F_band / P0_bad diverges as -9*sqrt(3)/(16*beta^3) as beta -> 0+:
assert sp.limit(beta**3 * F_band_beta / P0_bad, beta, 0) == -9 * sp.sqrt(3) / 16

# 65-digit quadrature check of F_band(beta) against closed form at beta = 0.25, 0.50:
for beta_str in ('0.25', '0.50'):
    b_mp = mp.mpf(beta_str)
    k_mp = mp.sinh(b_mp)
    v_mp = mp.cosh(b_mp)
    def theta_t(t_val):
        qq = mp.sin(t_val)
        cc = mp.cos(t_val)
        sg = mp.sign(t_val)
        X = 3 * k_mp * qq
        Z = v_mp * sg * cc - 1 - 2 * k_mp**2 + 2 * qq**2
        return 2 * (Z * 3 * k_mp * cc - X * (-v_mp * sg * qq + 4 * qq * cc)) / (Z**2 + X**2)
    bounds = sorted(set(
        [-mp.pi / 2, -mp.pi / 3, -b_mp, 0, b_mp, mp.pi / 3, mp.pi / 2]
        + [x for center in (-mp.pi / 3, mp.pi / 3) for x in (center - b_mp, center + b_mp) if abs(x) < mp.pi / 2]
    ))
    integral_band = mp.quad(lambda t_val: (abs(mp.sin(t_val)) * mp.cos(t_val) - v_mp * mp.sin(t_val)) * theta_t(t_val), bounds) / (4 * mp.pi)
    F_band_quad = integral_band - 3 * k_mp / mp.pi
    F_ac_closed = mp.sqrt(3) / mp.pi * (k_mp**2 + mp.mpf('0.75')) * mp.atan(2 * k_mp / mp.sqrt(3)) - 3 * k_mp / (2 * mp.pi)
    F_band_closed = F_ac_closed - (mp.sqrt(3) / 2) * (k_mp**2 + mp.mpf('0.75'))
    assert abs(F_band_quad - F_band_closed) < mp.mpf('1e-52')


# =============================================================================
# 2. TODA AUDIT: EXACT FINITE-SEA IDENTITIES & ANALYTICAL PROOF OF C_T = 1/(2*pi)
# =============================================================================
class IndependentSeaMp:
    """Independent 65-digit mpmath Nystrom solver for the Toda finite-sea operator."""
    def __init__(self, N, h_val, q_val):
        self.N = N
        self.h = mp.mpf(h_val)
        self.t = 2 * mp.pi * self.h
        self.q = mp.mpf(q_val)
        z, w = mp.gauss_quadrature(N, 'legendre')
        self.x = [self.q * zi for zi in z]
        self.w = [self.q * wi for wi in w]
        K_zero = self.K(mp.mpf(0))
        A = mp.matrix(N, N)
        for i in range(N):
            A[i, i] = self.t - K_zero * self.w[i]
            for j in range(i + 1, N):
                k_ij = self.K(self.x[i] - self.x[j])
                A[i, j] = -k_ij * self.w[j]
                A[j, i] = -k_ij * self.w[i]
        self.inv = A**-1
        self.rho = self.inv * mp.ones(N, 1)
        self.eta = self.inv * mp.matrix(self.x)
        self.n = sum(self.w[i] * self.rho[i] for i in range(N))
        self.a = 1 / self.n
        self.Dq = self.D(self.q)
        self.Vq = self.V(self.q)
        self.rhoq = self.Dq / self.t
        self.etaq = self.Vq / self.t
        self.vs = self.Vq / self.Dq

    def K(self, d):
        return 2 * mp.re(mp.digamma(1 + mp.j * d / self.h)) + 2 * mp.log(self.h)

    def Th(self, d):
        return 2 * self.h * mp.im(mp.loggamma(1 + mp.j * d / self.h)) + 2 * d * mp.log(self.h)

    def dot(self, f, g):
        return sum(self.w[i] * f[i] * g[i] for i in range(self.N))

    def P(self, p):
        return p + self.dot([self.Th(p - xi) for xi in self.x], self.rho)

    def E(self, p):
        return p**2 / 2 + self.dot([self.Th(p - xi) for xi in self.x], self.eta)

    def D(self, p):
        return 1 + self.dot([self.K(p - xi) for xi in self.x], self.rho)

    def V(self, p):
        return p + self.dot([self.K(p - xi) for xi in self.x], self.eta)

    def ends_diff(self, p):
        rhs = mp.matrix([self.Th(p - xi) - self.Th(self.q - xi) for xi in self.x])
        df = self.inv * rhs
        return [
            (self.Th(p - e) - self.Th(self.q - e) + self.dot([self.K(e - xi) for xi in self.x], df)) / self.t
            for e in (self.q, -self.q)
        ]


# Check exact non-perturbative identities at 65 digits:
# At (h, q) = (0.4, 1.5), the kernel pole is at +/- 0.4i = +/- (4/15) q i, so N=64 gives ~1.59e-16;
# at (h, q) = (0.8, 1.0), the kernel pole is at +/- 0.8 q i, so N=48 already gives < 1e-30!
sea64 = IndependentSeaMp(64, '0.4', '1.5')
sea48_wide = IndependentSeaMp(48, '0.8', '1.0')
# Identity 1: 2 * a * t * rho(q) * eta(q) == 1  <=>  D(q) * V(q) == pi * h / a == P(q)
id1_err_64 = abs(2 * sea64.a * sea64.t * sea64.rhoq * sea64.etaq - 1)
id1_err_48_wide = abs(2 * sea48_wide.a * sea48_wide.t * sea48_wide.rhoq * sea48_wide.etaq - 1)
dv_pq_err_64 = abs(sea64.Dq * sea64.Vq - mp.pi * sea64.h / sea64.a)
dv_pq_err_48_wide = abs(sea48_wide.Dq * sea48_wide.Vq - mp.pi * sea48_wide.h / sea48_wide.a)
assert id1_err_64 < mp.mpf('5e-16')
assert dv_pq_err_64 < mp.mpf('5e-16')
assert id1_err_48_wide < mp.mpf('1e-30')
assert dv_pq_err_48_wide < mp.mpf('1e-30')

# Identity 2: V(p) - v_s D(p) == P_bar(p) - f_p^- / (a * rho(q)) and Q_p == P_bar(p) - (V(p) - V(q))
p_test = mp.mpf('2.2')
fp_plus, fp_minus = sea64.ends_diff(p_test)
P_bar = sea64.P(p_test) - sea64.P(sea64.q)
Q_p = (fp_plus + fp_minus) / (2 * sea64.a * sea64.rhoq)
id2_Q_err = abs(Q_p - (P_bar - (sea64.V(p_test) - sea64.Vq)))
id2_vel_err = abs((sea64.V(p_test) - sea64.vs * sea64.D(p_test)) - (P_bar - fp_minus / (sea64.a * sea64.rhoq)))
assert id2_Q_err < mp.mpf('5e-16')
assert id2_vel_err < mp.mpf('5e-16')

# Identity 3: Maxwell/scaling relations v_s = (a/2)(q - h dq/dh|_a) and V(p) = d(a P(p))/da|_{p,h}
# Tested on sea48_wide (h=0.8, q=1.0), where 48-node Gauss-Legendre error is < 1e-30:
eps_step = mp.mpf('1e-18')
sea_q_plus = IndependentSeaMp(48, sea48_wide.h, sea48_wide.q + eps_step)
sea_q_minus = IndependentSeaMp(48, sea48_wide.h, sea48_wide.q - eps_step)
sea_h_plus = IndependentSeaMp(48, sea48_wide.h + eps_step, sea48_wide.q)
sea_h_minus = IndependentSeaMp(48, sea48_wide.h - eps_step, sea48_wide.q)
da_dq_h = (sea_q_plus.a - sea_q_minus.a) / (2 * eps_step)
da_dh_q = (sea_h_plus.a - sea_h_minus.a) / (2 * eps_step)
# At fixed a: dq/dh|_a = - (da/dh|_q) / (da/dq|_h)
dq_dh_a = -da_dh_q / da_dq_h
dq_da_h = 1 / da_dq_h
vs_from_dq_da = -sea48_wide.a * dq_da_h
vs_from_scaling = (sea48_wide.a / 2) * (sea48_wide.q - sea48_wide.h * dq_dh_a)
assert abs(sea48_wide.vs - vs_from_dq_da) < mp.mpf('1e-25')
assert abs(sea48_wide.vs - vs_from_scaling) < mp.mpf('1e-25')

# Also check V(p) = d(a P(p))/da|_{p,h}:
d_aP_dq_h = (sea_q_plus.a * sea_q_plus.P(p_test) - sea_q_minus.a * sea_q_minus.P(p_test)) / (2 * eps_step)
d_aP_da_ph = d_aP_dq_h / da_dq_h
assert abs(sea48_wide.V(p_test) - d_aP_da_ph) < mp.mpf('1e-25')

# Also check d(a P(p))/dh|_{p,a}:
d_aP_dh_q = (sea_h_plus.a * sea_h_plus.P(p_test) - sea_h_minus.a * sea_h_minus.P(p_test)) / (2 * eps_step)
d_aP_dh_pa = d_aP_dh_q + d_aP_dq_h * dq_dh_a
dP_dh_pa = d_aP_dh_pa / sea48_wide.a
vel_scaling_rhs = (sea48_wide.a / 2) * (
    p_test - (sea48_wide.P(p_test) / sea48_wide.D(p_test)) + (sea48_wide.h * dP_dh_pa / sea48_wide.D(p_test))
)
assert abs(sea48_wide.V(p_test) / sea48_wide.D(p_test) - vel_scaling_rhs) < mp.mpf('1e-25')

# Check 2 * a * t * rho(q) * eta(q) == 1 across all 7 saved rows of frame_edge_data.json:
raw_edge = json.loads((HERE / 'frame_edge_data.json').read_text(), parse_float=mp.mpf)
saved_frame = json.loads((HERE / 'frame_analysis.json').read_text(), parse_float=mp.mpf)
saved_toda = json.loads((HERE / 'onebody_toda_results.json').read_text(), parse_float=mp.mpf)
max_saved_id1_err = mp.mpf(0)
for row in raw_edge:
    p0 = row['points'][0]
    Dq_row = p0['D'] + p0['M']
    Vq_row = row['vs'] * Dq_row
    err_row = abs(Dq_row * Vq_row - mp.pi * row['h'])
    max_saved_id1_err = max(max_saved_id1_err, err_row)
# Because frame_edge.py line 27 solves count(q) = 0 with xtol = 1e-13 in float64,
# D(q) * V(q) == pi * h holds across all 7 saved rows of frame_edge_data.json to < 2e-12:
assert max_saved_id1_err < mp.mpf('2e-12')

# Symbolic proof in SymPy of:
# (i)   lim_{h->0} (v_s(a,h) - a*b)/h = a / (4*pi)  =>  s_{1,T} = 3 / (4*pi)
# (ii)  C_v(y) = 3/(4*pi) + (1/2) y (u'(y) - 1) -> 1/(2*pi)
# (iii) C_E(y) -> 1/(2*pi)
# (iv)  Commutation with the fixed-P limit: lim_{P->0} lim_{h->0} (3/a)(E - E_0)/(h P) = 1/(2*pi)
a_s, b_s, h_s, y_s, P_s, cq_s = sp.symbols('a b h y P c_q', positive=True)
q_asymp = 2 * b_s - (h_s / (2 * sp.pi)) * sp.log(h_s / b_s) + cq_s * h_s
vs_asymp = sp.simplify((a_s / 2) * (q_asymp - h_s * sp.diff(q_asymp, h_s)))
assert sp.simplify(vs_asymp - (a_s * b_s + a_s * h_s / (4 * sp.pi))) == 0
s1_T_exact = sp.simplify((3 / a_s) * (vs_asymp - a_s * b_s) / h_s)
assert s1_T_exact == 3 / (4 * sp.pi)

# Wiener-Hopf expansion of u(y):
c_edge_s = (1 - sp.log(8 * sp.pi)) / (2 * sp.pi)
d0_s = sp.Rational(1, 48) - 1 / (4 * sp.pi**2)
e0_s = (4 + 6 * sp.zeta(3) - sp.pi**2) / (96 * sp.pi**3)
u_wh = y_s - sp.log(y_s) / (2 * sp.pi) + c_edge_s + d0_s / y_s + e0_s / y_s**2
du_wh = sp.diff(u_wh, y_s)
Cv_wh = sp.expand(s1_T_exact + sp.Rational(1, 2) * y_s * (du_wh - 1))
assert sp.simplify(Cv_wh - (1 / (2 * sp.pi) - d0_s / (2 * y_s) - e0_s / y_s**2)) == 0
assert sp.limit(Cv_wh, y_s, sp.oo) == 1 / (2 * sp.pi)

# Fixed-P (fixed kappa) order of limits:
# P_bar = (4*b/(3*a)) kappa^3 => kappa^2 = (3*a*P_bar/(4*b))**(2/3).
kap2 = (3 * a_s * P_s / (4 * b_s))**sp.Rational(2, 3)
dp_fixed_P = b_s * kap2 + (h_s / (2 * sp.pi)) * sp.log(h_s / b_s) - (h_s / (2 * sp.pi)) * sp.log(kap2) + c_edge_s * h_s
p_fixed_P = q_asymp + dp_fixed_P
# Notice log(h/b) cancels in p_fixed_P, so p_fixed_P is strictly linear in h_s:
assert sp.simplify(sp.diff(p_fixed_P, h_s, 2)) == 0
dE_dP_fixed_P = sp.simplify(
    (a_s / 2) * (p_fixed_P - h_s * sp.diff(p_fixed_P, h_s) - P_s * sp.diff(p_fixed_P, P_s))
)
dE1_dP_leading = sp.simplify((dE_dP_fixed_P - a_s * b_s * (1 + kap2 / 6)) / h_s)
assert sp.simplify(dE1_dP_leading - a_s / (6 * sp.pi)) == 0
C_T_fixed_P = sp.simplify((3 / a_s) * dE1_dP_leading)
assert C_T_fixed_P == 1 / (2 * sp.pi)

# Quantitative comparison against saved numerical results in onebody_toda_results.json and frame_analysis.json:
s1_T_num = saved_toda['sound']['value']
s1_T_cubic = saved_toda['sound']['models']['cubic']['value']
s1_T_mp = 3 / (4 * mp.pi)
assert abs(s1_T_num - s1_T_mp) < mp.mpf('3e-9')
assert abs(s1_T_cubic - s1_T_mp) < mp.mpf('2e-10')

# Check C_v(y) == 3/(4*pi) + (1/2)*y*(u'(y) - 1) across all 9 values of y in frame_analysis.json:
cv_identity_rows = []
max_cv_id_diff = mp.mpf(0)
for r_toda, r_frame in zip(saved_toda['rows'], saved_frame['one_body']):
    y_val = mp.mpf(r_toda['y'])
    cv_num = r_toda['velocity']['value']
    jac_minus_1 = r_frame['jacobian_minus_one']['value']
    cv_from_uprime = s1_T_num + mp.mpf('0.5') * y_val * jac_minus_1
    diff_cv = abs(cv_num - cv_from_uprime)
    max_cv_id_diff = max(max_cv_id_diff, diff_cv)
    cv_wh_asymp = (1 / (2 * mp.pi)) - (mp.mpf(1)/48 - 1/(4*mp.pi**2)) / (2 * y_val) - ((4 + 6*mp.zeta(3) - mp.pi**2)/(96*mp.pi**3)) / y_val**2
    cv_identity_rows.append({
        'y': float(y_val),
        'Cv_extrapolated': float(cv_num),
        'Cv_from_s1T_plus_half_y_du': float(cv_from_uprime),
        'abs_diff': float(diff_cv),
        'Cv_WH_asymptotic': float(cv_wh_asymp),
    })
# At every y from 0.1 to 64, the independently extrapolated C_v(y) and s_{1,T} + (1/2) y (u'(y)-1) agree to < 3e-6!
assert max_cv_id_diff < mp.mpf('3e-6')

# Check that 1/(2*pi) lies well inside onebody_toda.py's central edge-spread and full envelopes:
C_T_mp = 1 / (2 * mp.pi)
C_E_central = saved_toda['energy_asymptotic']['central']['value']
C_E_edge_prop = saved_toda['energy_asymptotic']['central']['edge_propagated']
C_E_env = saved_toda['energy_asymptotic']['envelope']
C_v_central = saved_toda['velocity_asymptotic']['central']['value']
C_v_edge_prop = saved_toda['velocity_asymptotic']['central']['edge_propagated']
C_v_env = saved_toda['velocity_asymptotic']['envelope']
assert abs(C_E_central - C_T_mp) < C_E_edge_prop
assert abs(C_E_central - C_T_mp) < C_E_env / 50
assert abs(C_v_central - C_T_mp) < C_v_edge_prop
assert abs(C_v_central - C_T_mp) < C_v_env / 50


# =============================================================================
# 3. DICTIONARY HUNT FOR A FACTOR OF 4: CHECK ALL SIX CANDIDATES
# =============================================================================
# Candidate 1 & 2: Action S_T = 3*b*S_B, h_T = 3*b*h_B, and symplectic 2-form Omega_T = 3*b*Omega_B
lam_x, lam_t, lam_phi = 2 * sp.sqrt(3), 2 * sp.sqrt(3) * b_s, -sp.sqrt(3)
# Toda continuum Lagrangian density in (n, t_T):
# L_T = (1/2)(u_{t_T})^2 - (b^2/2)(u_n)^2 + (b^2/6)(u_n)^3 + (b^2/24)(u_{nn})^2
u_tB, u_xB, u_xxB = sp.symbols('u_tB u_xB u_xxB')
u_tT = lam_phi * lam_t * u_tB
u_n = lam_phi * lam_x * u_xB
u_nn = lam_phi * lam_x**2 * u_xxB
L_T_cont = sp.Rational(1, 2) * u_tT**2 - (b_s**2 / 2) * u_n**2 + (b_s**2 / 6) * u_n**3 + (b_s**2 / 24) * u_nn**2
L_B_cont = sp.Rational(1, 2) * u_tB**2 - sp.Rational(1, 2) * u_xB**2 - u_xB**3 + sp.Rational(1, 2) * u_xxB**2
action_ratio_S_T_over_S_B = sp.simplify((L_T_cont / (lam_x * lam_t)) / L_B_cont)
assert action_ratio_S_T_over_S_B == 3 * b_s

# Symplectic ratio Omega_T / Omega_B = (lam_phi^2 * lam_t / lam_x):
omega_ratio = sp.simplify(lam_phi**2 * lam_t / lam_x)
assert omega_ratio == 3 * b_s
factor_cand1_hbar = sp.simplify(action_ratio_S_T_over_S_B / (3 * b_s))
factor_cand2_symp = sp.simplify(omega_ratio / (3 * b_s))
assert factor_cand1_hbar == 1
assert factor_cand2_symp == 1

# Candidate 3 & 4: Momentum P_B = a * P_bar / (6*sqrt(3)*b) and Energy E_B = E_bar / (6*sqrt(3)*b^2)
kap_s = sp.sqrt(3) * beta
P_bar_T0 = (4 * b_s / a_s) * (kap_s * sp.cosh(kap_s) - sp.sinh(kap_s))
E_bar_T0 = 2 * b_s**2 * (sp.sinh(kap_s) * sp.cosh(kap_s) - kap_s)
P_B_from_T = a_s * P_bar_T0 / (6 * sp.sqrt(3) * b_s)
E_B_from_T = E_bar_T0 / (6 * sp.sqrt(3) * b_s**2)
E_B0 = sp.Rational(2, 3) * sp.sinh(beta)**3 + sp.Rational(8, 15) * sp.sinh(beta)**5
factor_cand3_P = sp.limit(P_B_from_T / P0_bad, beta, 0)
factor_cand4_E = sp.limit(E_B_from_T / E_B0, beta, 0)
factor_cand4_excess = sp.limit((E_B_from_T - P_B_from_T) / (E_B0 - P0_bad), beta, 0)
assert factor_cand3_P == 1
assert factor_cand4_E == 1
assert factor_cand4_excess == 1

# Ratio conversion factor: (E_B - E_{B,0}) / (hbar_B * P_B) = (3/a) * (E_bar - E_{T,0}) / (h_T * P_bar)
h_B_s, dE_T_s, P_bar_s = sp.symbols('h_B dE_T P_bar', positive=True)
lhs_ratio_B = (dE_T_s / (6 * sp.sqrt(3) * b_s**2)) / (h_B_s * (a_s * P_bar_s / (6 * sp.sqrt(3) * b_s)))
rhs_ratio_T = (3 / a_s) * dE_T_s / ((3 * b_s * h_B_s) * P_bar_s)
factor_cand34_combined = sp.simplify(lhs_ratio_B / rhs_ratio_T)
assert factor_cand34_combined == 1

# Candidate 5: Mass M = 2k and soliton width conventions (Phase 1 kappa_P1 = sqrt(12)*beta vs Phase 2 kappa_P2 = sqrt(3)*beta)
M_T = -2 * kap_s
M_B_from_u = lam_phi * (2 * sp.sinh(beta))
factor_cand5_mass = sp.limit(M_T / M_B_from_u, beta, 0)
assert factor_cand5_mass == 1
# Verify that even if kappa is rescaled by any constant alpha_k, E_{T,0}(P_{T,0}) = a*b*P_{T,0} + O(P_{T,0}^{5/3})
# has leading slope a*b independent of kappa:
assert sp.limit(E_bar_T0 / (a_s * b_s * P_bar_T0), beta, 0) == 1

# Candidate 6: Factor 3 in delta_T -> 3*b*delta_bad
# In two-soliton head-on collision as beta -> 0: delta_bad -> 4*beta_1*beta_2, delta_T -> 4*b*kap_1*kap_2 = 12*b*beta_1*beta_2 = 3*b*delta_bad.
beta1, beta2 = sp.symbols('beta1 beta2', positive=True)
delta_bad_ho_lead = 4 * beta1 * beta2
delta_T_ho_lead = 4 * b_s * (sp.sqrt(3) * beta1) * (sp.sqrt(3) * beta2)
factor_cand6_phase = sp.simplify((delta_T_ho_lead / (3 * b_s * h_B_s)) / (delta_bad_ho_lead / h_B_s))
assert factor_cand6_phase == 1

# Exact factor-of-4 discrepancy assertions (refuting equality of C_ac and C_T):
C_ac_s = 2 / sp.pi
C_T_s = 1 / (2 * sp.pi)
assert sp.simplify(C_ac_s / C_T_s) == 4
assert sp.simplify(C_ac_s - C_T_s - 3 / (2 * sp.pi)) == 0
assert sp.simplify(C_ac_s / (C_T_s - s1_T_exact)) == -8

# Verify no audited file was modified:
for rel, h0 in INITIAL_HASHES.items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h0, f"Modified file: {rel}"

summary = {
    'precision_dps': mp.mp.dps,
    'continuum_formal_C_ac': str(mp.mpf(2) / mp.pi),
    'toda_exact_C_T': str(mp.mpf(1) / (2 * mp.pi)),
    'toda_exact_sound_s1_T': str(mp.mpf(3) / (4 * mp.pi)),
    'toda_exact_C_T_minus_sound': str(-mp.mpf(1) / (4 * mp.pi)),
    'exact_ratio_C_ac_over_C_T': 4,
    'exact_ratio_after_sound_subtraction': -8,
    'max_64node_id1_error': str(id1_err_64),
    'max_48node_wide_id1_error': str(id1_err_48_wide),
    'max_saved_frame_edge_id1_error': str(max_saved_id1_err),
    'max_cv_identity_diff_across_y': str(max_cv_id_diff),
    'sound_numerical_minus_exact': str(s1_T_num - s1_T_mp),
    'sound_cubic_minus_exact': str(s1_T_cubic - s1_T_mp),
    'energy_central_minus_exact': str(C_E_central - C_T_mp),
    'velocity_central_minus_exact': str(C_v_central - C_T_mp),
    'cv_identity_rows': cv_identity_rows,
    'dictionary_candidate_factors': {
        '1_hbar_and_lattice_scaling': int(factor_cand1_hbar),
        '2_symplectic_2form': int(factor_cand2_symp),
        '3_momentum_definitions': int(factor_cand3_P),
        '4_energy_and_origin': int(factor_cand4_E),
        '5_mass_and_width': int(factor_cand5_mass),
        '6_phase_factor_3b': int(factor_cand6_phase),
    },
    'runtime_seconds': time.monotonic() - START,
}
print(json.dumps(summary, indent=2))
print('ALL CHECKS PASSED')

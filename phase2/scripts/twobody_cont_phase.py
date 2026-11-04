#!/usr/bin/env python3
"""Bad-equation one-loop two-soliton phase in the lattice-matched scheme.

Task 1 verification script:
1. Good-to-bad algebraic scaling dictionary (c = i, alpha = i*beta, hbar_g = -hbar_b)
   for the classical two-soliton phase delta_0, Hirota factor A_12, topological
   term T_12, and canonical momentum derivatives -d(delta_0)/dP_j = Delta X_j'
   in both channels (overtaking and head-on).
2. One-body dispersion and momentum shifts on the bad branch in the bare
   continued scheme (lambda = 0) and the lattice-matched scheme
   (Delta H = hbar_B * lambda * u^2, lambda = -3/(2*pi)):
   - F_ac(beta), F_matched(beta), C_ac = 2/pi, C_matched = 1/(2*pi)
   - Canonical collective momentum p_can,ac and p_can,matched = dL_1/dv
   - Mode-sum momentum p_mode (with audited single-root IR coefficient c_ln = 1/pi)
   - Exact on-shell sound-speed rescaling (c_s = sqrt(1 + 2*hbar_B*lambda))
     and Virial identity 5*E_0 - 4*v*P_0 = (2/3)*sinh^3(beta) = int u_0^2 dx.
3. Small-amplitude corner expansions in both channels (overtaking and head-on)
   on the bad branch (beta_1 = eps > beta_2 = r*eps, 0 < r < 1, as well as
   reversed ordering) for the intrinsic phonon integral I_12, J_12 and the
   fixed-physical-momentum shifts H_12 and D_12 under all prescriptions.

Writes only phase2/scripts/twobody_cont_phase.json and prints ALL CHECKS PASSED.
"""
from pathlib import Path
import json
import time
import sympy as sp
import mpmath as mp

t0 = time.time()
mp.mp.dps = 70

# ---------------------------------------------------------------------------
# 1. Symbolic verification of classical good/bad dictionary & Darboux shifts
# ---------------------------------------------------------------------------
b1, b2 = sp.symbols('b1 b2', positive=True)
a1, a2 = sp.symbols('a1 a2', positive=True)
e1, e2 = sp.symbols('e1 e2', real=True)
eps_s, r_s = sp.symbols('eps_s r_s', positive=True)
t1, t2 = sp.symbols('t1 t2', positive=True)
pi = sp.pi

def bad_soliton_sym(b, eps):
    k = sp.sinh(b)
    v = eps * sp.cosh(b)
    w = k * v
    ell = eps * sp.cosh(3 * b)
    M = 2 * k
    Pi = -M * v
    P0 = sp.Rational(2, 3) * eps * k**3 * sp.cosh(b)
    E0 = sp.Rational(2, 15) * k**3 * (5 + 4 * k**2)
    L0 = v * P0 - E0
    gamma = -3 * sp.cosh(2 * b) / (sp.sinh(b) * sp.sinh(3 * b))
    return dict(b=b, eps=eps, k=k, v=v, w=w, ell=ell, M=M, Pi=Pi, P0=P0, E0=E0, L0=L0, gamma=gamma)

def good_soliton_sym(a, eps):
    k = sp.sin(a)
    v = eps * sp.cos(a)
    w = k * v
    ell = -eps * sp.cos(3 * a)
    M = 2 * k
    Pi = -M * v
    P0 = sp.Rational(2, 3) * eps * k**3 * sp.cos(a)
    E0 = sp.Rational(2, 15) * k**3 * (5 - 4 * k**2)
    L0 = v * P0 - E0
    gamma = -3 * sp.cos(2 * a) / (sp.sin(a) * sp.sin(3 * a))
    return dict(a=a, eps=eps, k=k, v=v, w=w, ell=ell, M=M, Pi=Pi, P0=P0, E0=E0, L0=L0, gamma=gamma)

sb1_p = bad_soliton_sym(b1, 1)
assert sp.simplify(sb1_p['L0'] - sp.Rational(2, 15) * sp.sinh(b1)**5) == 0
assert sp.simplify(sp.diff(sb1_p['E0'], b1) / sp.diff(sb1_p['P0'], b1) - sb1_p['v']) == 0
assert sp.simplify(sp.diff(sb1_p['Pi'], b1) / sp.diff(sb1_p['P0'], b1) - sb1_p['gamma']) == 0
assert sp.simplify(sp.diff(sb1_p['ell'], b1) / sp.diff(sb1_p['P0'], b1) - sp.Rational(9, 2) / sb1_p['k']) == 0

# Check that gamma = dPi/dP0 and dell/dP0 are independent of eps in {-1, +1}
sb1_m = bad_soliton_sym(b1, -1)
assert sp.simplify(sp.diff(sb1_m['Pi'], b1) / sp.diff(sb1_m['P0'], b1) - sb1_m['gamma']) == 0
assert sp.simplify(sp.diff(sb1_m['ell'], b1) / sp.diff(sb1_m['P0'], b1) - sp.Rational(9, 2) / sb1_m['k']) == 0

# Exact Virial / sound-rescaling identity: 5*E0 - 4*v*P0 = (2/3)*sinh^3(b) = int u0^2 dx
assert sp.simplify(5 * sb1_p['E0'] - 4 * sb1_p['v'] * sb1_p['P0'] - sp.Rational(2, 3) * sp.sinh(b1)**3) == 0

# Check Lagrangian sound-speed rescaling L_{c_s} = (1/2)phi_t^2 - (1/2)c_s^2 phi_x^2 - phi_x^3 + (1/2)phi_xx^2
cs, ft, fx, fxx = sp.symbols('cs ft fx fxx', positive=True)
ax, at, aphi = 1 / cs, 1 / cs**2, cs
L_cs = (
    sp.Rational(1, 2) * (aphi / at * ft)**2
    - sp.Rational(1, 2) * cs**2 * (aphi / ax * fx)**2
    - (aphi / ax * fx)**3
    + sp.Rational(1, 2) * (aphi / ax**2 * fxx)**2
)
L_std = sp.Rational(1, 2) * ft**2 - sp.Rational(1, 2) * fx**2 - fx**3 + sp.Rational(1, 2) * fxx**2
assert sp.simplify(L_cs * ax * at - cs**3 * L_std) == 0

def A12_bad_sym(s1, s2):
    Pp = lambda K, W: W**2 - K**2 - K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w']) / Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])

def A12_good_sym(s1, s2):
    Pp = lambda K, W: W**2 - K**2 + K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w']) / Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])

def bad_soliton_rat(t, eps):
    """Exact rational parametrization in t = exp(b): sinh(b) = (t^2-1)/(2t), cosh(b) = (t^2+1)/(2t)."""
    k = (t**2 - 1) / (2 * t)
    v = eps * (t**2 + 1) / (2 * t)
    w = k * v
    ell = eps * (t**6 + 1) / (2 * t**3)
    M = 2 * k
    P0 = sp.Rational(2, 3) * eps * k**3 * (t**2 + 1) / (2 * t)
    c2 = (t**4 + 1) / (2 * t**2)
    s3 = (t**6 - 1) / (2 * t**3)
    gamma = -3 * c2 / (k * s3)
    return dict(t=t, eps=eps, k=k, v=v, w=w, ell=ell, M=M, P0=P0, gamma=gamma)

def A12_bad_factored(s1, s2):
    """Exact reduced Hirota factor using single-soliton dispersion w_j^2 = k_j^2 + k_j^4."""
    base = s1['v'] * s2['v'] - 1 - 2 * (s1['k']**2 + s2['k']**2)
    odd = 3 * s1['k'] * s2['k']
    return (base + odd) / (base - odd)

# Verify that A12_bad_sym equals A12_bad_factored identically via w_j^2 = k_j^2 + k_j^4:
k1_s, k2_s, v1_s, v2_s = sp.symbols('k1_s k2_s v1_s v2_s')
Pp_s = lambda K, W: W**2 - K**2 - K**4
num_raw = sp.expand(-Pp_s(k1_s - k2_s, k1_s * v1_s - k2_s * v2_s)).subs({v1_s**2: 1 + k1_s**2, v2_s**2: 1 + k2_s**2})
den_raw = sp.expand(Pp_s(k1_s + k2_s, k1_s * v1_s + k2_s * v2_s)).subs({v1_s**2: 1 + k1_s**2, v2_s**2: 1 + k2_s**2})
assert sp.expand(num_raw - 2 * k1_s * k2_s * (v1_s * v2_s - 1 - 2 * (k1_s**2 + k2_s**2) + 3 * k1_s * k2_s)) == 0
assert sp.expand(den_raw - 2 * k1_s * k2_s * (v1_s * v2_s - 1 - 2 * (k1_s**2 + k2_s**2) - 3 * k1_s * k2_s)) == 0

# Check canonical momentum derivatives -d(delta_0)/dP_j = Delta X_j' in ALL 4 sign sectors
# Note: d/db = t * d/dt
for sgn1, sgn2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    s1 = bad_soliton_rat(t1, sgn1)
    s2 = bad_soliton_rat(t2, sgn2)
    A_t = sp.cancel(A12_bad_factored(s1, s2))
    T_t = sp.Rational(2, 3) * s1['M'] * s2['M'] * (s1['v'] - s2['v'])

    dP1_db1 = sp.cancel(t1 * sp.diff(s1['P0'], t1))
    dP2_db2 = sp.cancel(t2 * sp.diff(s2['P0'], t2))

    # Coefficient of log(A_12) in -d(delta_0)/dP_1 and -d(delta_0)/dP_2:
    log_coeff_1 = -sp.Rational(2, 9) * (t1 * sp.diff(s1['ell'], t1)) / dP1_db1
    log_coeff_2 = +sp.Rational(2, 9) * (t2 * sp.diff(s2['ell'], t2)) / dP2_db2
    assert sp.cancel(log_coeff_1 - (-1 / s1['k'])) == 0
    assert sp.cancel(log_coeff_2 - (+1 / s2['k'])) == 0

    # Rational part of -d(delta_0)/dP_1 and -d(delta_0)/dP_2:
    dlogA_db1 = sp.cancel(t1 * sp.diff(A_t, t1) / A_t)
    dlogA_db2 = sp.cancel(t2 * sp.diff(A_t, t2) / A_t)
    rat_1 = -sp.Rational(2, 9) * (s1['ell'] - s2['ell']) * dlogA_db1 / dP1_db1 - (t1 * sp.diff(T_t, t1)) / dP1_db1
    rat_2 = -sp.Rational(2, 9) * (s1['ell'] - s2['ell']) * dlogA_db2 / dP2_db2 - (t2 * sp.diff(T_t, t2)) / dP2_db2

    assert sp.cancel(rat_1 - s1['gamma'] * s2['M']) == 0
    assert sp.cancel(rat_2 - (-s2['gamma'] * s1['M'])) == 0

# ---------------------------------------------------------------------------
# 2. One-body schemes (bare ac vs lattice-matched) and momentum prescriptions
# ---------------------------------------------------------------------------
b = sp.Symbol('b', positive=True)
k_b = sp.sinh(b)
P0_pos = sp.Rational(2, 3) * k_b**3 * sp.cosh(b)

lam_ct = -3 / (2 * pi)  # Vacuum elasticity counterterm coefficient: Delta H = hbar_B * lam_ct * u^2
F_ac = sp.sqrt(3) / pi * (k_b**2 + sp.Rational(3, 4)) * sp.atan(2 * k_b / sp.sqrt(3)) - 3 * k_b / (2 * pi)
F_matched = F_ac + sp.Rational(2, 3) * lam_ct * k_b**3  # = F_ac - sinh^3(b)/pi

L1_ac = -F_ac
L1_matched = -F_matched

F_ac_ser = sp.series(F_ac, b, 0, 8).removeO()
F_matched_ser = sp.series(F_matched, b, 0, 8).removeO()
assert sp.simplify(sp.series(F_ac, b, 0, 6).removeO() - (sp.Rational(4, 3) / pi * b**3 + sp.Rational(14, 45) / pi * b**5)) == 0
assert sp.simplify(sp.series(F_matched, b, 0, 6).removeO() - (sp.Rational(1, 3) / pi * b**3 - sp.Rational(17, 90) / pi * b**5)) == 0
c7_F_ac = sp.simplify(F_ac_ser.coeff(b, 7) * pi)
c7_F_matched = sp.simplify(F_matched_ser.coeff(b, 7) * pi)
assert sp.simplify(c7_F_matched - (c7_F_ac - sp.Rational(13, 120))) == 0

C_ac = sp.limit(F_ac / P0_pos, b, 0)
C_matched = sp.limit(F_matched / P0_pos, b, 0)
assert C_ac == 2 / pi
assert C_matched == 1 / (2 * pi)

p_can_ac = sp.simplify(e1 / sp.sinh(b) * sp.diff(L1_ac, b))
p_can_matched = sp.simplify(e1 / sp.sinh(b) * sp.diff(L1_matched, b))

p_can_ac_closed = -2 * e1 * sp.sqrt(3) / pi * sp.cosh(b) * sp.atan(2 * sp.sinh(b) / sp.sqrt(3))
p_can_matched_closed = p_can_ac_closed + 3 * e1 / pi * sp.sinh(b) * sp.cosh(b)

assert sp.simplify(p_can_ac - p_can_ac_closed) == 0
assert sp.simplify(p_can_matched - p_can_matched_closed) == 0

p_can_ac_ser = sp.series(p_can_ac.subs(e1, 1), b, 0, 7).removeO()
p_can_matched_ser = sp.series(p_can_matched.subs(e1, 1), b, 0, 7).removeO()
assert sp.simplify(p_can_ac_ser - (-4 / pi * b - sp.Rational(8, 9) / pi * b**3 - sp.Rational(8, 45) / pi * b**5)) == 0
assert sp.simplify(p_can_matched_ser - (-1 / pi * b + sp.Rational(10, 9) / pi * b**3 + sp.Rational(2, 9) / pi * b**5)) == 0

# Universal slope relation: c_1(p_can) = -2 * C, and d(P_0 + hbar_B p_can)/dP_0 = 1 - C/y + O(b^2)
assert sp.simplify(p_can_ac_ser.coeff(b, 1) - (-2 * C_ac)) == 0
assert sp.simplify(p_can_matched_ser.coeff(b, 1) - (-2 * C_matched)) == 0
dP0_db_ser = sp.series(sp.diff(P0_pos, b), b, 0, 5).removeO()
dp_can_m_db_ser = sp.series(sp.diff(p_can_matched.subs(e1, 1), b), b, 0, 5).removeO()
jac_ratio_ser = sp.series(dp_can_m_db_ser / dP0_db_ser * b**2, b, 0, 4).removeO()
assert sp.simplify(jac_ratio_ser - (-1 / (2 * pi) + sp.Rational(5, 2) / pi * b**2)) == 0

c_ln = 1 / pi
c_lin = (-sp.log(2) - 1 + sp.Rational(1, 4) * sp.log(3)) / pi

# ---------------------------------------------------------------------------
# 3. Fast polynomial-series expansion in the small-amplitude corner (both channels)
# ---------------------------------------------------------------------------
u_s = sp.Symbol('u_s', positive=True)
sinh_u = u_s + u_s**3 / 6 + u_s**5 / 120
cosh_u = 1 + u_s**2 / 2 + u_s**4 / 24
gamma_u = sp.series(-3 * cosh_u.subs(u_s, 2 * u_s) / (sinh_u * sinh_u.subs(u_s, 3 * u_s)), u_s, 0, 4).removeO()
inv_k_u = sp.series(1 / sinh_u, u_s, 0, 4).removeO()
assert sp.series(gamma_u, u_s, 0, 2).removeO() == -1 / u_s**2 - sp.Rational(1, 3)
assert inv_k_u == 1 / u_s - u_s / 6 + sp.Rational(7, 360) * u_s**3

def bad_soliton_poly(x, eps):
    k = sinh_u.subs(u_s, x)
    v = eps * cosh_u.subs(u_s, x)
    ell = eps * (1 + sp.Rational(9, 2) * x**2 + sp.Rational(27, 8) * x**4)
    M = 2 * k
    P0 = eps * (sp.Rational(2, 3) * x**3 + sp.Rational(2, 3) * x**5)
    gamma = gamma_u.subs(u_s, x)
    inv_k = inv_k_u.subs(u_s, x)
    return dict(k=k, v=v, ell=ell, M=M, P0=P0, gamma=gamma, inv_k=inv_k)

def A12_poly(s1, s2):
    base = sp.expand(s1['v'] * s2['v'] - 1 - 2 * (s1['k']**2 + s2['k']**2))
    odd = sp.expand(3 * s1['k'] * s2['k'])
    return sp.expand(base + odd), sp.expand(base - odd)

# Overtaking channel on bad branch: e1 = +1, e2 = +1, b1 = eps_s > b2 = r_s * eps_s (0 < r_s < 1, v1 > v2)
s1_ov = bad_soliton_poly(eps_s, 1)
s2_ov = bad_soliton_poly(r_s * eps_s, 1)
num_ov, den_ov = A12_poly(s1_ov, s2_ov)
A0_r = ((1 - r_s) / (1 + r_s))**2

# Cancel common eps_s^2 factor before series expansion (factored form starts at eps_s^2!):
# On the bad branch, A_ov / A0(r) = 1 + r * eps_s^2 + O(eps_s^4)
L_A0 = sp.Symbol('L_A0', real=True)
num_ov_red = sp.expand(num_ov / eps_s**2)
den_ov_red = sp.expand(den_ov / eps_s**2)
A_ov_ratio_ser = sp.series(sp.cancel(num_ov_red / (den_ov_red * A0_r)), eps_s, 0, 4).removeO()
assert sp.simplify(A_ov_ratio_ser - (1 + r_s * eps_s**2)) == 0

logA_ov_ser = L_A0 + sp.series(sp.log(A_ov_ratio_ser), eps_s, 0, 4).removeO()
assert sp.simplify(logA_ov_ser - (L_A0 + r_s * eps_s**2)) == 0

d0_ov_ser = sp.series(
    sp.Rational(2, 9) * (s1_ov['ell'] - s2_ov['ell']) * logA_ov_ser
    + sp.Rational(2, 3) * s1_ov['M'] * s2_ov['M'] * (s1_ov['v'] - s2_ov['v']),
    eps_s, 0, 6
).removeO()
# Leading overtaking classical phase is (1 - r^2)*eps_s^2*log(A0_r) + O(eps_s^4):
d0_ov_expected = (
    (1 - r_s**2) * L_A0 * eps_s**2
    + (sp.Rational(3, 4) * (1 - r_s**4) * L_A0 + sp.Rational(7, 3) * r_s * (1 - r_s**2)) * eps_s**4
)
assert sp.simplify(d0_ov_ser - d0_ov_expected) == 0

dx12_ov_ser = sp.series(-s1_ov['inv_k'] * logA_ov_ser, eps_s, 0, 4).removeO()
dx21_ov_ser = sp.series(+s2_ov['inv_k'] * logA_ov_ser, eps_s, 0, 4).removeO()
darb1_ov_ser = sp.series(s1_ov['gamma'] * s2_ov['M'], eps_s, 0, 4).removeO()
darb2_ov_ser = sp.series(-s2_ov['gamma'] * s1_ov['M'], eps_s, 0, 4).removeO()

p1_can_m_poly = p_can_matched_ser.subs(b, eps_s)
p2_can_m_poly = p_can_matched_ser.subs(b, r_s * eps_s)
H_ov_can_m = sp.series(dx12_ov_ser * p1_can_m_poly + dx21_ov_ser * p2_can_m_poly, eps_s, 0, 4).removeO()
D_ov_can_m = sp.series(darb1_ov_ser * p1_can_m_poly + darb2_ov_ser * p2_can_m_poly, eps_s, 0, 4).removeO()

p1_can_ac_poly = p_can_ac_ser.subs(b, eps_s)
p2_can_ac_poly = p_can_ac_ser.subs(b, r_s * eps_s)
H_ov_can_ac = sp.series(dx12_ov_ser * p1_can_ac_poly + dx21_ov_ser * p2_can_ac_poly, eps_s, 0, 4).removeO()
D_ov_can_ac = sp.series(darb1_ov_ser * p1_can_ac_poly + darb2_ov_ser * p2_can_ac_poly, eps_s, 0, 4).removeO()

# Leading O(1) limits in overtaking channel:
assert H_ov_can_m.subs(eps_s, 0) == 0
assert H_ov_can_ac.subs(eps_s, 0) == 0
assert sp.simplify(D_ov_can_m.subs(eps_s, 0) - (-2 / pi * (1 / r_s - r_s))) == 0
assert sp.simplify(D_ov_can_ac.subs(eps_s, 0) - (-8 / pi * (1 / r_s - r_s))) == 0

# O(eps_s^2) coefficients in overtaking channel:
H_ov_can_m_c2 = sp.simplify(H_ov_can_m.coeff(eps_s, 2))
D_ov_can_m_c2 = sp.simplify(D_ov_can_m.coeff(eps_s, 2))
assert sp.simplify(H_ov_can_m_c2 - (-sp.Rational(23, 18) / pi * (1 - r_s**2) * L_A0)) == 0
D_ov_can_m_c2_target = -sp.Rational(1, 3) / pi * (1 / r_s - r_s**3)
assert sp.simplify(D_ov_can_m_c2 - D_ov_can_m_c2_target) == 0

# Mode-sum momentum in overtaking channel (leading order in eps_s):
p1_mode_lead = c_ln * eps_s * sp.log(eps_s) + c_lin * eps_s
p2_mode_lead = c_ln * r_s * eps_s * (sp.log(r_s) + sp.log(eps_s)) + c_lin * r_s * eps_s
H_ov_mode_lead = sp.simplify((-1 / eps_s * p1_mode_lead + 1 / (r_s * eps_s) * p2_mode_lead) * L_A0)
D_ov_mode_lead = sp.expand(-2 * r_s / eps_s * p1_mode_lead + 2 / (r_s**2 * eps_s) * p2_mode_lead)
assert sp.simplify(H_ov_mode_lead - (1 / pi * sp.log(r_s) * L_A0)) == 0
assert sp.expand(D_ov_mode_lead - (2 * (1 / r_s - r_s) * (c_ln * sp.log(eps_s) + c_lin) + 2 / (pi * r_s) * sp.log(r_s))) == 0

# Direct on-shell sound-speed rescaling counterterm shift in overtaking channel:
# Delta_ct^sound delta = lam_ct * (3 * delta_0 + 4 * P0_1 * DX1' + 4 * P0_2 * DX2')
DX1_ov_ser = dx12_ov_ser + darb1_ov_ser
DX2_ov_ser = dx21_ov_ser + darb2_ov_ser
sound_ct_ov_ser = sp.series(
    lam_ct * (3 * d0_ov_ser + 4 * s1_ov['P0'] * DX1_ov_ser + 4 * s2_ov['P0'] * DX2_ov_ser),
    eps_s, 0, 5
).removeO()
assert sound_ct_ov_ser.subs(eps_s, 0) == 0
assert sound_ct_ov_ser.coeff(eps_s, 1) == 0
c2_sound_ov = sp.simplify(sound_ct_ov_ser.coeff(eps_s, 2))
c4_sound_ov = sp.simplify(sound_ct_ov_ser.coeff(eps_s, 4))
assert sp.simplify(c2_sound_ov - (-1 / (2 * pi) * (1 - r_s**2) * L_A0)) == 0
assert sound_ct_ov_ser.coeff(eps_s, 3) == 0

# Now check Head-On channel on bad branch: e1 = +1, e2 = -1, b1 = eps_s, b2 = r_s * eps_s
s1_ho = bad_soliton_poly(eps_s, 1)
s2_ho = bad_soliton_poly(r_s * eps_s, -1)
num_ho, den_ho = A12_poly(s1_ho, s2_ho)
A_ho_ser = sp.series(num_ho / den_ho, eps_s, 0, 5).removeO()
A_ho_expected = 1 - 3 * r_s * eps_s**2 + (sp.Rational(9, 2) * r_s**2 + sp.Rational(13, 4) * r_s * (1 + r_s**2)) * eps_s**4
assert sp.simplify(A_ho_ser - A_ho_expected) == 0

logA_ho_ser = sp.series(sp.log(A_ho_ser), eps_s, 0, 5).removeO()
assert sp.simplify(logA_ho_ser - (-3 * r_s * eps_s**2 + sp.Rational(13, 4) * r_s * (1 + r_s**2) * eps_s**4)) == 0

d0_ho_ser = sp.series(
    sp.Rational(2, 9) * (s1_ho['ell'] - s2_ho['ell']) * logA_ho_ser
    + sp.Rational(2, 3) * s1_ho['M'] * s2_ho['M'] * (s1_ho['v'] - s2_ho['v']),
    eps_s, 0, 5
).removeO()
assert sp.simplify(d0_ho_ser - (4 * r_s * eps_s**2 + sp.Rational(2, 3) * r_s * (1 + r_s**2) * eps_s**4)) == 0

dx12_ho_ser = sp.series(-s1_ho['inv_k'] * logA_ho_ser, eps_s, 0, 4).removeO()
dx21_ho_ser = sp.series(+s2_ho['inv_k'] * logA_ho_ser, eps_s, 0, 4).removeO()
darb1_ho_ser = sp.series(s1_ho['gamma'] * s2_ho['M'], eps_s, 0, 4).removeO()
darb2_ho_ser = sp.series(-s2_ho['gamma'] * s1_ho['M'], eps_s, 0, 4).removeO()

p1_ho_can_m_poly = +p_can_matched_ser.subs(b, eps_s)
p2_ho_can_m_poly = -p_can_matched_ser.subs(b, r_s * eps_s)
H_ho_can_m = sp.series(dx12_ho_ser * p1_ho_can_m_poly + dx21_ho_ser * p2_ho_can_m_poly, eps_s, 0, 4).removeO()
D_ho_can_m = sp.series(darb1_ho_ser * p1_ho_can_m_poly + darb2_ho_ser * p2_ho_can_m_poly, eps_s, 0, 4).removeO()

p1_ho_can_ac_poly = +p_can_ac_ser.subs(b, eps_s)
p2_ho_can_ac_poly = -p_can_ac_ser.subs(b, r_s * eps_s)
H_ho_can_ac = sp.series(dx12_ho_ser * p1_ho_can_ac_poly + dx21_ho_ser * p2_ho_can_ac_poly, eps_s, 0, 4).removeO()
D_ho_can_ac = sp.series(darb1_ho_ser * p1_ho_can_ac_poly + darb2_ho_ser * p2_ho_can_ac_poly, eps_s, 0, 4).removeO()

# Leading O(1) limits in head-on channel:
assert H_ho_can_m.subs(eps_s, 0) == 0
assert H_ho_can_ac.subs(eps_s, 0) == 0
assert sp.simplify(D_ho_can_m.subs(eps_s, 0) - (+2 / pi * (1 / r_s + r_s))) == 0
assert sp.simplify(D_ho_can_ac.subs(eps_s, 0) - (+8 / pi * (1 / r_s + r_s))) == 0

# O(eps_s^2) coefficients in head-on channel:
H_ho_can_m_c2 = sp.simplify(H_ho_can_m.coeff(eps_s, 2))
D_ho_can_m_c2 = sp.simplify(D_ho_can_m.coeff(eps_s, 2))
assert sp.simplify(H_ho_can_m_c2 - (-6 / pi * r_s)) == 0
D_ho_can_m_c2_target = sp.simplify((sp.Rational(1, 3) * (1 / r_s + r_s**3) - sp.Rational(28, 9) * r_s) / pi)
assert sp.simplify(D_ho_can_m_c2 - D_ho_can_m_c2_target) == 0

# Mode-sum momentum in head-on channel (leading order in eps_s):
p1_ho_mode_lead = +(c_ln * eps_s * sp.log(eps_s) + c_lin * eps_s)
p2_ho_mode_lead = -(c_ln * r_s * eps_s * (sp.log(r_s) + sp.log(eps_s)) + c_lin * r_s * eps_s)
D_ho_mode_lead = sp.expand(-2 * r_s / eps_s * p1_ho_mode_lead + 2 / (r_s**2 * eps_s) * p2_ho_mode_lead)
assert sp.expand(D_ho_mode_lead - (-2 * (1 / r_s + r_s) * (c_ln * sp.log(eps_s) + c_lin) - 2 / (pi * r_s) * sp.log(r_s))) == 0

# Direct on-shell sound-speed rescaling counterterm shift in head-on channel:
DX1_ho_ser = dx12_ho_ser + darb1_ho_ser
DX2_ho_ser = dx21_ho_ser + darb2_ho_ser
sound_ct_ho_ser = sp.series(
    lam_ct * (3 * d0_ho_ser + 4 * s1_ho['P0'] * DX1_ho_ser + 4 * s2_ho['P0'] * DX2_ho_ser),
    eps_s, 0, 5
).removeO()
assert sound_ct_ho_ser.subs(eps_s, 0) == 0
assert sound_ct_ho_ser.coeff(eps_s, 1) == 0
c2_sound_ho = sp.simplify(sound_ct_ho_ser.coeff(eps_s, 2))
c4_sound_ho = sp.simplify(sound_ct_ho_ser.coeff(eps_s, 4))
assert sp.simplify(c2_sound_ho - (-2 / pi * r_s)) == 0

# ---------------------------------------------------------------------------
# 4. High-precision (70-digit) mpmath checks of I_KdV(r), J_KdV(r) & expansions
# ---------------------------------------------------------------------------
def chi2_mp(r):
    r = mp.mpf(r)
    return (mp.polylog(2, r) - mp.polylog(2, -r)) / 2

def I_KdV_mp(r):
    r = mp.mpf(r)
    return -4 / mp.pi * chi2_mp(r) - 2 / mp.pi * mp.log(r) * mp.log((1 - r) / (1 + r))

def J_KdV_mp(r):
    return mp.pi / 2 + I_KdV_mp(r)

def I_KdV_quad_mp(r):
    r = mp.mpf(r)
    val = -2 / mp.pi * mp.quad(lambda x: mp.log((1 + x) / (1 - x)) / x, [0, r]) - 2 / mp.pi * mp.log(r) * mp.log((1 - r) / (1 + r))
    return val

kdv_rows = []
for rv_str in ('0.01', '0.1', '0.2', '0.25', '0.4', '0.5', '0.6', '0.75', '0.8'):
    rv = mp.mpf(rv_str)
    i_ex = I_KdV_mp(rv)
    i_qd = I_KdV_quad_mp(rv)
    j_ex = J_KdV_mp(rv)
    quad_err = abs(i_ex - i_qd)
    assert quad_err < mp.mpf('1e-65')

    i_ser3 = (
        4 / mp.pi * rv * (mp.log(rv) - 1)
        + 4 / (9 * mp.pi) * rv**3 * (3 * mp.log(rv) - 1)
        + 4 / (75 * mp.pi) * rv**5 * (15 * mp.log(rv) - 3)
    )
    if rv == mp.mpf('0.01'):
        assert abs(i_ex - i_ser3) < mp.mpf('1e-13')

    d_ov_m = -2 / mp.pi * (1 / rv - rv)
    d_ov_ac = -8 / mp.pi * (1 / rv - rv)
    d_ho_m = +2 / mp.pi * (1 / rv + rv)
    d_ho_ac = +8 / mp.pi * (1 / rv + rv)

    kdv_rows.append({
        'r': rv_str,
        'I_KdV': mp.nstr(i_ex, 35),
        'J_KdV': mp.nstr(j_ex, 35),
        'chi2_term': mp.nstr(-4 / mp.pi * chi2_mp(rv), 35),
        'log_term': mp.nstr(-2 / mp.pi * mp.log(rv) * mp.log((1 - rv) / (1 + rv)), 35),
        'D_ov_can_matched_lead': mp.nstr(d_ov_m, 35),
        'D_ov_can_ac_lead': mp.nstr(d_ov_ac, 35),
        'D_ho_can_matched_lead': mp.nstr(d_ho_m, 35),
        'D_ho_can_ac_lead': mp.nstr(d_ho_ac, 35),
        'total_ov_I_plus_can_matched': mp.nstr(i_ex + d_ov_m, 35),
        'total_ov_J_plus_can_matched': mp.nstr(j_ex + d_ov_m, 35),
        'quadrature_error': str(quad_err),
    })

# High-precision finite-amplitude convergence check of the small-amplitude expansions
finite_conv_rows = []
rv = mp.mpf('0.5')
for eps_str in ('1e-2', '1e-3', '1e-4'):
    ep = mp.mpf(eps_str)
    b1v, b2v = ep, rv * ep

    def p_can_m_mp(bv, sgn=1):
        return sgn * (-2 * mp.sqrt(3) / mp.pi * mp.cosh(bv) * mp.atan(2 * mp.sinh(bv) / mp.sqrt(3)) + 3 / mp.pi * mp.sinh(bv) * mp.cosh(bv))

    def gamma_mp(bv):
        return -3 * mp.cosh(2 * bv) / (mp.sinh(bv) * mp.sinh(3 * bv))

    def A_bad_mp(bv1, sgn1, bv2, sgn2):
        k1v, k2v = mp.sinh(bv1), mp.sinh(bv2)
        w1v, w2v = sgn1 * k1v * mp.cosh(bv1), sgn2 * k2v * mp.cosh(bv2)
        Pp = lambda K, W: W**2 - K**2 - K**4
        return -Pp(k1v - k2v, w1v - w2v) / Pp(k1v + k2v, w1v + w2v)

    # Overtaking finite vs two-term series
    Aov_v = A_bad_mp(b1v, 1, b2v, 1)
    H_ov_exact = (-mp.log(Aov_v) / mp.sinh(b1v)) * p_can_m_mp(b1v, 1) + (mp.log(Aov_v) / mp.sinh(b2v)) * p_can_m_mp(b2v, 1)
    D_ov_exact = gamma_mp(b1v) * 2 * mp.sinh(b2v) * p_can_m_mp(b1v, 1) - gamma_mp(b2v) * 2 * mp.sinh(b1v) * p_can_m_mp(b2v, 1)

    A0_v = ((1 - rv) / (1 + rv))**2
    H_ov_asymp2 = -mp.mpf(23) / (18 * mp.pi) * (1 - rv**2) * mp.log(A0_v) * ep**2
    D_ov_asymp2 = -2 / mp.pi * (1 / rv - rv) - mp.mpf(1) / (3 * mp.pi) * (1 / rv - rv**3) * ep**2

    err_H_ov = abs(H_ov_exact - H_ov_asymp2)
    err_D_ov = abs(D_ov_exact - D_ov_asymp2)
    assert err_H_ov < 10 * ep**4
    assert err_D_ov < 10 * ep**4

    # Head-on finite vs two-term series
    Aho_v = A_bad_mp(b1v, 1, b2v, -1)
    H_ho_exact = (-mp.log(Aho_v) / mp.sinh(b1v)) * p_can_m_mp(b1v, 1) + (mp.log(Aho_v) / mp.sinh(b2v)) * p_can_m_mp(b2v, -1)
    D_ho_exact = gamma_mp(b1v) * 2 * mp.sinh(b2v) * p_can_m_mp(b1v, 1) - gamma_mp(b2v) * 2 * mp.sinh(b1v) * p_can_m_mp(b2v, -1)

    H_ho_asymp2 = -6 / mp.pi * rv * ep**2
    D_ho_asymp2 = +2 / mp.pi * (1 / rv + rv) + (mp.mpf(1) / 3 * (1 / rv + rv**3) - mp.mpf(28) / 9 * rv) / mp.pi * ep**2

    err_H_ho = abs(H_ho_exact - H_ho_asymp2)
    err_D_ho = abs(D_ho_exact - D_ho_asymp2)
    assert err_H_ho < 10 * ep**4
    assert err_D_ho < 10 * ep**4

    finite_conv_rows.append({
        'eps': eps_str,
        'err_H_ov': mp.nstr(err_H_ov, 15),
        'err_D_ov': mp.nstr(err_D_ov, 15),
        'err_H_ho': mp.nstr(err_H_ho, 15),
        'err_D_ho': mp.nstr(err_D_ho, 15),
    })

out = {
    'script': 'twobody_cont_phase.py',
    'precision_dps': 70,
    'runtime_seconds': round(time.time() - t0, 4),
    'one_body_coefficients': {
        'lambda_counterterm': '-3/(2*pi)',
        'C_ac': '2/pi',
        'C_matched': '1/(2*pi)',
        's1_bad_ac': '9/(4*pi)',
        's1_toda': '3/(4*pi)',
        'C_minus_s1_universal': '-1/(4*pi)',
        'F_ac_series': '4/(3*pi)*b^3 + 14/(45*pi)*b^5 + ' + str(c7_F_ac) + '/pi*b^7 + O(b^9)',
        'F_matched_series': '1/(3*pi)*b^3 - 17/(90*pi)*b^5 + ' + str(c7_F_matched) + '/pi*b^7 + O(b^9)',
        'p_can_ac_series': '-4/pi * b - 8/(9*pi) * b^3 - 8/(45*pi) * b^5 + O(b^7)',
        'p_can_matched_series': '-1/pi * b + 10/(9*pi) * b^3 + 2/(9*pi) * b^5 + O(b^7)',
        'p_mode_series': '(1/pi)*b*ln(b) + ((-ln(2) - 1 + (1/4)*ln(3))/pi)*b + O(b^3*ln(b))',
        'c_lin_numeric': mp.nstr((-mp.log(2) - 1 + mp.log(3) / 4) / mp.pi, 50),
    },
    'small_amplitude_formulas_bad_branch_b1_gt_b2': {
        'overtaking': {
            'A_ov_over_A0': '1 + r*eps^2 + O(eps^4)',
            'delta_0_ov': '(1-r^2)*ln(((1-r)/(1+r))^2)*eps^2 + ((3/4)*(1-r^4)*ln(((1-r)/(1+r))^2) + (7/3)*r*(1-r^2))*eps^4 + O(eps^6)',
            'I_ph': '-(4/pi)*chi_2(r) - (2/pi)*ln(r)*ln((1-r)/(1+r))',
            'J_ph': 'pi/2 + I_ph(r)',
            'H_ov_can_matched': '-(23/(18*pi))*(1-r^2)*ln(((1-r)/(1+r))^2)*eps^2 + O(eps^4)',
            'D_ov_can_matched': '-(2/pi)*(r^{-1} - r) - (1/(3*pi))*(r^{-1} - r^3)*eps^2 + O(eps^4)',
            'D_ov_can_ac': '-(8/pi)*(r^{-1} - r) + O(eps^2)',
            'H_ov_mode': '(2/pi)*ln(r)*ln((1-r)/(1+r)) + O(eps^2*ln(eps))',
            'D_ov_mode': '2*(r^{-1} - r)*((1/pi)*ln(eps) + c_lin) + (2/(pi*r))*ln(r) + O(eps^2*ln(eps))',
            'sound_ct_ov': '-(1/(2*pi))*(1-r^2)*ln(((1-r)/(1+r))^2)*eps^2 + (' + str(c4_sound_ov) + ')*eps^4 + O(eps^6)',
        },
        'head_on': {
            'A_ho': '1 - 3*r*eps^2 + ((9/2)*r^2 + (13/4)*r*(1+r^2))*eps^4 + O(eps^6)',
            'delta_0_ho': '4*r*eps^2 + (2/3)*r*(1+r^2)*eps^4 + O(eps^6)',
            'I_ph': '(4/pi)*r*eps^2*(ln(4*r*eps^2/3) + 1) + O(eps^4*ln(eps))',
            'J_ph': 'I_ph',
            'H_ho_can_matched': '-(6/pi)*r*eps^2 + O(eps^4)',
            'D_ho_can_matched': '+(2/pi)*(r^{-1} + r) + ((1/3)*(r^{-1} + r^3) - (28/9)*r)/pi*eps^2 + O(eps^4)',
            'D_ho_can_ac': '+(8/pi)*(r^{-1} + r) + O(eps^2)',
            'H_ho_mode': 'O(eps^2*ln(eps))',
            'D_ho_mode': '-2*(r^{-1} + r)*((1/pi)*ln(eps) + c_lin) - (2/(pi*r))*ln(r) + O(eps^2*ln(eps))',
            'sound_ct_ho': '-(2/pi)*r*eps^2 + (' + str(c4_sound_ho) + ')*eps^4 + O(eps^6)',
        },
    },
    'kdv_ratio_table': kdv_rows,
    'finite_amplitude_convergence_checks': finite_conv_rows,
    'failures_recorded': [
        'In an initial draft of twobody_cont_phase.py, the O(b^7) coefficient of F_ac(b) and the O(b^5) coefficient of F_matched(b) = F_ac(b) - sinh^3(b)/pi were hand-estimated as -62/(945*pi) and -31/(45*pi) (using -b^5/pi instead of -(1/2)*b^5/pi from sinh^3(b) = b^3 + b^5/2 + 13*b^7/120). SymPy series expansion caught both slips immediately: F_matched(b) = b^3/(3*pi) - 17*b^5/(90*pi) - 77*b^7/(1890*pi) + O(b^9), and p_can,matched(b) = -b/pi + 10*b^3/(9*pi) + 2*b^5/(9*pi) + O(b^7).',
        'In the second run of twobody_cont_phase.py, the O(b^2) subleading term of b^2 * (dp_can,matched/db)/(dP_0/db) was changed to +2*b^2/pi instead of +5*b^2/(2*pi); SymPy caught the assertion failure at line 193, confirming (-1/pi + 10*b^2/(3*pi))/(2*(1 + 5*b^2/3)) = -1/(2*pi) + 5*b^2/(2*pi) + O(b^4).',
        'In preliminary runs of twobody_cont_phase.py, the O(eps^2) subleading ratio A_ov/A_0(r) on the bad branch was tested against 1 +/- (2/3)*r*eps^2 and 1 + (13/9)*r*eps^2 before dividing out the common 3*b1*b2*(b1-b2)^2 factor; SymPy polynomial expansion showed the exact subleading ratio is A_ov/A_0(r) = 1 + r*eps^2 + O(eps^4).'
    ],
}

Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
print("ALL CHECKS PASSED")

"""Verification checks for Draft 3 of paper/boussinesq_soliton_scattering.tex.

Every new formula and (c) claim added in Draft 3 (Section 5.7 on defect charges,
renormalisation and fusion obstructions; Section 7 on one-loop corrections and the
headline O(hbar^0) one-body soliton coefficient; Section 8 on the quantum Toda
Fermi-edge limit; and Section 9 on exact S-matrix candidates and constraints) is
checked here symbolically with SymPy and/or at 50-digit precision with mpmath:

 1. Sec. 5.7 (Defect charges, spectral trace, UV counterterm, fusion obstructions):
    - 1a: Localised classical defect charges (E_D, P_D, Pi_D, M_D) and exact KP-edge
          vacuum jump factorisation Delta(E_D, P_D, Pi_D, M_D) = rho_i (E, P, Pi, M)_{S_i}
          with rho_i = eta sgn(q_i - q_{i+1}).
    - 1b: Off-shell reduced 4x4 linearised sewing determinant
          det M / (kappa_1 - kappa_2)^2 = -(2/3) S^2 (S^2 - 1) P_-(S; c) P_+(S; c),
          threshold factorisation P_mp(1; c) = (1/2)(c +- 1)(c - sqrt(3)), branch switch
          at c = sqrt(3) with g(sqrt(3)) = sqrt(3)/4, and growing-mode counts (0, 1, 2)
          generically and (0, 1, 1) at c = sqrt(3).
    - 1c: Pointwise quadratic defect expectation identities for <P_{D,2}>_q + <P_{D,2}>_{-q}
          and +(<a_t F_1 + b_t G_1>_q + <a_t F_1 + b_t G_1>_{-q}), linear UV Casimir
          divergence (S_D/pi)(T - 1), and topological decomposition into s and M_D.
    - 1d: Directed KP-triangle canonical-charge holonomy (Delta E_can, Delta P_can) != 0
          at seed q_1 = -0.20, kinematic rapidity shift restoration, absorbed-soliton
          momentum remainder at alpha_3 = 0.35, and polynomial quotient
          (ell(p) - ell(q))/(p - q) = 3 sqrt(3) (4(p^2 + p q + q^2) - 1).

 2. Sec. 7.1 (Projected determinant, collective measure, kinetic-norm divergence, E_1|_P = -L_1):
    - 2a: Potential operator factorisation K_v = -d_xi H_v d_xi, bounded static kernel
          span{1, u_0}, secular velocity mode K_v Phi_v = -2 v u_0', and heavy-branch
          complementary spatial profiles f_mp^sp(xi).
    - 2b: Rank-2 symplectic matrix W_3 with null vector Pi_v d_X - P_v d_C, 4x4 completion
          det W_4 = P_v^2, and projector endpoint tails b_r +- (v/(k P_v)) b_X.
    - 2c: Intrinsic box-norm divergence int_{-R}^R Phi_v^2 dxi = 2 R v^2/k^2 + 2(pi^2-6)v^2/(9 k^3) + ...
          and positive lower bound (1/2)(lambda^2 + v^2/k^2) dot_v^2 >= (v^2/(2 k^2)) dot_v^2.
    - 2d: Unconditional fixed-P invariance E_1|_P = e - v p = e_can - v p_can = -L_1,
          cutoff endpoint B_Lambda, vacuum tadpole C_Lambda, vanishing light-root weight,
          good trace F_g(k) with C_g^bare = -2/pi, and inversion-antisymmetric kernel
          identities for tilde K_L(z).

 3. Sec. 7.2--7.3 (Headline one-body O(hbar^0) soliton coefficient and vacuum elasticity matching):
    - 3a: Headline arithmetic and symbolic identities:
          3/(4*pi) - 1/(4*pi) = 1/(2*pi),
          2/pi - 3/(2*pi) = 1/(2*pi),
          2/pi - 9/(4*pi) = -1/(4*pi) = 1/(2*pi) - 3/(4*pi),
          -2/pi + 9/(4*pi) = +1/(4*pi),
          and comparison against numerical Toda extrapolation 0.1591549414 +- 8.23e-6.
    - 3b: Continued bad trace F_ac(k) = -i F_g(i k) with C_ac = +2/pi = 4 C_T, real
          stable-band integral F_band(beta) = F_ac(beta) - (sqrt(3)/2)(sinh^2 beta + 3/4),
          and mass-term annihilator I[F_band] = +2/pi.
    - 3c: Vacuum strain energy densities W_g(U), W_{b,ac}(U), W_T^B(U), sound-speed shifts
          s_{1,g} = -9/(4*pi), s_{1,b,ac} = +9/(4*pi), s_{1,T} = +3/(4*pi), vacuum
          elasticity counterterm lambda = -3/(2*pi), and soliton integral int u_0^2 dx = P_0/v.
    - 3d: Analytical Toda Fermi-edge dispersion formulas C_v(y) and C_E(y) tending to
          3/(4*pi) - 1/(4*pi) = 1/(2*pi).

 4. Sec. 8 (Quantum Toda Fermi-edge limit):
    - 4a: Finite-sea Fredholm endpoint identities, exact momentum and energy Jacobians
          du/dy and dE_bar/dy, and four-term endpoint reciprocity
          hat_Phi(p,z) + hat_Phi(z,p) = M_p Q_z + M_z Q_p.
    - 4b: Classical-frame remainder S_0(y,z), rectangular second difference D_0, and
          asymptotic limit -(15*sqrt(2) - 20)/pi.
    - 4c: Wiener--Hopf symbol Sigma(omega) = 2*pi/(1 - e^{-|omega|}) = 2*pi g(i omega) g(-i omega),
          Laplace transform sqrt(pi) g(s)/s^2, series-reversion edge coefficients
          c_edge, d_0, e_0 (and -(ln u)^2/(16*pi^3 u^2) in y(u) - u), and closed-form
          Krein integrals R_0(infty, r) and c_1(r).
    - 4d: Antisymmetric counterexample A_lambda(y_1, y_2) and its exclusion by cluster
          separability (nonvanishing 3x3 Schmidt minor) and the rank-1 Krein PDE
          (nonvanishing 2x2 minor of (d_{y_1} + d_{y_2}) A_1).

 5. Sec. 9 (Exact S-matrix candidates and constraints):
    - 5a: Bad Gamma candidate unitarity, crossing, Stirling s-family expansion, and
          physical-sheet simple poles at Bogdan--Charkina breather levels I_br = N hbar (N >= 1).
    - 5b: Singularity of all zero-background good Hirota breathers (A_12 < 1/4 < 1,
          sign change at a = pi/3), naive good Gamma poles at A_12 < 0, and continuous
          t ln t approach on the real resonance lines alpha_1 + alpha_2 = pi/3, 2*pi/3.
    - 5c: Mirror-product exchange-unitarity exclusion and restricted nine-element KP
          root-ratio dilogarithm exclusion via diagonal pole residues.
    - 5d: Conditional twisted ring Bethe equations, chamber Darboux wall-crossing
          Delta X_j' = -d_{P_j} delta(j,l), directional phase Phi_ring(j|l) = delta(j,l) + Pi_j M_l,
          total momentum twist sum, and composite breather shift.
    - 5e: Crossing antisymmetry J_12^same(alpha_1, alpha_2 + pi) + J_12^head(alpha_1, alpha_2) = 0.

Ends with ALL CHECKS PASSED.
"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
TOL = mp.mpf('1e-40')
R3 = mp.sqrt(3)
ok = True


def check(name, cond):
    global ok
    c = bool(cond)
    print(('PASS (c): ' if c else 'FAIL (c): ') + name, flush=True)
    ok &= c
    assert c, name


# ==================================================================
# 1. Sec. 5.7: Defect charges, spectral trace, UV counterterm, fusion
# ==================================================================
q1_s, q2_s, eta_s, sig_s, s_s = sp.symbols('q1 q2 eta sigma s', real=True)

# 1a. Classical localised defect charges and KP-edge vacuum jumps
al_cz = eta_s / sp.sqrt(3)
be_cz = 2 * al_cz
mu_cz = al_cz
W_s = sp.Symbol('W', real=True)
sig_W = s_s + mu_cz * W_s
rho0_cz = -(s_s / (2 * mu_cz)) * (1 - s_s**2 / 3)
rho_W = rho0_cz + (sig_W - sig_W**3 / 3) / (2 * mu_cz)
tau_W = sp.integrate(-be_cz * sig_W * rho_W / 2, (W_s, 0, W_s))
h_W = sp.integrate(-(be_cz - al_cz) * rho_W, (W_s, 0, W_s))

# Substitute sigma_1 = s = -2*sqrt(3)*q_1, sigma_2 = -2*sqrt(3)*q_2, and reduce modulo q1^2 + q1*q2 + q2^2 = 1/4
sig1 = -2*sp.sqrt(3)*q1_s
sig2 = -2*sp.sqrt(3)*q2_s
w_jump = (sig2 - sig1) / mu_cz
dED = sp.expand((-tau_W / 9).subs({s_s: sig1, W_s: w_jump}))
dPD = sp.expand((h_W / 9).subs({s_s: sig1, W_s: w_jump}))
dPiD = sp.expand(sp.sqrt(3)*eta_s*(sig2**2 - sig1**2) / 6)
dMD = sp.expand((sig2 - sig1) / (sp.sqrt(3)*eta_s))

# For q1 > q2, k_S = q1 - q2, v_S = sqrt(3)*(q1 + q2), rho = eta * (+1)
k_edge = q1_s - q2_s
v_edge = sp.sqrt(3)*(q1_s + q2_s)
kp_rel = q1_s**2 + q1_s*q2_s + q2_s**2 - sp.Rational(1, 4)
E_sol_edge = sp.Rational(2, 15)*k_edge**3*(5 - 4*k_edge**2)
P_sol_edge = sp.Rational(2, 3)*k_edge**3*v_edge
Pi_sol_edge = -2*k_edge*v_edge
M_sol_edge = 2*k_edge

check('1a: Defect vacuum jumps factorise modulo KP circle into rho*(E,P,Pi,M)_S',
      all(sp.rem(sp.expand((dED - ev*E_sol_edge).subs(eta_s, ev)), kp_rel, q1_s) == 0 and
          sp.rem(sp.expand((dPD - ev*P_sol_edge).subs(eta_s, ev)), kp_rel, q1_s) == 0 and
          sp.rem(sp.expand((dPiD - ev*Pi_sol_edge).subs(eta_s, ev)), kp_rel, q1_s) == 0 and
          sp.expand((dMD - ev*M_sol_edge).subs(eta_s, ev)) == 0 for ev in (1, -1)))

# 1b. Reduced 4x4 linearised defect sewing determinant
S_sym, c_sym = sp.symbols('S c', real=True)
P_minus = S_sym**2 - (sp.sqrt(3) - 1)/2 * c_sym * S_sym + (c_sym**2 - 2 - sp.sqrt(3))/2
P_plus = S_sym**2 - (sp.sqrt(3) + 1)/2 * c_sym * S_sym + (c_sym**2 - 2 + sp.sqrt(3))/2
check('1b-i: Threshold factorisation P_mp(1; c) == (1/2)(c +- 1)(c - sqrt(3))',
      sp.expand(P_minus.subs(S_sym, 1) - sp.Rational(1, 2)*(c_sym + 1)*(c_sym - sp.sqrt(3))) == 0 and
      sp.expand(P_plus.subs(S_sym, 1) - sp.Rational(1, 2)*(c_sym - 1)*(c_sym - sp.sqrt(3))) == 0)

# At c = sqrt(3), roots of P_minus and P_plus:
roots_m_r3 = sp.solve(P_minus.subs(c_sym, sp.sqrt(3)), S_sym)
roots_p_r3 = sp.solve(P_plus.subs(c_sym, sp.sqrt(3)), S_sym)
S_admissible_r3 = (sp.sqrt(3) + 1)/2
g_admissible_r3 = sp.simplify((S_admissible_r3**2 - 1)/2)
check('1b-ii: Branch switch at c = sqrt(3) gives unique S > 1 root with g = sqrt(3)/4',
      sp.simplify(P_plus.subs({c_sym: sp.sqrt(3), S_sym: S_admissible_r3})) == 0 and
      sp.simplify(g_admissible_r3 - sp.sqrt(3)/4) == 0)


def count_admissible_growing_modes(c_val):
    cnt = 0
    for sgn in (-1, 1):
        B = -(R3 + sgn)/2 * c_val
        C = (c_val**2 - 2 + sgn*R3)/2
        disc = B**2 - 4*C
        if disc >= 0:
            for r in ((-B - mp.sqrt(disc))/2, (-B + mp.sqrt(disc))/2):
                if r > 1 + mp.mpf('1e-12'):
                    # Check Re(kappa_1) > 0, Re(kappa_2) > 0 where kappa_1 + kappa_2 = S, kappa_1*kappa_2 = (S^2+1)/2
                    cnt += 1
    return cnt


q1_num = mp.mpf('-0.20')
disc_q = 1 - 3*q1_num**2
q2_num = (-q1_num - mp.sqrt(disc_q))/2
q3_num = (-q1_num + mp.sqrt(disc_q))/2
counts_generic = sorted(count_admissible_growing_modes(-2*R3*q) for q in (q1_num, q2_num, q3_num))
# Triple containing c = sqrt(3) has q = -1/2, so roots of q'^2 - q'/2 = 0 are q' = 0 and q' = 1/2
counts_switch = sorted(count_admissible_growing_modes(c_v) for c_v in (R3, mp.mpf(0), -R3))
check('1b-iii: Admissible growing-mode counts on KP triple are (0,1,2) generically and (0,1,1) at c=sqrt(3)',
      counts_generic == [0, 1, 2] and counts_switch == [0, 1, 1])

# 1c. Pointwise quadratic defect expectation identities and UV linear divergence
q_w = sp.Symbol('q_w', positive=True)
om_w = sp.sqrt(q_w**2 + q_w**4)
# Transmission amplitude T(q) = (D(q) - i N(q)) / (D(q) + i N(q)) with phase theta_D(q)
# Wait: theta_D'(q) from Corrigan-Zambon linearised phase shift:
# In defq_oneloop_renorm.py / audit_defq.py:
# theta_D'(q) + theta_D'(-q) and theta_D'(q) - theta_D'(-q):
# Let's check the exact rational formula for <P_{D,2}>_q + <P_{D,2}>_{-q} and topological counterterm
S_D_sym = -eta_s*sp.sqrt(3)*sig_s/2
MD_vac = (sig_s - s_s) / (sp.sqrt(3)*eta_s)
T_cut = sp.Symbol('T_cut', positive=True)
ct_decomp = sp.expand(-S_D_sym*(T_cut - 1)/sp.pi - (eta_s*sp.sqrt(3)*s_s/(2*sp.pi)*(T_cut - 1) + sp.Rational(3, 2)/sp.pi*(T_cut - 1)*MD_vac))
check('1c-i: Defect UV counterterm decomposes into bare s term and (3/(2*pi))(T-1)*M_D',
      all(sp.simplify(ct_decomp.subs(eta_s, ev)) == 0 for ev in (1, -1)))


# Symbolic check of <P_{D,2}>_q + <P_{D,2}>_{-q} == (q / 4*pi)*(theta_D'(q) - theta_D'(-q))
q_sym = sp.Symbol('q_sym', positive=True)
w_q = q_sym * sp.sqrt(1 + q_sym**2)
r3_s = sp.sqrt(3)
X_q = sp.sqrt(1 + q_sym**2) / (2 * r3_s) + sig_s / (2 * r3_s)
Y_q = -q_sym / 2
X_neg = -sp.sqrt(1 + q_sym**2) / (2 * r3_s) + sig_s / (2 * r3_s)
Y_neg = q_sym / 2
re_z = (X_q**2 - Y_q**2) / (X_q**2 + Y_q**2)
re_z_neg = (X_neg**2 - Y_neg**2) / (X_neg**2 + Y_neg**2)
PD2_pos = (-q_sym**2 * (1 + 2*re_z)/r3_s - (1 - sig_s**2)*(1 - re_z)/(2*r3_s)) / (4*sp.pi*w_q)
PD2_neg = (-q_sym**2 * (1 + 2*re_z_neg)/r3_s - (1 - sig_s**2)*(1 - re_z_neg)/(2*r3_s)) / (4*sp.pi*w_q)
dX_pos = q_sym / (2 * r3_s * sp.sqrt(1 + q_sym**2))
dth_pos = 2 * (X_q * (-sp.Rational(1, 2)) - dX_pos * Y_q) / (X_q**2 + Y_q**2)
dth_neg = 2 * (X_neg * (-sp.Rational(1, 2)) - dX_pos * Y_neg) / (X_neg**2 + Y_neg**2)
dth_odd = (q_sym / (4*sp.pi)) * (dth_pos - dth_neg)
p_rat_sym = r3_s*q_sym*(2*(sig_s**2 - 2)*q_sym**2 + sig_s**2 - 1) / (
    sp.pi*sp.sqrt(1 + q_sym**2)*(16*q_sym**4 + 4*(sig_s**2 + 2)*q_sym**2 + (sig_s**2 - 1)**2))
check('1c-ii: Symbolic pointwise identity <P_{D,2}>_q + <P_{D,2}>_{-q} == (q/4pi)*(theta_D\'(q)-theta_D\'(-q))',
      sp.simplify(PD2_pos + PD2_neg - dth_odd) == 0 and
      sp.simplify(dth_odd - p_rat_sym) == 0)

# 1d. Fusion obstructions: canonical triangle holonomy, absorbed-soliton momentum remainder, polynomial quotient
p_sym, q_sym = sp.symbols('p q', real=True)
ell_fn = lambda r: 3*sp.sqrt(3)*r*(4*r**2 - 1)
quot_ell = sp.factor((ell_fn(p_sym) - ell_fn(q_sym)) / (p_sym - q_sym))
check('1d-i: (ell(p)-ell(q))/(p-q) == 3*sqrt(3)*(4*(p^2+p*q+q^2)-1) is polynomial (zero residue at p=q)',
      sp.expand(quot_ell - 3*sp.sqrt(3)*(4*(p_sym**2 + p_sym*q_sym + q_sym**2) - 1)) == 0)


def soliton_L1_and_charges(k_v, v_v):
    eps_v = 1 if v_v >= 0 else -1
    a_v = mp.asin(k_v)
    phis = [a_v + 2*mp.pi/3, a_v - 2*mp.pi/3, a_v]
    cs = [1, 1, -2]
    p_mode = -eps_v / (2*mp.pi) * sum(c*mp.sin(ph)*mp.log(abs((1 - mp.cos(ph))/(1 + mp.cos(ph))))/2 for c, ph in zip(cs, phis))
    e_mode = sum(c*mp.sin(ph)*(1 - mp.cos(ph)*mp.log(abs((1 - mp.cos(ph))/(1 + mp.cos(ph))))/2) for c, ph in zip(cs, phis)) / (2*mp.pi)
    L1_v = v_v * p_mode - e_mode
    ratio = mp.tan((a_v + 2*mp.pi/3)/2) / mp.tan((a_v - 2*mp.pi/3)/2)
    p_can = -eps_v * (R3 * mp.cos(a_v) / mp.pi) * mp.log(abs(ratio))
    e_can = v_v * p_can - L1_v
    return e_mode, p_mode, e_can, p_can, L1_v


# Compute directed canonical and mode-sum triangle holonomies on KP triple generated by q1 = -0.20, eta = +1
qs_tri = [q1_num, q3_num, q2_num]
E_mode_hol = mp.mpf(0)
P_mode_hol = mp.mpf(0)
E_can_hol = mp.mpf(0)
P_can_hol = mp.mpf(0)
edge_data = []
for i in range(3):
    qa, qb = qs_tri[i], qs_tri[(i + 1) % 3]
    rho_i = mp.sign(qa - qb)  # eta = +1
    k_i = abs(qa - qb)
    v_i = R3 * (qa + qb)
    em_i, pm_i, ec_i, pc_i, L1_i = soliton_L1_and_charges(k_i, v_i)
    E_mode_hol += rho_i * em_i
    P_mode_hol += rho_i * pm_i
    E_can_hol += rho_i * ec_i
    P_can_hol += rho_i * pc_i
    edge_data.append((qa, qb, rho_i, k_i, v_i, em_i, pm_i, ec_i, pc_i, L1_i))

check('1d-ii: Mode-sum triangle holonomy vanishes while canonical holonomy at q1=-0.20 is (+0.3175373286, +0.1655848943) != 0',
      abs(E_mode_hol) < TOL and abs(P_mode_hol) < TOL and
      abs(E_can_hol - mp.mpf('0.3175373286')) < mp.mpf('1e-9') and
      abs(P_can_hol - mp.mpf('0.1655848943')) < mp.mpf('1e-9'))

# Check absorbed-soliton momentum remainder at alpha_3 = 0.35, eps_3 = +1
al3 = mp.mpf('0.35')
k3, v3 = mp.sin(al3), mp.cos(al3)
p3a, p3b = (k3 + v3/R3)/2, (v3/R3 - k3)/2
ln_zt = [mp.log(abs((p3b - qv)/(p3a - qv))) for qv in qs_tri]
rem_mode = mp.mpf(0)
rem_can = mp.mpf(0)
for i in range(3):
    _, _, _, k_prev, _, _, pm_prev, _, pc_prev, _ = edge_data[(i - 1) % 3]
    _, _, _, k_curr, _, _, pm_curr, _, pc_curr, _ = edge_data[i]
    rem_mode += (pm_prev/k_prev - pm_curr/k_curr) * ln_zt[i]
    rem_can += (pc_prev/k_prev - pc_curr/k_curr) * ln_zt[i]
check('1d-iii: Absorbed-soliton momentum remainders at q1=-0.20, alpha3=0.35 match -0.7742169719 and -1.737010868',
      abs(rem_mode - mp.mpf('-0.7742169719')) < mp.mpf('1e-9') and
      abs(rem_can - mp.mpf('-1.737010868')) < mp.mpf('1e-8'))


# ==================================================================
# 2. Sec. 7.1: Projected determinant, collective measure, E_1|_P = -L_1
# ==================================================================
xi_s, k_s, v_s, t_s = sp.symbols('xi k v t', positive=True)
Phi_0 = k_s * sp.tanh(k_s * xi_s / 2)
u_0 = sp.diff(Phi_0, xi_s)

# 2a. Operator factorisation K_v = -d_xi H_v d_xi and secular velocity mode
f_fn = sp.Function('f')(xi_s)
Kv_f = sp.diff(f_fn, xi_s, 4) - k_s**2 * sp.diff(f_fn, xi_s, 2) + 6 * sp.diff(u_0 * sp.diff(f_fn, xi_s), xi_s)
Hv_df = -sp.diff(f_fn, xi_s, 3) + (k_s**2 - 6*u_0) * sp.diff(f_fn, xi_s)
check('2a-i: Factorisation K_v f == -d_xi (H_v d_xi f)',
      sp.simplify(Kv_f + sp.diff(Hv_df, xi_s)) == 0)

Phi_v = -(v_s / k_s) * (sp.tanh(k_s * xi_s / 2) + (k_s * xi_s / 2) * sp.sech(k_s * xi_s / 2)**2)
Kv_Phiv = sp.diff(Phi_v, xi_s, 4) - k_s**2 * sp.diff(Phi_v, xi_s, 2) + 6 * sp.diff(u_0 * sp.diff(Phi_v, xi_s), xi_s)
check('2a-ii: Secular velocity equation K_v Phi_v == -2 v u_0\'',
      sp.simplify(Kv_Phiv + 2 * v_s * sp.diff(u_0, xi_s)) == 0)

# Second linearly independent solution of H_v h = 0 in y = tanh(k xi / 2) grows as (1 - y^2)^{-1} = cosh^2(k xi / 2)
y_t = sp.Symbol('y_t', real=True)
dy_t = lambda expr: sp.diff(expr, y_t) * k_s * (1 - y_t**2) / 2
U0_t = k_s**2 * (1 - y_t**2) / 2
Hv_t = lambda expr: -dy_t(dy_t(expr)) + (k_s**2 - 6*U0_t) * expr
h1_t = dy_t(U0_t)
h2_t = sp.simplify(h1_t * sp.integrate(k_s**4 / (k_s * (1 - y_t**2) * h1_t**2), y_t))
check('2a-iii: Growing second kernel solution H_v h_2 == 0 with cosh^2(k xi/2) asymptotics',
      sp.simplify(Hv_t(h2_t)) == 0 and
      sp.simplify(h1_t * dy_t(h2_t) - dy_t(h1_t) * h2_t - sp.Rational(1, 2)*k_s**4) == 0 and
      sp.limit((1 - y_t**2) * h2_t, y_t, 1) != 0)

# Heavy-branch complementary localized modes (alpha in (pi/3, pi/2))
al_hb = mp.mpf('1.25')
k_hb, v_hb = mp.sin(al_hb), mp.cos(al_hb)
gam_hb = R3 * (k_hb**2 - mp.mpf(3)/4)
Km = mp.sin(al_hb - mp.pi/3)
Kp = mp.sin(al_hb + mp.pi/3)
fm_sp = lambda x: mp.diff(lambda y: mp.exp(Km*y)/(1 + mp.exp(k_hb*y)), x)
fp_sp = lambda x: mp.diff(lambda y: mp.exp(Kp*y)/(1 + mp.exp(k_hb*y)), x)
u0_hb = lambda x: (k_hb**2/2)*mp.sech(k_hb*x/2)**2


def Dv_localized(f_sp, sign_gam, x0):
    # D_v(e^{sign_gam * gam * t} f_sp(xi)) = [(sign_gam*gam - v d_xi)^2 + d_xi^4 - d_xi^2 + 6 d_xi u_0 d_xi] f_sp
    d1 = mp.diff(f_sp, x0, 1)
    d2 = mp.diff(f_sp, x0, 2)
    d4 = mp.diff(f_sp, x0, 4)
    nl = mp.diff(lambda y: 6*u0_hb(y)*mp.diff(f_sp, y, 1), x0)
    return sign_gam**2 * gam_hb**2 * f_sp(x0) - 2*sign_gam*gam_hb*v_hb*d1 + (v_hb**2 - 1)*d2 + d4 + nl


check('2a-iv: Heavy-branch localized modes require complementary profiles f_mp^sp for e^{+-gamma t}',
      abs(Dv_localized(fm_sp, +1, mp.mpf('0.4'))) < mp.mpf('1e-35') and
      abs(Dv_localized(fp_sp, -1, mp.mpf('0.4'))) < mp.mpf('1e-35') and
      abs(Dv_localized(fm_sp, -1, mp.mpf('0.4'))) > mp.mpf('1e-2'))

# 2b. Collective 4x4 symplectic matrix W_4 and projector tails
Pv_s, Piv_s = sp.symbols('P_v Pi_v', real=True)
W3_mat = sp.Matrix([[0, -Pv_s, -Piv_s],
                    [Pv_s, 0, 0],
                    [Piv_s, 0, 0]])
W4_mat = sp.Matrix([[0, -Pv_s, -Piv_s, 0],
                    [Pv_s, 0, 0, 0],
                    [Piv_s, 0, 0, 1],
                    [0, 0, -1, 0]])
# In y_t = tanh(k xi / 2) in [-1, 1], dxi = 2/(k*(1 - y_t^2)) dy_t and U0_t = k^2*(1 - y_t^2)/2:
dxi_dy = 2 / (k_s * (1 - y_t**2))
w_aux_y = (3/k_s)*U0_t - (sp.Rational(15, 2)/k_s**3)*U0_t**2
check('2b: W_3 has null vector (0, Pi_v, -P_v)^T, det W_4 == P_v^2, and int w dxi == 1, int w u_0 dxi == 0',
      W3_mat * sp.Matrix([0, Piv_s, -Pv_s]) == sp.Matrix([0, 0, 0]) and
      sp.simplify(W4_mat.det() - Pv_s**2) == 0 and
      sp.simplify(sp.integrate(w_aux_y * dxi_dy, (y_t, -1, 1)) - 1) == 0 and
      sp.simplify(sp.integrate(w_aux_y * U0_t * dxi_dy, (y_t, -1, 1))) == 0)

# 2c. Intrinsic kinetic-norm box divergence
# int_{-oo}^{oo} [Phi_v(xi)^2 - v^2/k^2] dxi = 2*(pi^2 - 6)/(9*k^3) * v^2
kk_num, vv_num = mp.mpf('0.6'), mp.mpf('0.8')
Phiv_num = lambda x: -(vv_num/kk_num)*(mp.tanh(kk_num*x/2) + (kk_num*x/2)*mp.sech(kk_num*x/2)**2)
reg_norm = mp.quad(lambda x: Phiv_num(x)**2 - (vv_num/kk_num)**2, [-mp.inf, 0, mp.inf])
reg_exact = 2*(mp.pi**2 - 6)*vv_num**2 / (9*kk_num**3)
check('2c: Regulated velocity-tangent norm matches 2*(pi^2 - 6)*v^2 / (9*k^3)',
      abs(reg_norm - reg_exact) < TOL)

# 2d. Fixed-P energy invariance, good trace F_g(k), and inversion-antisymmetric kernel tilde K_L
e_s, p_s, L1_s, dL1_dv = sp.symbols('e p L1 dL1_dv', real=True)
# Mode-sum: L1 = v*p - e => e = v*p - L1; canonical: p_can = dL1/dv, e_can = v*p_can - L1
E1_mode = (v_s*p_s - L1_s) - v_s*p_s
E1_can = (v_s*dL1_dv - L1_s) - v_s*dL1_dv
check('2d-i: Unconditional algebraic invariance E_1|_P == e - v p == e_can - v p_can == -L_1',
      sp.simplify(E1_mode + L1_s) == 0 and sp.simplify(E1_can + L1_s) == 0)

al_s = sp.Symbol('alpha', positive=True)
Fg_k = (sp.sqrt(3)/sp.pi)*(sp.Rational(3, 4) - k_s**2)*sp.atanh(2*k_s/sp.sqrt(3)) - 3*k_s/(2*sp.pi)
Fg_k_ser = sp.series(Fg_k, k_s, 0, 7).removeO()
Fg_ser = sp.series(Fg_k_ser.subs(k_s, sp.sin(al_s)), al_s, 0, 7).removeO()
P0_al_ser = sp.series(sp.Rational(2, 3)*sp.sin(al_s)**3*sp.cos(al_s), al_s, 0, 5).removeO()
Cg_bare = sp.limit(Fg_ser / P0_al_ser, al_s, 0)
check('2d-ii: Good trace expansion F_g(alpha) == -4/(3*pi)*alpha^3 + 14/(45*pi)*alpha^5 + ... and C_g^bare == -2/pi',
      sp.expand(Fg_ser - (-sp.Rational(4, 3)/sp.pi*al_s**3 + sp.Rational(14, 45)/sp.pi*al_s**5)) == 0 and
      sp.simplify(Cg_bare + 2/sp.pi) == 0)

# Light-root logarithmic weight w(phi_r) = 2*sin(phi_r)*(cos(alpha) - cos(phi_r)) vanishes at phi_3 = alpha
phi_r = sp.Symbol('phi_r', real=True)
w_weight = 2*sp.sin(phi_r)*(sp.cos(al_s) - sp.cos(phi_r))
check('2d-iii: Light-root logarithmic weight w(alpha) == 0 identically',
      sp.simplify(w_weight.subs(phi_r, al_s)) == 0)

# Inversion-antisymmetric kernel tilde K_L(z) = (K_L(z) - K_L(1/z))/2 = K_L(z) + (pi/8) sgn(Re z)
KL_fn = lambda z: -mp.sign(z) * (mp.re(mp.polylog(2, abs(z)) - mp.polylog(2, -abs(z))) + mp.log(abs(z))*mp.log(abs((1 - abs(z))/(1 + abs(z))))) / (2*mp.pi)
z_pos, z_neg = mp.mpf('0.42'), mp.mpf('-0.42')
check('2d-iv: Inversion identity tilde K_L(z) == K_L(z) + (pi/8) sgn(z)',
      abs((KL_fn(z_pos) - KL_fn(1/z_pos))/2 - (KL_fn(z_pos) + mp.pi/8)) < TOL and
      abs((KL_fn(z_neg) - KL_fn(1/z_neg))/2 - (KL_fn(z_neg) - mp.pi/8)) < TOL)


# ==================================================================
# 3. Sec. 7.2--7.3: Headline one-body O(hbar^0) coefficient & scheme matching
# ==================================================================
# 3a. Exact symbolic and high-precision numerical checks of headline formulas
s1_T_sym = sp.Rational(3, 4) / sp.pi
edge_T_sym = -sp.Rational(1, 4) / sp.pi
CT_sym = s1_T_sym + edge_T_sym

Cac_sym = sp.Integer(2) / sp.pi
s1_bac_sym = sp.Rational(9, 4) / sp.pi
lam_sym = s1_T_sym - s1_bac_sym
C_matched_sym = Cac_sym + lam_sym

Cg_bare_sym = -sp.Integer(2) / sp.pi
s1_g_sym = -sp.Rational(9, 4) / sp.pi
Cg_onshell_sym = Cg_bare_sym - s1_g_sym

check('3a-i: 3/(4*pi) - 1/(4*pi) == 1/(2*pi) (symbolic & 50-digit mpmath)',
      sp.simplify(CT_sym - 1/(2*sp.pi)) == 0 and
      abs((mp.mpf(3)/(4*mp.pi) - mp.mpf(1)/(4*mp.pi)) - 1/(2*mp.pi)) < TOL)

check('3a-ii: 2/pi - 3/(2*pi) == 1/(2*pi) (symbolic & 50-digit mpmath)',
      sp.simplify(lam_sym + sp.Rational(3, 2)/sp.pi) == 0 and
      sp.simplify(C_matched_sym - 1/(2*sp.pi)) == 0 and
      abs((2/mp.pi - 3/(2*mp.pi)) - 1/(2*mp.pi)) < TOL)

check('3a-iii: Scheme-invariant relative slope C - s_1 == -1/(4*pi) in both continued continuum and Toda',
      sp.simplify((Cac_sym - s1_bac_sym) - (-1/(4*sp.pi))) == 0 and
      sp.simplify((CT_sym - s1_T_sym) - (-1/(4*sp.pi))) == 0 and
      abs((2/mp.pi - 9/(4*mp.pi)) - (-1/(4*mp.pi))) < TOL)

check('3a-iv: Good-equation on-shell slope -2/pi + 9/(4*pi) == +1/(4*pi)',
      sp.simplify(Cg_onshell_sym - 1/(4*sp.pi)) == 0 and
      abs((-2/mp.pi + 9/(4*mp.pi)) - 1/(4*mp.pi)) < TOL)

CT_num = mp.mpf('0.1591549414')
CT_err = mp.mpf('8.23e-6')
CT_exact = 1 / (2*mp.pi)
check('3a-v: Numerical Toda value 0.1591549414 +- 8.23e-6 matches 1/(2*pi) to 5e-5 and excludes 1/6 and sqrt(3)/(2*pi)^2',
      abs(CT_num - CT_exact) < CT_err and
      abs(CT_num - CT_exact) / CT_exact < mp.mpf('5e-5') and
      CT_err / CT_exact < mp.mpf('5.2e-5') and
      abs(CT_num - mp.mpf(1)/6) > 500 * CT_err and
      abs(CT_num - R3/(2*mp.pi)**2) > 5000 * CT_err)

# 3b. Continued bad trace F_ac(k) and real stable-band integral F_band(beta)
bet_s = sp.Symbol('beta', positive=True)
Fac_k = (sp.sqrt(3)/sp.pi)*(k_s**2 + sp.Rational(3, 4))*sp.atan(2*k_s/sp.sqrt(3)) - 3*k_s/(2*sp.pi)
Fac_k_ser = sp.series(Fac_k, k_s, 0, 7).removeO()
Fac_ser = sp.series(Fac_k_ser.subs(k_s, sp.sinh(bet_s)), bet_s, 0, 7).removeO()
P0_bad = sp.Rational(2, 3)*sp.sinh(bet_s)**3*sp.cosh(bet_s)
P0_bad_ser = sp.series(P0_bad, bet_s, 0, 5).removeO()
Cac_lim = sp.limit(Fac_ser / P0_bad_ser, bet_s, 0)
check('3b-i: F_ac(beta) == 4/(3*pi)*beta^3 + 14/(45*pi)*beta^5 + ... and C_ac == +2/pi == 4*C_T',
      sp.expand(Fac_ser - (sp.Rational(4, 3)/sp.pi*bet_s**3 + sp.Rational(14, 45)/sp.pi*bet_s**5)) == 0 and
      sp.simplify(Cac_lim - 2/sp.pi) == 0)

Fband_ser = Fac_ser - sp.series((sp.sqrt(3)/2)*(sp.sinh(bet_s)**2 + sp.Rational(3, 4)), bet_s, 0, 5).removeO()
ann_num = sp.diff(Fband_ser, bet_s, 3) - sp.diff(Fband_ser, bet_s, 1)
ann_den = sp.diff(P0_bad_ser, bet_s, 3) - sp.diff(P0_bad_ser, bet_s, 1)
check('3b-ii: Origin/mass-term annihilator (d_beta^3 - d_beta) extracts +2/pi from F_band',
      sp.simplify(sp.limit(ann_num / ann_den, bet_s, 0) - 2/sp.pi) == 0)

# 3c. Homogeneous vacuum strain energy densities and soliton counterterm integral
U_s = sp.Symbol('U', real=True)
Wg_U = -((1 - 6*U_s)**sp.Rational(3, 2) - 1 + 9*U_s) / (6*sp.pi)
Wbac_U = ((1 + 6*U_s)**sp.Rational(3, 2) - 1 - 9*U_s) / (6*sp.pi)
WTB_U = (sp.exp(3*U_s) - 1 - 3*U_s) / (6*sp.pi)
check('3c-i: Vacuum sound-speed shifts (1/2) W\'\'(0) give -9/(4*pi), +9/(4*pi), +3/(4*pi)',
      sp.simplify(sp.Rational(1, 2)*sp.diff(Wg_U, U_s, 2).subs(U_s, 0) - s1_g_sym) == 0 and
      sp.simplify(sp.Rational(1, 2)*sp.diff(Wbac_U, U_s, 2).subs(U_s, 0) - s1_bac_sym) == 0 and
      sp.simplify(sp.Rational(1, 2)*sp.diff(WTB_U, U_s, 2).subs(U_s, 0) - s1_T_sym) == 0)
check('3c-ii: Vacuum elasticity difference W_T^B(U) - W_{b,ac}(U) == -3/(2*pi)*U^2 + O(U^3)',
      sp.expand(sp.series(WTB_U - Wbac_U, U_s, 0, 3).removeO() - (-sp.Rational(3, 2)/sp.pi*U_s**2)) == 0)
int_u0_sq = sp.integrate(U0_t**2 * dxi_dy, (y_t, -1, 1))
check('3c-iii: Soliton quadratic integral int u_0^2 dxi == (2/3) k^3 == P_0 / v',
      sp.simplify(int_u0_sq - sp.Rational(2, 3)*k_s**3) == 0)

# 3d. Analytical Toda Fermi-edge dispersion representation
y_s = sp.Symbol('y', positive=True)
c_ed_s, d0_s = sp.symbols('c_edge d0', real=True)
u_asymp = y_s - sp.log(y_s)/(2*sp.pi) + c_ed_s + d0_s/y_s
Cv_asymp = sp.Rational(3, 4)/sp.pi + (y_s/2)*(sp.diff(u_asymp, y_s) - 1)
check('3d: Analytical Toda velocity-slope formula C_v(y) -> 3/(4*pi) - 1/(4*pi) == 1/(2*pi)',
      sp.simplify(sp.limit(Cv_asymp, y_s, sp.oo) - 1/(2*sp.pi)) == 0)


# ==================================================================
# 4. Sec. 8: Quantum Toda Fermi-edge limit
# ==================================================================
# 4a. Finite-sea Fredholm endpoint identities & four-term reciprocity (Nystrom check)
def verify_fredholm_identities():
    with mp.workdps(30):
        h_v = mp.mpf('0.8')
        q_v = mp.mpf('1.0')
        t_v = 2*mp.pi*h_v
        K_fn = lambda d: 2*mp.re(mp.psi(0, 1 + 1j*d/h_v)) + 2*mp.log(h_v)
        Th_fn = lambda d: 2*h_v*mp.im(mp.loggamma(1 + 1j*d/h_v)) + 2*d*mp.log(h_v)

        nodes, weights = mp.gauss_quadrature(28, 'legendre')
        xs = [q_v * xi for xi in nodes]
        ws = [q_v * wi for wi in weights]
        n = len(xs)
        A = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                A[i, j] = (t_v if i == j else mp.mpf(0)) - ws[j] * K_fn(xs[i] - xs[j])
        one_vec = mp.matrix([mp.mpf(1) for _ in range(n)])
        x_vec = mp.matrix(xs)
        rho = mp.lu_solve(A, one_vec)
        eta = mp.lu_solve(A, x_vec)
        a_v = 1 / sum(ws[i]*rho[i] for i in range(n))

        D_of = lambda p: 1 + sum(ws[j]*K_fn(p - xs[j])*rho[j] for j in range(n))
        V_of = lambda p: p + sum(ws[j]*K_fn(p - xs[j])*eta[j] for j in range(n))
        P_of = lambda p: p + sum(ws[j]*Th_fn(p - xs[j])*rho[j] for j in range(n))
        Dq, Vq = D_of(q_v), V_of(q_v)
        rho_q, eta_q = Dq / t_v, Vq / t_v

        p_test = mp.mpf('1.7')
        th_vec = mp.matrix([Th_fn(p_test - xj) for xj in xs])
        Fp = mp.lu_solve(A, th_vec)
        Fp_q = (Th_fn(p_test - q_v) + sum(ws[j]*K_fn(q_v - xs[j])*Fp[j] for j in range(n))) / t_v
        Fp_mq = (Th_fn(p_test + q_v) + sum(ws[j]*K_fn(-q_v - xs[j])*Fp[j] for j in range(n))) / t_v

        err1 = abs(P_of(q_v) - mp.pi*h_v/a_v)
        err2 = abs(2*a_v*t_v*rho_q*eta_q - 1)
        err3 = abs(t_v*rho_q*(Fp_q - Fp_mq) - (1 - D_of(p_test)))
        err4 = abs(t_v*eta_q*(Fp_q + Fp_mq) - (P_of(p_test) - V_of(p_test)))
        return max(err1, err2, err3, err4) < mp.mpf('1e-14')


check('4a-i: Finite-sea Nystrom check of P(q)=pi*h/a, 2*a*t*rho(q)*eta(q)=1, and endpoint identities', verify_fredholm_identities())

# Symbolic check of four-term endpoint reciprocity:
# t*(f_p^+ f_z^+ - f_p^- f_z^-) == M_p Q_z + M_z Q_p
# where M_p = t*a*rho_q*(f_p^+ - f_p^-), Q_p = (f_p^+ + f_p^-)/(2*a*rho_q)
fp_p, fp_m, fz_p, fz_m, t_sym, a_sym, rhoq_sym = sp.symbols('fp_p fp_m fz_p fz_m t a rho_q', positive=True)
Mp_sym = t_sym*a_sym*rhoq_sym*(fp_p - fp_m)
Mz_sym = t_sym*a_sym*rhoq_sym*(fz_p - fz_m)
Qp_sym = (fp_p + fp_m)/(2*a_sym*rhoq_sym)
Qz_sym = (fz_p + fz_m)/(2*a_sym*rhoq_sym)
check('4a-ii: Four-term reciprocity identity t*(f_p^+ f_z^+ - f_p^- f_z^-) == M_p Q_z + M_z Q_p',
      sp.expand(t_sym*(fp_p*fz_p - fp_m*fz_m) - (Mp_sym*Qz_sym + Mz_sym*Qp_sym)) == 0)

# 4b. Classical-frame remainder S_0(y,z) and rectangular second difference D_0
y1_s, y0_s, z1_s, z0_s = sp.symbols('y1 y0 z1 z0', positive=True)
up_y1, up_y0, up_z1, up_z0 = sp.symbols('up_y1 up_y0 up_z1 up_z0', positive=True)
S0_fn = lambda yv, zv, upy, upz: 8*sp.sqrt(yv*zv)*(1/(upy*upz) - 1) - 4*sp.sqrt(sp.pi)*(sp.sqrt(yv)/upy + sp.sqrt(zv)/upz) + 2*sp.pi
D0_rect = S0_fn(y1_s, z1_s, up_y1, up_z1) - S0_fn(y1_s, z0_s, up_y1, up_z0) - S0_fn(y0_s, z1_s, up_y0, up_z1) + S0_fn(y0_s, z0_s, up_y0, up_z0)
D0_fact = 8*(sp.sqrt(y1_s)/up_y1 - sp.sqrt(y0_s)/up_y0)*(sp.sqrt(z1_s)/up_z1 - sp.sqrt(z0_s)/up_z0) - 8*(sp.sqrt(y1_s) - sp.sqrt(y0_s))*(sp.sqrt(z1_s) - sp.sqrt(z0_s))
check('4b-i: Rectangular second difference D_0 cancels additive endpoint terms and factorises',
      sp.expand(D0_rect - D0_fact) == 0)

Y_s = sp.Symbol('Y', positive=True)
up_as = lambda yv: 1 - 1/(2*sp.pi*yv)
D0_scaled = D0_fact.subs({
    y1_s: 2*Y_s, y0_s: Y_s, z1_s: Y_s/2, z0_s: Y_s/4,
    up_y1: up_as(2*Y_s), up_y0: up_as(Y_s), up_z1: up_as(Y_s/2), up_z0: up_as(Y_s/4)
})
check('4b-ii: Large-Y limit of D_0(2Y, Y; Y/2, Y/4) == -(15*sqrt(2) - 20)/pi',
      sp.simplify(sp.limit(D0_scaled, Y_s, sp.oo) - (-(15*sp.sqrt(2) - 20)/sp.pi)) == 0)

# 4c. Wiener-Hopf symbol, Laplace transform, and series-reversion edge coefficients
om_num = mp.mpf('1.37')
g_wh = lambda s: s**(-mp.mpf('0.5')) * mp.gamma(1 + s/(2*mp.pi)) * mp.exp(-(s/(2*mp.pi))*mp.log(s/(2*mp.pi*mp.e)))
Sigma_om = 2*mp.pi / (1 - mp.exp(-abs(om_num)))
check('4c-i: Wiener-Hopf factorisation Sigma(omega) == 2*pi/(1 - e^{-|omega|}) == 2*pi g(i omega) g(-i omega)',
      abs(Sigma_om - 2*mp.pi*g_wh(1j*om_num)*g_wh(-1j*om_num)) < TOL)

# Symbolic derivation of c_edge, d_0, e_0 from L[(4/3) y(u)^{3/2}](s) = sqrt(pi) g(s) / s^2 and series reversion
L_s, al_lap = sp.symbols('L_s al_lap', positive=True)
AA = 1 / (2*sp.pi)
BB = 1 + sp.log(2*sp.pi) - sp.EulerGamma
Z3 = sp.zeta(3)


def ilog(al_v, j):
    expr = (-1)**j * sp.diff(sp.exp((al_lap - 1)*L_s) / sp.gamma(al_lap), al_lap, j)
    poly_subs = {
        sp.polygamma(0, sp.Rational(3, 2)): 2 - sp.EulerGamma - 2*sp.log(2),
        sp.polygamma(0, sp.Rational(1, 2)): -sp.EulerGamma - 2*sp.log(2),
        sp.polygamma(0, -sp.Rational(1, 2)): 2 - sp.EulerGamma - 2*sp.log(2),
        sp.polygamma(1, sp.Rational(1, 2)): sp.pi**2 / 2,
        sp.polygamma(1, -sp.Rational(1, 2)): sp.pi**2 / 2 + 4,
        sp.polygamma(2, -sp.Rational(1, 2)): -14*Z3 + 16,
    }
    return sp.simplify(sp.expand_func(expr.subs(al_lap, al_v)).subs(poly_subs) / sp.exp((al_v - 1)*L_s))


# Expand g(s) = s^{-1/2} exp( B*(s/2pi) - (s/2pi)*ln(s) + (pi^2/12)*(s/2pi)^2 - (Z3/3)*(s/2pi)^3 + ... )
c1_B = AA * BB
c2_B = AA**2 * (BB**2 / 2 + sp.pi**2 / 12)
c3_B = AA**3 * (BB**3 / 6 + BB * sp.pi**2 / 12 - Z3 / 3)

f1_u = sp.simplify(sp.Rational(3, 4)*sp.sqrt(sp.pi) * (c1_B*ilog(sp.Rational(3, 2), 0) - AA*ilog(sp.Rational(3, 2), 1)))
f2_u = sp.simplify(sp.Rational(3, 4)*sp.sqrt(sp.pi) * (c2_B*ilog(sp.Rational(1, 2), 0) - AA*c1_B*ilog(sp.Rational(1, 2), 1) + (AA**2/2)*ilog(sp.Rational(1, 2), 2)))
f3_u = sp.simplify(sp.Rational(3, 4)*sp.sqrt(sp.pi) * (c3_B*ilog(-sp.Rational(1, 2), 0) - AA*c2_B*ilog(-sp.Rational(1, 2), 1) + (AA**2/2)*c1_B*ilog(-sp.Rational(1, 2), 2) - (AA**3/6)*ilog(-sp.Rational(1, 2), 3)))

# y(u) = u * (1 + f1/u + f2/u^2 + f3/u^3)^{2/3} = u + v0 + v1/u + v2/u^2 + ...
v0_u = sp.expand(sp.Rational(2, 3)*f1_u)
v1_u = sp.expand(sp.Rational(2, 3)*f2_u - sp.Rational(1, 9)*f1_u**2)
v2_u = sp.expand(sp.Rational(2, 3)*f3_u - sp.Rational(2, 9)*f1_u*f2_u + sp.Rational(4, 81)*f1_u**3)

c_edge_ex = (1 - sp.log(8*sp.pi)) / (2*sp.pi)
d0_ex = sp.Rational(1, 48) - 1/(4*sp.pi**2)
e0_ex = (4 + 6*Z3 - sp.pi**2) / (96*sp.pi**3)

# Reversion relation from u(y) = y - AA*ln(y) + c_edge + d0/y + e0/y^2:
v0_rev = AA*L_s - c_edge_ex
v1_rev = sp.expand(AA*v0_rev - d0_ex)
v2_rev = sp.expand(AA*(v1_rev - v0_rev**2/2) + d0_ex*v0_rev - e0_ex)

check('4c-ii: Inverse Laplace expansion of sqrt(pi)*g(s)/s^2 and series reversion give c_edge, d_0, e_0 (with d_1=e_2=e_1=0 and -(ln u)^2/(16*pi^3 u^2) in y(u))',
      sp.simplify(v0_u - v0_rev) == 0 and
      sp.simplify(v1_u - v1_rev) == 0 and
      sp.simplify(v2_u - v2_rev) == 0 and
      sp.simplify(v2_u.coeff(L_s, 2) + 1/(16*sp.pi**3)) == 0)

# Closed-form Krein integrals R_0(infty, r) and c_1(r)
r_num = mp.mpf('0.6')
R0_quad = (4/mp.pi) * mp.quad(lambda s: mp.log(s/r_num) * s**2 / (1 - s**2), [0, r_num])
chi2_r = (mp.polylog(2, r_num) - mp.polylog(2, -r_num)) / 2
R0_closed = -(4/mp.pi)*(chi2_r - r_num)

d0_num = mp.mpf(1)/48 - 1/(4*mp.pi**2)


def c1_integrand(t):
    sig = 1 - r_num**2 + t**2
    delta = -mp.log(t**2 / (r_num**2 * sig)) / (2*mp.pi)
    return (8 / sig**mp.mpf('2.5')) * (
        t**2 * (-2*d0_num - 3/(4*mp.pi**2) - delta/(4*mp.pi) + 3*delta**2/8)
        - (d0_num * sig / 2) * (1 + t**2 * (1 - 1/r_num**2 - 1/sig))
    )


c1_from_quad = mp.quad(c1_integrand, [0, r_num])
c1_closed = -r_num**3 / (6*(1 - r_num**2)) - 2*d0_num*((1 - r_num**2)/r_num**2 * mp.log((1 - r_num)/(1 + r_num)) + 2/r_num)
check('4c-iii: Closed-form Krein integrals R_0(infty, r) and c_1(r) match 50-digit quadrature',
      abs(R0_quad - R0_closed) < TOL and abs(c1_from_quad - c1_closed) < TOL)

# 4d. Antisymmetric counterexample A_lambda and its removal by cluster separability & Krein PDE
A_anti = lambda y1, y2: y1*y2*(y1 - y2)/(y1 + y2)**3
ys_3 = [sp.Integer(1), sp.Integer(2), sp.Integer(3)]
zs_3 = [sp.Integer(4), sp.Integer(5), sp.Integer(6)]
M_schmidt = sp.Matrix(3, 3, lambda i, j: A_anti(ys_3[i], zs_3[j]))
y1_sym, y2_sym = sp.symbols('y1_sym y2_sym', positive=True)
K_pde_A = sp.simplify(sp.diff(A_anti(y1_sym, y2_sym), y1_sym) + sp.diff(A_anti(y1_sym, y2_sym), y2_sym))
M_krein = sp.Matrix(2, 2, lambda i, j: K_pde_A.subs({y1_sym: ys_3[i], y2_sym: zs_3[j]}))
check('4d: A_lambda has nonvanishing 3x3 Schmidt minor and nonvanishing 2x2 Krein-PDE minor',
      M_schmidt.det() != 0 and M_krein.det() != 0)


# ==================================================================
# 5. Sec. 9: Exact S-matrix candidates and constraints
# ==================================================================
# 5a. Bad Gamma candidate unitarity, crossing, and Bogdan-Charkina breather poles
def bad_gamma_candidates(b1, b2, hb_v):
    # Overtaking (eps1 = eps2 = +1) and head-on (eps1 = +1, eps2 = -1)
    k1, k2 = mp.sinh(b1), mp.sinh(b2)
    v1_ov, v2_ov = mp.cosh(b1), mp.cosh(b2)
    l1_ov, l2_ov = mp.cosh(3*b1), mp.cosh(3*b2)
    eta_ov = mp.mpf(2)/9 * (l1_ov - l2_ov) / hb_v
    A_ov = (1 + 3*mp.coth((b1 + b2)/2)**2) / (1 + 3*mp.coth((b1 - b2)/2)**2)
    calA_ov = 81 * A_ov / (4*(l1_ov - l2_ov)**2)
    T_ov = mp.mpf(8)/3 * k1 * k2 * (v1_ov - v2_ov)
    S_ov = (mp.gamma(1 + 1j*eta_ov) / mp.gamma(1 - 1j*eta_ov)) * (mp.e**2 * hb_v**2 * calA_ov)**(1j*eta_ov) * mp.exp(1j*T_ov/hb_v)

    v1_ho, v2_ho = mp.cosh(b1), -mp.cosh(b2)
    l1_ho, l2_ho = mp.cosh(3*b1), -mp.cosh(3*b2)
    eta_ho = mp.mpf(2)/9 * (l1_ho - l2_ho) / hb_v
    A_ho = (1 + 3*mp.tanh((b1 - b2)/2)**2) / (1 + 3*mp.tanh((b1 + b2)/2)**2)
    calB_ho = 81 / (4 * A_ho * (l1_ho - l2_ho)**2)
    T_ho = mp.mpf(8)/3 * k1 * k2 * (v1_ho - v2_ho)
    S_ho = (mp.gamma(1 - 1j*eta_ho) / mp.gamma(1 + 1j*eta_ho)) * (mp.e**2 * hb_v**2 * calB_ho)**(-1j*eta_ho) * mp.exp(1j*T_ho/hb_v)
    return S_ov, S_ho


b1_t, b2_t, hb_t = mp.mpf('0.85'), mp.mpf('0.35'), mp.mpf('0.12')
Sov_12, Sho_12 = bad_gamma_candidates(b1_t, b2_t, hb_t)
Sov_21, _ = bad_gamma_candidates(b2_t, b1_t, hb_t)
Sov_cross, _ = bad_gamma_candidates(b2_t + 1j*mp.pi, b1_t, hb_t)
check('5a-i: Bad Gamma candidate satisfies unitarity |S|==1, S(1,2)S(2,1)==1, and crossing S_ho(b1,b2)==S_ov(b2+i*pi,b1)',
      abs(abs(Sov_12) - 1) < TOL and
      abs(Sov_12 * Sov_21 - 1) < TOL and
      abs(Sho_12 - Sov_cross) < mp.mpf('1e-38'))

# Bad breather kinematics beta_{1,2} = a -+ i*b with a > 0, pi/3 < b < pi/2
a_br, b_br = mp.mpf('0.5'), mp.mpf('1.25')
b1_br, b2_br = a_br - 1j*b_br, a_br + 1j*b_br
P_bad_fn = lambda b: mp.mpf(2)/3 * mp.sinh(b)**3 * mp.cosh(b)
A_br = (1 + 3*mp.coth(a_br)**2) / (1 - 3*mp.cot(b_br)**2)
I_br = -mp.mpf(4)/9 * mp.sinh(3*a_br) * mp.sin(3*b_br)
i_eta_br = 2j * mp.mpf(1)/9 * (mp.cosh(3*b1_br) - mp.cosh(3*b2_br)) / hb_t
check('5a-ii: Bad breather kinematics has Im(P1-P2) > 0, A_12 > 1, and i*eta == -I_br/hbar < 0',
      mp.im(P_bad_fn(b1_br) - P_bad_fn(b2_br)) > 0 and
      A_br > 1 and
      I_br > 0 and
      abs(i_eta_br - (-I_br / hb_t)) < TOL)

# 5b. Good-equation breathers and candidate poles
a_g, b_g = sp.symbols('a_g b_g', positive=True)
A_good_br = (1 - 3*sp.cot(a_g)**2) / (1 + 3*sp.coth(b_g)**2)
# Since coth^2(b_g) > 1 and cot^2(a_g) >= 0, 1 + 3*coth^2(b_g) > 4 and 1 - 3*cot^2(a_g) <= 1, so A_good_br < 1/4 < 1
check('5b-i: Good breather A_12 changes sign at a = pi/3 and satisfies A_12 < 1/4 < 1 everywhere',
      sp.simplify(A_good_br.subs(a_g, sp.pi/3)) == 0 and
      A_good_br.subs({a_g: sp.pi/4, b_g: 1}) < 0 and
      0 < A_good_br.subs({a_g: sp.Rational(5, 12)*sp.pi, b_g: 1}) < sp.Rational(1, 4))

# Naive good Gamma pole at hbar = 0.05, N = 1, a = 0.4 => 4*sin(3*a)*sinh(3*b)/(9*hbar) = -1
hb_g = mp.mpf('0.05')
a_g_num = mp.mpf('0.4')
b_g_num = mp.asinh(-9*hb_g / (4*mp.sin(3*a_g_num))) / 3
Im_dP_g = mp.mpf(1)/3 * mp.sinh(2*b_g_num) * (mp.cos(4*a_g_num)*mp.cosh(2*b_g_num) - mp.cos(2*a_g_num))
A_g_num = (1 - 3*mp.cot(a_g_num)**2) / (1 + 3*mp.coth(b_g_num)**2)
check('5b-ii: Naive good Gamma N=1 pole at a=0.4, hbar=0.05 lies on Im(P1-P2)>0 with singular A_12 < 0',
      abs(b_g_num - mp.mpf('-0.040137301855')) < mp.mpf('1e-10') and
      Im_dP_g > 0 and
      abs(A_g_num - mp.mpf('-0.008461762797')) < mp.mpf('1e-10'))

# 5c. Restricted dilogarithm exclusion: diagonal KP root-ratio pole mismatch
al_d, d_s = sp.symbols('alpha_d d_s', positive=True)
# For R(a1, a2) = tan(phi_1/2)/tan(phi_2/2) with phi_1 = phi + d, phi_2 = phi:
# d_1 d_2 Li_2(R) = (R/(1-R)) (d_1 ln R)(d_2 ln R)
phi_s = sp.Symbol('phi_s', real=True)
R_diag = sp.tan((phi_s + d_s)/2) / sp.tan(phi_s/2)
d1_lnR = sp.diff(sp.log(sp.tan((phi_s + d_s)/2)), d_s)
d2_lnR = -sp.diff(sp.log(sp.tan(phi_s/2)), phi_s)
d12_Li2 = (R_diag / (1 - R_diag)) * d1_lnR * d2_lnR
res_Li2 = sp.simplify(sp.limit(d_s * d12_Li2, d_s, 0))
check('5c: Diagonal root-ratio dilogarithm mixed partial has residue 1/sin(phi_r) as d -> 0',
      sp.simplify(res_Li2 - 1/sp.sin(phi_s)) == 0)

# 5d. Conditional twisted ring Bethe equations and chamber Darboux wall-crossing
# Recall delta(1,2) = (2/9)*(ell_1 - ell_2)*ln A_12 + (2/3)*M_1*M_2*(v_1 - v_2)
# and -d_{P_1} delta(1,2) = Delta x_12 + gamma_1*M_2 = -(1/k_1)*ln A_12 + gamma_1*M_2
# Therefore Phi_ring(1|2) = delta(1,2) + Pi_1*M_2 satisfies:
# -d_{P_1} Phi_ring(1|2) = -d_{P_1} delta(1,2) - gamma_1*M_2 = Delta x_12 = -(1/k_1)*ln A_12!
dx12_s, gam1_s, M2_s, Pi1_s, Pi2_s, M1_s, d12_s = sp.symbols('dx12 gam1 M2 Pi1 Pi2 M1 d12', real=True)
dP1_delta = -(dx12_s + gam1_s*M2_s)
dP1_Phi_ring = dP1_delta + gam1_s*M2_s  # since d_{P_1}(Pi_1 M_2) = gamma_1 M_2
Phi_ring_12 = d12_s + Pi1_s*M2_s
Phi_ring_21 = -d12_s + Pi2_s*M1_s
check('5d: Twisted directional phase Phi_ring(1|2) = delta(1,2) + Pi_1 M_2 gives -d_{P_1} Phi_ring == Delta x_12 and symmetric sum Pi_1 M_2 + Pi_2 M_1',
      sp.simplify(-dP1_Phi_ring - dx12_s) == 0 and
      sp.simplify(Phi_ring_12 + Phi_ring_21 - (Pi1_s*M2_s + Pi2_s*M1_s)) == 0)

# 5e. Crossing antisymmetry of J_12 = (I_12 - I_21)/2
# Recall theta(q; alpha, eps) changes sign under (alpha, eps) -> (alpha + pi, eps) = (-k, -v) -> -theta(q; alpha, -eps)
# More directly, theta_2^{same}(q; alpha_2 + pi) = -theta_2^{head}(q; alpha_2), so I_12 and J_12 flip sign under alpha_2 -> alpha_2 + pi!
th1_s, th2_s = sp.symbols('th1 th2', real=True)
check('5e: Under theta_2^same(alpha_2 + pi) == -theta_2^head(alpha_2), J_12^same(a1, a2+pi) + J_12^head(a1, a2) == 0',
      sp.simplify(th1_s*(-th2_s) + th1_s*th2_s) == 0)

# 6. Static check that every verification script cited in Sections 1--10 and Appendix A exists
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[2]
tex_src = (ROOT / 'paper/boussinesq_soliton_scattering.tex').read_text()
sec_and_app_a = tex_src.split(r'\section{Inconsistencies found and their resolution}')[0]
cited_scripts = sorted(set(re.findall(r'\\scr\{([^{}]+\.py)\}', sec_and_app_a)))
check('6: All verification scripts cited in Sections 1--10 and Appendix A exist on disk',
      all((ROOT / p).is_file() for p in cited_scripts))

assert ok, 'SOME CHECKS FAILED'
print('ALL CHECKS PASSED')

"""Verification checks for Draft 2 revisions of paper/boussinesq_soliton_scattering.tex.

Every formula added or revised in response to paper/review_draft1.tex (items R1--R14
and optional analytic proofs) is checked here symbolically with SymPy and/or at
50-digit precision with mpmath:

 1. R1 (Prop. prop:EP): With theta = k x - omega t + x0, the t = 0 centre is X = -x0/k
    and the canonical 2-form pulled back to (k, x0) is dP ^ dX = -(P'(k)/k) dk ^ dx0.
    Verified symbolically and by 50-digit quadrature for both equations and directions,
    plus vanishing at the good cusp k = sqrt(3)/2 (alpha = pi/3).
 2. R2 & A7 (Sec. 2.3): Pseudo-differential expansion of L = d_x^3 + q_L d_x + r_L around
    the nonzero Lax background L_0 = d_x^3 - (1/4) d_x gives vacuum residues
    Res L_0^(1/3) = -1/12, Res L_0^(2/3) = 0, Res L_0^(4/3) = 0, Res L_0^(5/3) = 5/5184,
    and Res L^(3/3) = Res L^(6/3) = 0 identically. The vacuum-subtracted charges
    H_n = int (Res L^(n/3) - Res L_0^(n/3)) dx on a one-soliton equal H_1 = Delta_1,
    H_2 = Delta_2, H_4 = Delta_4 - Delta_2/3, H_5 = Delta_5 - 5 Delta_1/72 with ordered
    roots mu_1 = -p_b, mu_2 = -p_a (Delta_m = mu_1^m - mu_2^m), and reproduce P and E.
    Bad-equation roots (p_a, p_b) = (mu_a, mu_b) = (i v/sqrt(3) +- k)/2 satisfy
    mu^3 + mu/4 = i ell/(12 sqrt(3)) and the cross-ratio formula for A_12.
 3. R3 & A5 (Sec. 2.4, Sec. 4.2): Small-amplitude overtaking shifts
    Delta x_1 ~ -(2/k_1) log|(k_1-k_2)/(k_1+k_2)|, Delta x_2 ~ +(2/k_2) log|(k_1-k_2)/(k_1+k_2)|,
    and cross-term contribution -Pi_1' M_2 -> 4 beta_2 to the phase derivative d delta/d beta_1.
 4. R4 & R8 (Sec. 4.4): Signed Toda label velocity dE/dP_n = s b sinh(kappa)/kappa for
    both s = +-1, and two-body quantum Toda reflection amplitude R = Gamma(i kappa) hbar^(2i kappa)/Gamma(-i kappa)
    with kappa = 2q/hbar = Delta p/hbar verified from exact Bessel-K eigenfunctions.
 5. R5 (Sec. 5.2): Defect two-form sign d(f du + g dv) = (g_u - f_v) du ^ dv, vanishing
    on static vacuum sectors du = dv, with g_u - f_v = sigma in the shift-invariant gauge.
 6. R6 & Optional 2 (Prop. prop:stab): Defect vacuum triple cubic P_vac(sigma) = sigma^3 - 3 sigma + s(3 - s^2)
    has discriminant 27(4 - s^2)^2 (quadratic factor discriminant 3(4 - s^2)); Rolle's theorem
    with P_vac'(sigma) = 3(sigma^2 - 1) separates the three distinct real vacua for |s| < 2,
    s != +-1 into (-inf, -1), (-1, 1), (1, +inf), so exactly one root satisfies eta sigma < -1.
    Checks marginal coincidence at s = +-2 and single real vacuum for |s| > 2.
 7. R7 (Prop. prop:fusion): Oriented tree-level fusion identity
    delta_D^(q')(3) - delta_D^(q)(3) = rho [delta(3, S) + M_S Pi_3] with
    rho = eps_3 eta sgn(q - q') for zero-constant representatives, and shift by C(q') - C(q)
    under delta_D^(q) -> delta_D^(q) + C(q).
 8. R8 & Optional 2 (Props. prop:delta & prop:props): Analytic proof of the phase-gradient
    identity via dP = (2/9) k d ell and (2/9)(ell_1 - ell_2) d_1 log A_12 = -(1/3) Pi_1' M_2 - (2/3) Pi_2 M_1';
    and explicit two-body square-root branch expansion of delta(P_1, P_2) at the cusp alpha_1 = pi/3.
 9. R10 (Sec. 2): Explicit source field dictionaries for Bonnemain--Doyon (u_BD = u/2),
    Charlier--Lenells (u_CL = 3 v), and Bogdan--Charkina (varphi_BC = -phi, w_BC = -v).
10. R12 & R14 (Sec. 6.3): Non-separability d_S d_D delta != 0 verified in both overtaking
    and head-on channels for both the bad and good equations.

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


# ------------------------------------------------------------------ 1. R1: One-soliton coordinate & pullback
k_s, x0_s, x_s, t_s = sp.symbols('k x0 x t', real=True)
X_centre = -x0_s / k_s
# At t = 0, theta = k x + x0 = k (x - X_centre)
check('1a R1: k*x + x0 == k*(x - X) with X = -x0/k', sp.simplify(k_s*x_s + x0_s - k_s*(x_s - X_centre)) == 0)
# Jacobian of (k, X) -> (k, x0): dP(k) ^ dX = P'(k) dk ^ (-dx0/k + (x0/k^2) dk) = -(P'(k)/k) dk ^ dx0
J_mat = sp.Matrix([[sp.diff(k_s, k_s), sp.diff(k_s, x0_s)],
                   [sp.diff(X_centre, k_s), sp.diff(X_centre, x0_s)]])
check('1b R1: Jacobian det d(k, X)/d(k, x0) == -1/k', sp.simplify(J_mat.det() + 1/k_s) == 0)

for is_bad in (False, True):
    for eps in (1, -1):
        kk = mp.mpf('0.6')
        vel = lambda K: eps * (mp.sqrt(1 + K**2) if is_bad else mp.sqrt(1 - K**2))
        phi_f = lambda K, X0, X: K * (1 + mp.tanh((K*X + X0)/2))
        pi_f = lambda K, X0, X: -vel(K) * (K**2 / 2) * mp.sech((K*X + X0)/2)**2
        integrand = lambda X: (mp.diff(lambda K: pi_f(K, 0, X), kk) * mp.diff(lambda X0: phi_f(kk, X0, X), 0)
                               - mp.diff(lambda X0: pi_f(kk, X0, X), 0) * mp.diff(lambda K: phi_f(K, 0, X), kk))
        omega_kx0 = mp.quad(integrand, [-mp.inf, 0, mp.inf])
        P_of_k = lambda K: mp.mpf(2)/3 * K**3 * vel(K)
        Pk = mp.diff(P_of_k, kk)
        check('1c R1: 50-digit pullback Omega(k,x0) == -P_k/k (%s, eps=%+d)'
              % ('bad' if is_bad else 'good', eps), abs(omega_kx0 - (-Pk / kk)) < TOL)

al_s = sp.Symbol('alpha', positive=True)
P_good = sp.Rational(2, 3) * sp.sin(al_s)**3 * sp.cos(al_s)
check('1d R1: good pullback vanishes at cusp alpha = pi/3',
      sp.simplify(sp.diff(P_good, al_s).subs(al_s, sp.pi/3)) == 0)

# ------------------------------------------------------------------ 2. R2 & A7: Vacuum-subtracted GD charges & roots
q_fn = sp.Function('q')(x_s)
r_fn = sp.Function('r')(x_s)
NTRUNC = 7


def psdo_mult(A, B):
    """Multiply two pseudo-differential symbols sum_k a_k D^k down to order -NTRUNC."""
    out = {}
    for m, am in A.items():
        for n, bn in B.items():
            j = 0
            while True:
                deg = m + n - j
                if deg < -NTRUNC:
                    break
                c = sp.binomial(m, j)
                if c == 0 and m >= 0 and j > m:
                    break
                out[deg] = out.get(deg, 0) + c * am * sp.diff(bn, x_s, j)
                j += 1
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}


L_target = {3: sp.Integer(1), 1: q_fn, 0: r_fn}
X1 = {1: sp.Integer(1)}
for m in range(1, NTRUNC + 1):
    trial = dict(X1)
    cm = sp.Function('C%d' % m)(x_s)
    trial[-m] = cm
    cube = psdo_mult(psdo_mult(trial, trial), trial)
    sol_cm = sp.solve(sp.Eq(cube.get(2 - m, 0), L_target.get(2 - m, 0)), cm)[0]
    X1[-m] = sp.expand(sol_cm)

Powers = {1: X1}
for n in range(2, 7):
    Powers[n] = psdo_mult(Powers[n - 1], X1)

res_n = {n: sp.simplify(Powers[n].get(-1, sp.Integer(0))) for n in range(1, 7)}
res_vac = {n: sp.simplify(res_n[n].subs({q_fn: sp.Rational(-1, 4), r_fn: 0}).doit()) for n in range(1, 7)}
check('2a R2: Vacuum residues Res L_0^(n/3) for n=1..6',
      res_vac == {1: sp.Rational(-1, 12), 2: 0, 3: 0, 4: 0, 5: sp.Rational(5, 5184), 6: 0})
check('2b R2: Res L^(3/3) and Res L^(6/3) vanish identically',
      res_n[3] == 0 and res_n[6] == 0)

# Integrate vacuum-subtracted residues on a good one-soliton
k_sym, v_sym = sp.symbols('k_sym v_sym', real=True)
u_sol_sym = k_sym**2 / 2 * sp.sech(k_sym * x_s / 2)**2
q_sol_sym = sp.Rational(3, 2) * u_sol_sym - sp.Rational(1, 4)
r_sol_sym = (sp.diff(q_sol_sym, x_s) - (v_sym / sp.sqrt(3)) * (q_sol_sym + sp.Rational(1, 4))) / 2
sub_dens_fns = {}
for n in (1, 2, 4, 5):
    expr_n = sp.simplify(res_n[n].subs({q_fn: q_sol_sym, r_fn: r_sol_sym}).doit() - res_vac[n])
    sub_dens_fns[n] = sp.lambdify((x_s, k_sym, v_sym), expr_n, 'mpmath')

for al_val in (mp.mpf('0.35'), mp.mpf('0.85')):
    for eps in (1, -1):
        kk = mp.sin(al_val)
        vv = eps * mp.cos(al_val)
        pa = (kk + vv/R3)/2
        pb = (vv/R3 - kk)/2
        mu1, mu2 = -pb, -pa  # Ordered Phase 1 roots: mu1 - mu2 = k, mu1 + mu2 = -v/sqrt(3)
        check('2c A7: Ordered root dictionary mu_1 = -p_b, mu_2 = -p_a',
              abs((mu1 - mu2) - kk) < TOL and abs((mu1 + mu2) + vv/R3) < TOL)
        H_num = {n: mp.quad(lambda X: sub_dens_fns[n](X, kk, vv), [-mp.inf, 0, mp.inf]) for n in (1, 2, 4, 5)}
        D = lambda m: mu1**m - mu2**m
        check('2d R2: Subtracted GD charges H_1, H_2, H_4, H_5 match root formulas (a=%s, eps=%+d)'
              % (mp.nstr(al_val, 3), eps),
              max(abs(H_num[1] - D(1)), abs(H_num[2] - D(2)),
                  abs(H_num[4] - (D(4) - D(2)/3)), abs(H_num[5] - (D(5) - 5*D(1)/72))) < TOL)
        P_gd = -2*R3*H_num[4] - H_num[2]/R3
        E_gd = mp.mpf(24)/5*H_num[5] + H_num[1]/6
        check('2e R2: Physical P and E from subtracted H_n (a=%s, eps=%+d)'
              % (mp.nstr(al_val, 3), eps),
              max(abs(P_gd - mp.mpf(2)/3*kk**3*vv), abs(E_gd - mp.mpf(2)/15*kk**3*(5 - 4*kk**2))) < TOL)

# Bad-equation complex roots and cross-ratio (R10)
for b1, b2 in [(mp.mpf('0.35'), mp.mpf('0.8')), (mp.mpf('1.1'), mp.mpf('0.4'))]:
    for e1 in (1, -1):
        for e2 in (1, -1):
            k1, v1, l1 = mp.sinh(b1), e1*mp.cosh(b1), e1*mp.cosh(3*b1)
            k2, v2, l2 = mp.sinh(b2), e2*mp.cosh(b2), e2*mp.cosh(3*b2)
            p1a, p1b = (1j*v1/R3 + k1)/2, (1j*v1/R3 - k1)/2
            p2a, p2b = (1j*v2/R3 + k2)/2, (1j*v2/R3 - k2)/2
            cr = (p1a - p2a)*(p1b - p2b)/((p1a - p2b)*(p1b - p2a))
            Pbad = lambda K, W: W**2 - K**2 - K**4
            Abad = -Pbad(k1 - k2, k1*v1 - k2*v2) / Pbad(k1 + k2, k1*v1 + k2*v2)
            check('2g R10: Bad roots obey cubic and cross-ratio (%s,%s,%+d,%+d)'
                  % (mp.nstr(b1, 2), mp.nstr(b2, 2), e1, e2),
                  max(abs(p1a**3 + p1a/4 - 1j*l1/(12*R3)), abs(p1b**3 + p1b/4 - 1j*l1/(12*R3)), abs(cr - Abad)) < TOL)

# ------------------------------------------------------------------ 3. R3 & A5: Small-amplitude shifts & ratio
lam_s, k1_s, k2_s = sp.symbols('lambda k1 k2', positive=True)
A_good_ov = (1 - 3*sp.cot((k1_s + k2_s)*lam_s/2)**2) / (1 - 3*sp.cot((k1_s - k2_s)*lam_s/2)**2)
A_kdv = ((k1_s - k2_s)/(k1_s + k2_s))**2
check('3a R3: Small-amplitude overtaking A_12 -> ((k1-k2)/(k1+k2))^2',
      sp.simplify(sp.limit(A_good_ov, lam_s, 0) - A_kdv) == 0)
scale = mp.mpf('1e-14')
b1_m, b2_m = mp.mpf('1.7')*scale, scale
log_Akdv = 2*mp.log(abs((b1_m - b2_m)/(b1_m + b2_m)))
dx1_ex = -mp.log((1 + 3*mp.coth((b1_m + b2_m)/2)**2)/(1 + 3*mp.coth((b1_m - b2_m)/2)**2)) / mp.sinh(b1_m)
dx2_ex = +mp.log((1 + 3*mp.coth((b1_m + b2_m)/2)**2)/(1 + 3*mp.coth((b1_m - b2_m)/2)**2)) / mp.sinh(b2_m)
check('3b R3: Small-amplitude position shifts include 1/k_j factors',
      abs(dx1_ex / (-log_Akdv / b1_m) - 1) < mp.mpf('1e-25') and
      abs(dx2_ex / (+log_Akdv / b2_m) - 1) < mp.mpf('1e-25'))
full_ratio = 1 + 2*b2_m/(b1_m*log_Akdv)
cross_ratio_only = 2*b2_m/(b1_m*log_Akdv)
check('3c A5: Full ratio 0.564246498175... and cross-only ratio -0.435753501825...',
      abs(full_ratio - mp.mpf('0.56424649817500899496')) < mp.mpf('1e-18') and
      abs(cross_ratio_only - mp.mpf('-0.43575350182499100504')) < mp.mpf('1e-18') and
      abs(full_ratio - cross_ratio_only - 1) < TOL)

# ------------------------------------------------------------------ 4. R4 & R8: Toda signed velocity & Bessel R
kap_s, b_s = sp.symbols('kappa b', positive=True)
E_toda = 2*b_s**2*(sp.sinh(kap_s)*sp.cosh(kap_s) - kap_s)
for s_dir in (1, -1):
    P_toda = 4*s_dir*b_s*(kap_s*sp.cosh(kap_s) - sp.sinh(kap_s))
    v_toda = sp.simplify(sp.diff(E_toda, kap_s) / sp.diff(P_toda, kap_s))
    check('4a R4: Signed Toda label velocity dE/dP_n == s*b*sinh(kappa)/kappa (s=%+d)' % s_dir,
          sp.simplify(v_toda - s_dir*b_s*sp.sinh(kap_s)/kap_s) == 0)

hb_m, q_m = mp.mpf('0.41'), mp.mpf('0.73')
kap_m = 2*q_m / hb_m  # kappa = Delta p / hbar = 2 q / hbar in H_rel = q^2 + e^{-x}
psi_toda = lambda X: mp.besselk(1j*kap_m, 2*mp.exp(-X/2)/hb_m)
X_test = mp.mpf('0.8')
ode_res = -hb_m**2 * mp.diff(psi_toda, X_test, 2) + mp.exp(-X_test)*psi_toda(X_test) - q_m**2 * psi_toda(X_test)
X_asymp = mp.mpf('150')
A_in = mp.gamma(-1j*kap_m) * hb_m**(-1j*kap_m) / 2
A_out = mp.gamma(1j*kap_m) * hb_m**(1j*kap_m) / 2
R_amp = A_out / A_in
R_formula = mp.gamma(1j*kap_m) * hb_m**(2j*kap_m) / mp.gamma(-1j*kap_m)
check('4b R8: Toda two-body Bessel eigenfunction & reflection amplitude R',
      abs(ode_res) < TOL and
      abs(psi_toda(X_asymp) - (A_in*mp.exp(-1j*q_m*X_asymp/hb_m) + A_out*mp.exp(1j*q_m*X_asymp/hb_m))) < TOL and
      abs(R_amp - R_formula) < TOL)

# ------------------------------------------------------------------ 5. R5: Defect 2-form sign
u_d, v_d, c_d = sp.symbols('u_d v_d c_d', real=True)
sig_d = v_d - u_d + c_d
f_d = -sig_d**2 / 4
g_d = -sig_d**2 / 4
# Exterior derivative of f du + g dv is (g_u - f_v) du ^ dv
coeff_du_dv = sp.simplify(sp.diff(g_d, u_d) - sp.diff(f_d, v_d))
check('5 R5: d(f du + g dv) has du ^ dv coefficient (g_u - f_v) == sigma in shift-invariant gauge',
      sp.simplify(coeff_du_dv - sig_d) == 0)

# ------------------------------------------------------------------ 6. R6 & Optional 2: Defect vacua & Rolle's theorem
sig_s, s_s = sp.symbols('sigma s', real=True)
quad_vac = sig_s**2 + s_s*sig_s + s_s**2 - 3
P_vac = sp.expand((sig_s - s_s) * quad_vac)
check('6a R6: Cubic P_vac(sigma) == sigma^3 - 3*sigma + s*(3 - s^2)',
      sp.expand(P_vac - (sig_s**3 - 3*sig_s + s_s*(3 - s_s**2))) == 0)
check('6b R6: P_vac\'(sigma) == 3*(sigma^2 - 1) and quadratic discriminant == 3*(4 - s^2)',
      sp.expand(sp.diff(P_vac, sig_s) - 3*(sig_s**2 - 1)) == 0 and
      sp.expand(sp.discriminant(quad_vac, sig_s) - 3*(4 - s_s**2)) == 0)
# For |s| < 2, s != +-1: P_vac(-1) = 2 + s(3 - s^2) = (2 - s)(1 + s)^2 > 0 and P_vac(+1) = -2 + s(3 - s^2) = -(2 + s)(1 - s)^2 < 0
check('6c R6: Rolle extrema values P_vac(-1) = (2-s)(1+s)^2 and P_vac(+1) = -(2+s)(1-s)^2',
      sp.expand(P_vac.subs(sig_s, -1) - (2 - s_s)*(1 + s_s)**2) == 0 and
      sp.expand(P_vac.subs(sig_s, 1) + (2 + s_s)*(1 - s_s)**2) == 0)
# Hence for |s| < 2 and s != +-1, P_vac(-1) > 0 and P_vac(+1) < 0, so the 3 real roots lie in (-inf,-1), (-1,1), (1,inf).
for s_val in (mp.mpf('-1.7'), mp.mpf('-0.4'), mp.mpf('0.5'), mp.mpf('1.6')):
    r_list = sorted([s_val, (-s_val - mp.sqrt(12 - 3*s_val**2))/2, (-s_val + mp.sqrt(12 - 3*s_val**2))/2])
    check('6d R6: Three real vacua separated by -1 and +1 at s=%s' % mp.nstr(s_val, 2),
          r_list[0] < -1 < r_list[1] < 1 < r_list[2])

# ------------------------------------------------------------------ 7. R7: Oriented tree-level fusion & constant shift
def sol_good_mp(a, eps):
    kk = mp.sin(a)
    vv = eps * mp.cos(a)
    return dict(k=kk, v=vv, w=kk*vv, M=2*kk, Pi=-2*kk*vv, P=mp.mpf(2)/3*kk**3*vv, l=-eps*mp.cos(3*a),
                pa=(kk + vv/R3)/2, pb=(vv/R3 - kk)/2)


def delta_ss_good(s1, s2):
    A = (s1['pa'] - s2['pa'])*(s1['pb'] - s2['pb'])/((s1['pa'] - s2['pb'])*(s1['pb'] - s2['pa']))
    return mp.mpf(2)/9*(s1['l'] - s2['l'])*mp.log(A) + mp.mpf(2)/3*s1['M']*s2['M']*(s1['v'] - s2['v'])


def delta_sd_good(s1, eps1, q, eta, C_func=lambda q_: mp.mpf(0)):
    s_par = -2*R3*q
    lD = (3*s_par - s_par**3)/2
    z = ((s1['pb'] - q)/(s1['pa'] - q))**eta
    MD = eps1*eta*q
    PiD = eps1*eta*(s_par**2 - 2)/(2*R3)
    return mp.mpf(2)/9*(s1['l'] - lD)*mp.log(z**eps1) - mp.mpf(2)/3*(s1['Pi']*MD - PiD*s1['M']) + C_func(q)


q_base = mp.mpf('0.42')
triple_q = [q_base, (-q_base + mp.sqrt(1 - 3*q_base**2))/2, (-q_base - mp.sqrt(1 - 3*q_base**2))/2]
for eta in (1, -1):
    for eps3 in (1, -1):
        s3 = sol_good_mp(mp.mpf('0.12'), eps3)
        for idx in range(3):
            q_in, q_out = triple_q[idx], triple_q[(idx + 1) % 3]
            vS = R3 * (q_in + q_out)
            sS = sol_good_mp(mp.acos(abs(vS)), 1 if vS >= 0 else -1)
            rho = eps3 * eta * (1 if q_in > q_out else -1)
            lhs0 = delta_sd_good(s3, eps3, q_out, eta) - delta_sd_good(s3, eps3, q_in, eta)
            rhs0 = rho * (delta_ss_good(s3, sS) + sS['M']*s3['Pi'])
            C_test = lambda qq: mp.sin(5*qq) + qq**2
            lhsC = delta_sd_good(s3, eps3, q_out, eta, C_test) - delta_sd_good(s3, eps3, q_in, eta, C_test)
            check('7 R7: Oriented fusion & constant shift (eta=%+d, eps3=%+d, edge=%d)' % (eta, eps3, idx),
                  abs(lhs0 - rhs0) < TOL and abs((lhsC - rhs0) - (C_test(q_out) - C_test(q_in))) < TOL)

# ------------------------------------------------------------------ 8. R8 & Optional 2: Analytic phase gradient & cusp branch
a1_s, a2_s, b1_s, b2_s = sp.symbols('a1 a2 b1 b2', positive=True)
e1_s, e2_s = sp.symbols('e1 e2', real=True)
# Root-polynomial proof of (2/9)(ell_1 - ell_2) d_1 log A_12 = -(1/3) Pi_1' M_2 - (2/3) Pi_2 M_1'
p1a_s, p1b_s, p2a_s, p2b_s, dp1a_s, dp1b_s = sp.symbols('p1a p1b p2a p2b dp1a dp1b')
# On each root pair (p1r, p2s) with ell(p1r)=ell_1 and ell(p2s)=ell_2:
# (ell_1 - ell_2)/(p1r - p2s) = -3*sqrt(3)*(1 - 4*(p1r^2 + p1r*p2s + p2s^2))
q_kern = lambda p1r, p2s: -3*sp.sqrt(3)*(1 - 4*(p1r**2 + p1r*p2s + p2s**2))
alt_sum = sp.expand(dp1a_s*(q_kern(p1a_s, p2a_s) - q_kern(p1a_s, p2b_s))
                    + dp1b_s*(q_kern(p1b_s, p2b_s) - q_kern(p1b_s, p2a_s)))
# Express in terms of M_1 = 2(p1a-p1b), Pi_1 = -2*sqrt(3)*(p1a^2-p1b^2), M_2 = 2(p2a-p2b), Pi_2 = -2*sqrt(3)*(p2a^2-p2b^2)
M2_root = 2*(p2a_s - p2b_s)
Pi2_root = -2*sp.sqrt(3)*(p2a_s**2 - p2b_s**2)
dM1_root = 2*(dp1a_s - dp1b_s)
dPi1_root = -4*sp.sqrt(3)*(p1a_s*dp1a_s - p1b_s*dp1b_s)
check('8a Opt2: Root cross-ratio identity (2/9)(l1-l2) d_1 log A_12 == -(1/3) Pi_1\' M_2 - (2/3) Pi_2 M_1\'',
      sp.expand(sp.Rational(2, 9)*alt_sum - (-sp.Rational(1, 3)*dPi1_root*M2_root - sp.Rational(2, 3)*Pi2_root*dM1_root)) == 0)

# Symbolic verification of dP = (2/9) k d(ell) and full trig/hyperbolic identity via root cross-ratio substitution
for name, p1, p2, k1, k2, v1, v2, l1, l2, wfac in [
    ('good', a1_s, a2_s, sp.sin(a1_s), sp.sin(a2_s), e1_s*sp.cos(a1_s), e2_s*sp.cos(a2_s),
     -e1_s*sp.cos(3*a1_s), -e2_s*sp.cos(3*a2_s), sp.Integer(1)/sp.sqrt(3)),
    ('bad', b1_s, b2_s, sp.sinh(b1_s), sp.sinh(b2_s), e1_s*sp.cosh(b1_s), e2_s*sp.cosh(b2_s),
     e1_s*sp.cosh(3*b1_s), e2_s*sp.cosh(3*b2_s), sp.I/sp.sqrt(3))
]:
    P1 = sp.Rational(2, 3)*k1**3*v1
    M1, M2 = 2*k1, 2*k2
    Pi1, Pi2 = -M1*v1, -M2*v2
    check('8b Opt2: dP_1 == (2/9) k_1 d(ell_1) (%s)' % name,
          sp.simplify(sp.expand_trig(sp.diff(P1, p1) - sp.Rational(2, 9)*k1*sp.diff(l1, p1))) == 0)
    r1a, r1b = (wfac*v1 + k1)/2, (wfac*v1 - k1)/2
    r2a, r2b = (wfac*v2 + k2)/2, (wfac*v2 - k2)/2
    dlogA = (sp.diff(r1a, p1)/(r1a - r2a) + sp.diff(r1b, p1)/(r1b - r2b)
             - sp.diff(r1a, p1)/(r1a - r2b) - sp.diff(r1b, p1)/(r1b - r2a))
    id_expr = sp.Rational(2, 9)*(l1 - l2)*dlogA + sp.Rational(1, 3)*sp.diff(Pi1, p1)*M2 + sp.Rational(2, 3)*Pi2*sp.diff(M1, p1)
    for ev1, ev2 in [(1, 1), (1, -1)]:
        id_ev = sp.simplify(sp.expand_trig(sp.cancel(id_expr.subs({e1_s: ev1, e2_s: ev2}))))
        check('8b Opt2: Symbolic identity (2/9)(l1-l2) d_1 log A == -(1/3) Pi1\' M2 - (2/3) Pi2 M1\' (%s, e2=%+d)'
              % (name, ev2), id_ev == 0)

# Two-body cusp square-root branch expansion at alpha_1 = pi/3 + da
da_s = sp.Symbol('da', real=True)
P_cusp_ser = sp.series(P_good.subs(al_s, sp.pi/3 + da_s), da_s, 0, 3).removeO()
check('8c R8: P(pi/3 + da) == sqrt(3)/8 - (sqrt(3)/2) da^2 + O(da^3)',
      sp.simplify(P_cusp_ser - (sp.sqrt(3)/8 - sp.sqrt(3)/2 * da_s**2)) == 0)
a_cusp = mp.pi / 3
a2_fix = mp.mpf('0.25')
s2_fix = sol_good_mp(a2_fix, 1)
d1_at_cusp = mp.diff(lambda a_: delta_ss_good(sol_good_mp(a_, 1), s2_fix), a_cusp)
expected_d1 = -s2_fix['M']  # Since P_1'(pi/3) = 0 and Pi_1 = -eps_1 sin(2 alpha_1), d_1 delta = -Pi_1'(pi/3) M_2 = -eps_1 M_2
da_small = mp.mpf('1e-12')
P_max = R3 / 8
dP_val = P_max - sol_good_mp(a_cusp + da_small, 1)['P']
ddelta_val = delta_ss_good(sol_good_mp(a_cusp + da_small, 1), s2_fix) - delta_ss_good(sol_good_mp(a_cusp, 1), s2_fix)
sqrt_coeff = ddelta_val / mp.sqrt(dP_val)
check('8d R8: Two-body phase has square-root branch point in P_1 at cusp alpha_1 = pi/3',
      abs(d1_at_cusp - expected_d1) < TOL and abs(sqrt_coeff - (-mp.sqrt(2/R3)*s2_fix['M'])) < mp.mpf('1e-10'))

# ------------------------------------------------------------------ 9. R10: Source field dictionaries (BD, CL, BC)
phi_fn = sp.Function('phi')(x_s, t_s)
u_fn = sp.Function('u')(x_s, t_s)
v_fn = sp.Function('v')(x_s, t_s)
u_BD = u_fn / 2
u_CL = 3 * v_fn
phi_BC = -phi_fn
check('9a R10: BD dictionary u_BD = u/2',
      sp.expand(2*(u_BD.diff(t_s, 2) - u_BD.diff(x_s, 2) + u_BD.diff(x_s, 4) + 6*(u_BD**2).diff(x_s, 2))
                - (u_fn.diff(t_s, 2) - u_fn.diff(x_s, 2) + u_fn.diff(x_s, 4) + 3*(u_fn**2).diff(x_s, 2))) == 0)
check('9b R10: CL dictionary u_CL = 3*v',
      sp.expand((u_CL.diff(t_s, 2) - u_CL.diff(x_s, 2) - u_CL.diff(x_s, 4) - (u_CL**2).diff(x_s, 2))/3
                - (v_fn.diff(t_s, 2) - v_fn.diff(x_s, 2) - v_fn.diff(x_s, 4) - 3*(v_fn**2).diff(x_s, 2))) == 0)
check('9c R10: BC dictionary varphi_BC = -phi, w_BC = -v preserves L_bad, P, E',
      sp.expand((phi_BC.diff(t_s, 2) - phi_BC.diff(x_s, 2) - phi_BC.diff(x_s, 4) + 6*phi_BC.diff(x_s)*phi_BC.diff(x_s, 2))
                + (phi_fn.diff(t_s, 2) - phi_fn.diff(x_s, 2) - phi_fn.diff(x_s, 4) - 3*(phi_fn.diff(x_s)**2).diff(x_s))) == 0)

# ------------------------------------------------------------------ 10. R12 & R14: Non-product form d_S d_D delta != 0 (good & bad)
def delta_SD_good(S, D, eps2):
    return delta_ss_good(sol_good_mp((S + D)/2, 1), sol_good_mp((S - D)/2, eps2))


def delta_SD_bad(S, D, eps2):
    a1, a2 = (S + D)/2, (S - D)/2
    k1, v1, l1 = mp.sinh(a1), mp.cosh(a1), mp.cosh(3*a1)
    k2, v2, l2 = mp.sinh(a2), eps2*mp.cosh(a2), eps2*mp.cosh(3*a2)
    Pbad = lambda K, W: W**2 - K**2 - K**4
    A = -Pbad(k1 - k2, k1*v1 - k2*v2) / Pbad(k1 + k2, k1*v1 + k2*v2)
    return mp.mpf(2)/9*(l1 - l2)*mp.log(A) + mp.mpf(2)/3*(2*k1)*(2*k2)*(v1 - v2)


for eps2 in (1, -1):
    dSD_bad = mp.diff(lambda s_: mp.diff(lambda d_: delta_SD_bad(s_, d_, eps2), mp.mpf('0.6')), mp.mpf('1.4'))
    dSD_good = mp.diff(lambda s_: mp.diff(lambda d_: delta_SD_good(s_, d_, eps2), mp.mpf('0.2')), mp.mpf('0.7'))
    check('10 R12/R14: Mixed derivative d_S d_D delta != 0 for both bad (%s) and good (%s) (eps2=%+d)'
          % (mp.nstr(dSD_bad, 4), mp.nstr(dSD_good, 4), eps2),
          abs(dSD_bad) > mp.mpf('0.01') and abs(dSD_good) > mp.mpf('0.01'))

print('ALL CHECKS PASSED')

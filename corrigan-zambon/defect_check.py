"""Corrigan-Zambon integrable defect (arXiv:2306.03772, §3) re-derived with kappa^2 = K a free symbol.

Bulk Lagrangian (their 2.1):  L = u_t^2/(2c^2) - u_x^2/2 - (lam/3) u_x^3 - (K/2) u_xx^2,
K > 0 'good', K < 0 'bad' (their eq. 2.2 with kappa^2 -> K).  Defect: D = u_t F + v_t G + H at x0.

Part 1 (from scratch): sewing conditions from varying the action; momentum balance
   Phi_u - Phi_v + d/dt(u_x F + v_x G) = dP_D/dt
must hold for some P_D(u,u_x,v,v_x).  With a general type-I ansatz (F, G linear in u_x, v_x with
constant coefficients, H a general quadratic in u_x, v_x with coefficient functions of u, v) we
derive the NECESSARY conditions from (i) terms quadratic in time derivatives and (ii) terms cubic
in space derivatives:   a2 = a1 = alpha, b2 = b1 = beta,  beta^2 - alpha^2 = K/c^2,  and for lam != 0,
beta = 2 alpha  =>  alpha^2 = K/(3c^2).

Part 2: SUFFICIENCY: their full solution, written for symbolic alpha (K := 3 c^2 alpha^2), makes the
momentum balance an exact time derivative identically.  Hence the construction is algebraic in alpha
and continues to K < 0 only with alpha, beta purely imaginary.

Part 3: reality. For K < 0 the coefficient of u_t u_x in D is alpha in iR, so D is not real for real
fields: no real type-I defect of this family exists for the bad equation when lam != 0.  For lam = 0
(linear bad equation) real solutions alpha^2 - beta^2 = |K|/c^2 do exist.
"""
import sympy as sp

c, lam, K = sp.symbols('c lambda K', nonzero=True)
u, ux, v, vx, ut, vt, uxt, vxt = sp.symbols('u u_x v v_x u_t v_t u_xt v_xt')
Z = [u, ux, v, vx]
Zt = {u: ut, ux: uxt, v: vt, vx: vxt}


def ddt(expr):
    """time derivative of a function of (u,ux,v,vx) evaluated at x0"""
    return sum(sp.diff(expr, z)*Zt[z] for z in Z)


def balance(F, G, H):
    """Phi_u - Phi_v + d/dt(ux F + vx G) after eliminating u_xx, u_xxx, v_xx, v_xxx by sewing."""
    D_u = ut*sp.diff(F, u) + vt*sp.diff(G, u) + sp.diff(H, u)
    D_ux = ut*sp.diff(F, ux) + vt*sp.diff(G, ux) + sp.diff(H, ux)
    D_v = ut*sp.diff(F, v) + vt*sp.diff(G, v) + sp.diff(H, v)
    D_vx = ut*sp.diff(F, vx) + vt*sp.diff(G, vx) + sp.diff(H, vx)
    dLdux = lambda w: -w - lam*w**2            # dL/dw_x
    # variation of the action (derived in the note):
    #  du : dL/dux - d_x dL/duxx + D_u - d_t D_ut = 0,   dux : dL/duxx + D_ux = 0  (dL/duxx = -K uxx)
    #  dv : -(dL/dvx - d_x dL/dvxx) + D_v - d_t D_vt = 0, dvx : -dL/dvxx + D_vx = 0
    uxx = D_ux/K
    uxxx = (-dLdux(ux) - D_u + ddt(F))/K           # -ux - lam ux^2 + K uxxx + D_u - dF/dt = 0
    vxx = -D_vx/K
    vxxx = (-dLdux(vx) + D_v - ddt(G))/K           # vx + lam vx^2 - K vxxx + D_v - dG/dt = 0
    Phi = lambda wt, wx, wxx, wxxx: wt**2/(2*c**2) + wx**2/2 + 2*lam*wx**3/3 - K*wx*wxxx + K*wxx**2/2
    lhs = Phi(ut, ux, uxx, uxxx) - Phi(vt, vx, vxx, vxxx) + ddt(ux*F + vx*G)
    return sp.expand(lhs)


# ---------------- Part 1: necessary conditions -------------------------------------------------
a1, b1, a2, b2 = sp.symbols('a1 b1 a2 b2')
f = sp.Function('f')(u, v); g = sp.Function('g')(u, v)
h11, h12, h22, ru, rv, tau = [sp.Function(n)(u, v) for n in ('h11', 'h12', 'h22', 'r_u', 'r_v', 'tau')]
F = a1*ux + b1*vx + f
G = b2*ux + a2*vx + g
H = (h11*ux**2 + 2*h12*ux*vx + h22*vx**2)/2 + ru*ux + rv*vx + tau
E = balance(F, G, H)
P = sp.Poly(E, ut, vt, uxt, vxt)
# (i) quadratic in time derivatives must vanish
quad = [P.coeff_monomial(m) for m in (ut**2, vt**2, ut*vt)]
# coefficients of uxt, vxt are F, G automatically?
assert sp.simplify(P.coeff_monomial(uxt) - F) == 0 and sp.simplify(P.coeff_monomial(vxt) - G) == 0
eqs = []
for q in quad:
    eqs += sp.Poly(sp.expand(q), ux, vx).coeffs()
sols = sp.solve(eqs, [a2, b2, a1], dict=True)
print('(i) time-quadratic conditions, nondegenerate solutions:')
for s_ in sols:
    print('     ', s_)
# exactness requires dF/dv_x = dG/du_x (paper 3.15), i.e. b2 = b1; this removes the b2 = -b1 branches
kept = [s_ for s_ in sols if sp.simplify(s_[b2] - b1) == 0]
assert kept and all(sp.simplify(s_[a2] - s_[a1]) == 0 for s_ in kept)
assert all(sp.simplify(s_[a1]**2 - (b1**2 - K/c**2)) == 0 for s_ in kept)
print('     exactness (b2 = b1) leaves a2 = a1 =: alpha, b1 =: beta with alpha^2 = beta^2 - K/c^2')
# show the residual constraint on that branch
E1 = balance(a1*ux + b1*vx + f, b1*ux + a1*vx + g, H)
q1 = sp.factor(sp.Poly(E1, ut, vt).coeff_monomial(ut**2))
print('     on branch a2=a1, b2=b1: u_t^2 coefficient =', q1, ' => beta^2 - alpha^2 = K/c^2')
# (ii) cubic space-derivative terms in the time-derivative-free part
E1 = E1.subs(sp.Symbol('a1'), a1)
E0 = sp.Poly(E1, ut, vt, uxt, vxt).coeff_monomial(1)
P0 = sp.Poly(sp.expand(E0), ux, vx)
cubic = {m: P0.coeff_monomial(m) for m in (ux**3, vx**3, ux**2*vx, ux*vx**2)}
for m, cf in cubic.items():
    print('(ii) coefficient of', m, ':', sp.factor(cf))
# from these: H quadratic coefficients must be h11 = h22 = beta*sigma, h12 = alpha*sigma (paper 3.31);
# insert that and read off the conditions on sigma
sig = sp.Function('sigma')(u, v)
E2 = balance(a1*ux + b1*vx + f, b1*ux + a1*vx + g,
             (sig*(b1*ux**2 + 2*a1*ux*vx + b1*vx**2))/2 + ru*ux + rv*vx + tau)
E2 = sp.expand(E2.subs(a1**2, b1**2 - K/c**2))
P2 = sp.Poly(sp.Poly(E2, ut, vt, uxt, vxt).coeff_monomial(1), ux, vx)
c30, c03 = sp.factor(P2.coeff_monomial(ux**3)), sp.factor(P2.coeff_monomial(vx**3))
c21, c12 = sp.factor(P2.coeff_monomial(ux**2*vx)), sp.factor(P2.coeff_monomial(ux*vx**2))
print('(ii) with H_2 = sigma(beta ux^2 + 2 alpha ux vx + beta vx^2)/2:')
for name, cf in [('ux^3', c30), ('vx^3', c03), ('ux^2 vx', c21), ('ux vx^2', c12)]:
    print('       ', name, ':', cf)
s_u, s_v = sp.symbols('s_u s_v')
sub = {sp.Derivative(sig, u): s_u, sp.Derivative(sig, v): s_v}
sol2 = sp.solve([c30.subs(sub), c03.subs(sub), c21.subs(sub), c12.subs(sub)], [s_u, s_v, b1], dict=True)
print('     solutions (sigma_u, sigma_v, beta):', sol2)
assert any(sp.simplify(s[b1] - 2*a1) == 0 for s in sol2 if b1 in s)
print('  => for lam != 0: beta = 2 alpha, and beta^2 - alpha^2 = K/c^2 gives alpha^2 = K/(3 c^2)')

# ---------------- Part 2: sufficiency of the full solution for symbolic alpha ---------------------
al, s0, r0, t0 = sp.symbols('alpha sigma0 rho0 tau0')
Kv = 3*c**2*al**2
be = 2*al
mu = 2*lam/(3*be)
w = u - v
sigma = s0 + mu*w
rho = r0 + (sigma - sigma**3/(3*c**2))/(2*mu)
W = sp.Symbol('W')
# d tau/du = -beta sigma rho/(2K), tau a function of w = u - v (paper 3.39-3.42)
tau_ = sp.integrate((-be*sigma*rho/(2*Kv)).subs(u, W + v), W).subs(W, w) + t0
# f, g with f_v - g_u = sigma/c^2
fs = sp.integrate(sigma, v)/c**2
gs = sp.Integer(0)
Fs = al*ux + be*vx + fs
Gs = be*ux + al*vx + gs
Hs = sigma*(be*ux**2 + 2*al*ux*vx + be*vx**2)/2 + rho*(ux + vx) + tau_
Eb = balance(Fs, Gs, Hs).subs(K, Kv)
Eb = sp.expand(Eb)
Pb = sp.Poly(Eb, ut, vt, uxt, vxt)
rest = {m: sp.simplify(Pb.coeff_monomial(m)) for m in (ut**2, vt**2, ut*vt, 1)}
print('\nPart 2: leftover non-exact pieces (must all be 0):', rest)
A_ = sp.simplify(Pb.coeff_monomial(ut)); B_ = sp.simplify(Pb.coeff_monomial(uxt))
C_ = sp.simplify(Pb.coeff_monomial(vt)); E_ = sp.simplify(Pb.coeff_monomial(vxt))
one_form = [A_, B_, C_, E_]
curl = [sp.simplify(sp.diff(one_form[i], Z[j]) - sp.diff(one_form[j], Z[i]))
        for i in range(4) for j in range(i + 1, 4)]
print('        closedness of dP_D (all 6 curls zero):', all(cc == 0 for cc in curl))
assert all(r == 0 for r in rest.values()) and all(cc == 0 for cc in curl)

# ---------------- Part 3: reality for K < 0 -----------------------------------------------------
Kn = sp.Symbol('kappa', positive=True)        # bad: K = -kappa^2
alpha_bad = sp.solve(sp.Eq(3*c**2*al**2, -Kn**2), al)
print('\nPart 3: bad equation (K = -kappa^2), lam != 0: alpha =', alpha_bad, '-> purely imaginary;')
print('        coefficient of u_t u_x in D is alpha, so D is not real for real fields.')
al_l, be_l = sp.symbols('alpha_l beta_l', real=True)
print('        lam = 0 instead: beta^2 - alpha^2 = -kappa^2/c^2 has real solutions, e.g. alpha = kappa/c, beta = 0'
      ' (allowed, since beta = 2 alpha is not forced when lam = 0).')
print('ALL CHECKS PASSED')

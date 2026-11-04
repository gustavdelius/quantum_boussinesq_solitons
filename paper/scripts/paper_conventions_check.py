"""Checks for formulas that the paper states in a form not already verified verbatim elsewhere.

Paper: paper/boussinesq_soliton_scattering.tex.  Everything here is elementary; it re-derives the
soliton-sector conventions of the paper and records the convention inconsistencies listed in the
paper's appendix "Inconsistencies found".

 1. Lagrangians.  L_good = 1/2 phi_t^2 - 1/2 phi_x^2 + phi_x^3 - 1/2 phi_xx^2 gives
    u_tt = u_xx - u_xxxx - 3(u^2)_xx  (u = phi_x).  The bad equation v_tt = v_xx + v_xxxx + 3(v^2)_xx
    needs L_bad = 1/2 phi_t^2 - 1/2 phi_x^2 - phi_x^3 + 1/2 phi_xx^2.  The bad Lagrangian printed in
    phase2/report.tex (Sec. 1), with +phi_x^3, gives instead v_tt = v_xx + v_xxxx - 3(v^2)_xx
    (recorded as an inconsistency; the positive soliton v = 2(log tau)_xx does not solve it).
 2. One-solitons u = (k^2/2) sech^2((k x - w t)/2) solve the respective equations exactly.
 3. Charges by 40-digit quadrature (both equations, both directions):
    M = int u = 2k, Pi = int pi = -vM, P = -int pi phi_x = (2/3) k^3 v,
    E = int (1/2 pi^2 + 1/2 phi_x^2 -+ phi_x^3 +- 1/2 phi_xx^2) = (2/15) k^3 (5 -+ 4k^2), dE/dP = v.
    The energy printed in phase2/oneloop_phase.tex eq. (onesol_data), E = (1/3) k^3 (2 cos^2 a + 1),
    disagrees (recorded as an inconsistency).
 4. KP roots (good): p_a = (k + v/sqrt3)/2, p_b = (v/sqrt3 - k)/2 satisfy p_a^2 + p_a p_b + p_b^2 = 1/4 and
    ell(p) = -3 sqrt3 p (1 - 4p^2) = -eps cos 3a.  Relation to the Phase-1 roots mu (mu_1 - mu_2 = k,
    mu_1 + mu_2 = -v/sqrt3): {mu_1, mu_2} = {-p_b, -p_a}, and the Lax eigenvalue
    lambda = mu^3 - mu/4 = -ell/(12 sqrt3).  A_12 equals the cross-ratio of the roots (both orientations).
 5. Good/bad dictionary by continuation alpha = i beta:
    (k, M, Pi, P, E, ell, v) -> (i k, i M, i Pi, -i P, -i E, -ell, v)_bad and A -> A_bad, hence
    delta_good(i beta_1, i beta_2) = -delta_bad(beta_1, beta_2) (both orientations).
 6. Unitarity delta(1,2) = -delta(2,1); crossing-type relations
    bad:  delta_ov(b1, b2 + i pi) = -delta_ho(b1, b2);   good: delta_ov(a1, a2 + pi) = -delta_ho(a1, a2).
 7. Small-amplitude limits (bad, both apply to good with beta -> alpha):
    overtaking: [d delta/d b1] / [2 b1 log A] -> 1 + 2 b2 / (b1 log A_KdV), A_KdV = ((b1-b2)/(b1+b2))^2,
      = 0.5642... at b1/b2 = 1.7; the extra term comes from the cross-term (Darboux) part
      -(dPi_1/db1) M_2 -> 4 b2 of d delta/d b1;
    head-on: delta_ho -> 4 b1 b2 = (16/3) b1 b2 [topological] - (4/3) b1 b2 [dynamical].
 8. Bogdan-Charkina adiabatic invariant: i (2/9)(ell_1 - ell_2) = -(4/9) sinh 3a sin 3b at beta = a +- i b.
 9. Darboux coefficient (good): gamma = (dPi/da)/(dP/da) = -3 cos 2a / (sin a sin 3a);
    dP/da = (2/3) eps sin a sin 3a (bad: dP/db = (2/3) eps sinh b sinh 3b).
10. Defect (Phase 2c convention, s = sigma_0, q = -s/(2 sqrt3)): ell(q) = (3s - s^3)/2 and the CZ
    factor z = (A - eta sqrt3)/(A + eta sqrt3), A = (v + s)/k, equals ((p_b - q)/(p_a - q))^eta.
    The middle expression of phase2/oneloop_defect_phase.tex eq. (z1_sol),
    ((k - v/sqrt3 - s/sqrt3)/(k + v/sqrt3 + s/sqrt3))^(-eta), equals -1/(its own root form) instead
    (recorded as an inconsistency).
Ends with ALL CHECKS PASSED.
"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
TOL = mp.mpf(10)**-30
ok = True


def check(name, cond):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name)
    ok &= bool(cond)


# ------------------------------------------------------------------ 1. Lagrangians -> PDEs
x, t = sp.symbols('x t', real=True)
phi = sp.Function('phi')(x, t)


def el_pde(a, b):
    """L = 1/2 phi_t^2 - 1/2 phi_x^2 + a phi_x^3 + (b/2) phi_xx^2; return E-L expression differentiated
    once in x, written in u = phi_x, as  u_tt - [u_xx + c4 u_xxxx + c2 (u^2)_xx]."""
    L = (sp.Rational(1, 2)*phi.diff(t)**2 - sp.Rational(1, 2)*phi.diff(x)**2 + a*phi.diff(x)**3
         + sp.Rational(b, 2)*phi.diff(x, 2)**2)
    eq = sp.euler_equations(L, phi, [x, t])[0].lhs
    return sp.expand(sp.diff(eq, x))


def residual(a, b, c4, c2):
    """E-L(x-derivative) + [u_tt - u_xx - c4 u_xxxx - c2 (u^2)_xx] should vanish identically (up to sign)."""
    el = el_pde(a, b)
    # substitute phi = integral of u: test on a generic polynomial-exponential phi instead
    f = sp.exp(-x**2)*sp.sin(t + x/3) + x*sp.exp(-x**2 - t**2)
    lhs = el.subs(phi, f).doit()
    uu = sp.diff(f, x)
    target = (sp.diff(uu, t, 2) - sp.diff(uu, x, 2) - c4*sp.diff(uu, x, 4) - c2*sp.diff(uu**2, x, 2))
    # E-L equation is -(target) = 0 (overall sign conventional); compare up to sign
    pts = [(sp.Rational(3, 10), sp.Rational(1, 7)), (-sp.Rational(6, 5), sp.Rational(2, 3))]
    r1 = max(abs(sp.N((lhs + target).subs({x: p, t: q}), 30)) for p, q in pts)
    r2 = max(abs(sp.N((lhs - target).subs({x: p, t: q}), 30)) for p, q in pts)
    return min(r1, r2)


check('1a good Lagrangian -> u_tt = u_xx - u_xxxx - 3(u^2)_xx', residual(1, -1, -1, -3) < 1e-25)
check('1b bad Lagrangian (-phi_x^3, +phi_xx^2/2) -> v_tt = v_xx + v_xxxx + 3(v^2)_xx', residual(-1, 1, 1, 3) < 1e-25)
check('1c report.tex bad Lagrangian (+phi_x^3) gives the WRONG nonlinear sign v_tt = ... - 3(v^2)_xx',
      residual(1, 1, 1, -3) < 1e-25 and residual(1, 1, 1, 3) > 1e-5)

# ------------------------------------------------------------------ 2. one-solitons solve the PDEs
k = sp.Symbol('k', positive=True)
for name, sg, w in [('good', -1, k*sp.sqrt(1 - k**2)), ('bad', +1, k*sp.sqrt(1 + k**2))]:
    th = k*x - w*t
    U = k**2/2*sp.sech(th/2)**2
    res = sp.diff(U, t, 2) - sp.diff(U, x, 2) - sg*sp.diff(U, x, 4) - sg*3*sp.diff(U**2, x, 2)
    val = max(abs(sp.N(res.subs({k: kk, x: xx, t: tt}), 40))
              for kk, xx, tt in [(sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 7)),
                                 (sp.Rational(4, 5), -sp.Rational(3, 2), sp.Rational(1, 9))])
    check('2 %s one-soliton solves its equation (residual %s)' % (name, mp.nstr(mp.mpf(str(val)), 3)), val < 1e-35)
    # the bad soliton does NOT solve the equation with the report.tex nonlinear sign
    if name == 'bad':
        res2 = sp.diff(U, t, 2) - sp.diff(U, x, 2) - sp.diff(U, x, 4) + 3*sp.diff(U**2, x, 2)
        val2 = abs(sp.N(res2.subs({k: sp.Rational(1, 3), x: sp.Rational(1, 2), t: sp.Rational(2, 7)}), 30))
        check('2 positive bad soliton fails v_tt = v_xx + v_xxxx - 3(v^2)_xx', val2 > 1e-6)


# ------------------------------------------------------------------ 3. charges by quadrature
def charges_num(kk, V, bad):
    """Soliton u = (k^2/2) sech^2(k xi/2), travelling with velocity V: phi_t = -V u."""
    uf = lambda X: kk**2/2*mp.sech(kk*X/2)**2
    ux = lambda X: -kk**3/2*mp.sech(kk*X/2)**2*mp.tanh(kk*X/2)
    a, b = (-1, 1) if bad else (1, -1)
    M = mp.quad(uf, [-mp.inf, 0, mp.inf])
    Pi = mp.quad(lambda X: -V*uf(X), [-mp.inf, 0, mp.inf])
    P = mp.quad(lambda X: V*uf(X)**2, [-mp.inf, 0, mp.inf])           # -int pi phi_x, pi = -V u
    # H = pi phi_t - L = 1/2 pi^2 + 1/2 phi_x^2 - a phi_x^3 - (b/2) phi_xx^2
    E = mp.quad(lambda X: V**2*uf(X)**2/2 + uf(X)**2/2 - a*uf(X)**3 - mp.mpf(b)/2*ux(X)**2, [-mp.inf, 0, mp.inf])
    return M, Pi, P, E


for bad in (False, True):
    for par in (mp.mpf('0.37'), mp.mpf('0.9'), mp.mpf('1.3')):
        if not bad and par > mp.pi/2:
            continue
        for eps in (1, -1):
            if bad:
                kk, V = mp.sinh(par), eps*mp.cosh(par)
                Ef = 2*kk**3*(5 + 4*kk**2)/15
            else:
                kk, V = mp.sin(par), eps*mp.cos(par)
                Ef = 2*kk**3*(5 - 4*kk**2)/15
            M, Pi, P, E = charges_num(kk, V, bad)
            err = max(abs(M - 2*kk), abs(Pi + V*M), abs(P - 2*kk**3*V/3), abs(E - Ef))
            check('3 %s par=%s eps=%+d: M, Pi, P, E closed forms (err %s)'
                  % ('bad ' if bad else 'good', mp.nstr(par, 3), eps, mp.nstr(err, 3)), err < TOL)
            if not bad:
                E_oneloop = kk**3*(2*mp.cos(par)**2 + 1)/3
                check('3 good: oneloop_phase.tex energy (1/3)k^3(2cos^2a+1) differs from quadrature (diff %s)'
                      % mp.nstr(E_oneloop - E, 5), abs(E_oneloop - E) > 1e-3)

# dE/dP = v symbolically
al, be = sp.symbols('alpha beta', positive=True)
eps_s = sp.Symbol('epsilon')
Pg = sp.Rational(2, 3)*eps_s*sp.sin(al)**3*sp.cos(al); Eg = sp.Rational(2, 15)*sp.sin(al)**3*(5 - 4*sp.sin(al)**2)
Pb = sp.Rational(2, 3)*eps_s*sp.sinh(be)**3*sp.cosh(be); Eb = sp.Rational(2, 15)*sp.sinh(be)**3*(5 + 4*sp.sinh(be)**2)
r_g = sp.simplify(sp.diff(Eg, al) - eps_s*sp.cos(al)*sp.diff(Pg, al)).subs(eps_s**2, 1)
r_b = sp.simplify(sp.diff(Eb, be) - eps_s*sp.cosh(be)*sp.diff(Pb, be)).subs(eps_s**2, 1)
check('3 dE/dP = v (good, symbolic)', sp.simplify(r_g) == 0)
check('3 dE/dP = v (bad, symbolic)', sp.simplify(r_b) == 0)
Eol = sp.sin(al)**3*(2*sp.cos(al)**2 + 1)/3
check('3 oneloop_phase.tex energy violates dE/dP = v', sp.simplify(sp.diff(Eol, al) - sp.cos(al)*sp.diff(Pg.subs(eps_s, 1), al)) != 0)

# 9. derivative and Darboux coefficients
check('9 dP/da = (2/3) eps sin a sin 3a', sp.simplify(sp.expand_trig(sp.diff(Pg, al) - sp.Rational(2, 3)*eps_s*sp.sin(al)*sp.sin(3*al))) == 0)
check('9 dP/db = (2/3) eps sinh b sinh 3b', sp.simplify(sp.expand_trig(sp.diff(Pb, be) - sp.Rational(2, 3)*eps_s*sp.sinh(be)*sp.sinh(3*be))) == 0)
Pig = -2*sp.sin(al)*eps_s*sp.cos(al)
gam = sp.diff(Pig, al)/sp.diff(Pg, al)
check('9 gamma = -3 cos 2a/(sin a sin 3a)', sp.simplify(sp.expand_trig(gam + 3*sp.cos(2*al)/(sp.sin(al)*sp.sin(3*al)))) == 0)


# ------------------------------------------------------------------ data and phases (mpmath)
def good(a, eps):
    kk = mp.sin(a); v = eps*mp.cos(a)
    return dict(k=kk, v=v, w=kk*v, M=2*kk, Pi=-2*kk*v, P=mp.mpf(2)/3*kk**3*v, l=-eps*mp.cos(3*a),
                E=2*kk**3*(5 - 4*kk**2)/15)


def bad(b, eps):
    kk = mp.sinh(b); v = eps*mp.cosh(b)
    return dict(k=kk, v=v, w=kk*v, M=2*kk, Pi=-2*kk*v, P=mp.mpf(2)/3*kk**3*v, l=eps*mp.cosh(3*b),
                E=2*kk**3*(5 + 4*kk**2)/15)


def Aof(s1, s2, isbad):
    sg = -1 if isbad else 1
    Pp = lambda K, W: W**2 - K**2 + sg*K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


def delta(s1, s2, isbad):
    return mp.mpf(2)/9*(s1['l'] - s2['l'])*mp.log(Aof(s1, s2, isbad)) + mp.mpf(2)/3*s1['M']*s2['M']*(s1['v'] - s2['v'])


r3 = mp.sqrt(3)

# ------------------------------------------------------------------ 4. KP roots
def roots_good(a, eps):
    kk = mp.sin(a); c = eps*mp.cos(a)
    return (kk + c/r3)/2, (c/r3 - kk)/2


ellp = lambda p: -3*r3*p*(1 - 4*p**2)
worst = 0
for a in (mp.mpf('0.3'), mp.mpf('0.8'), mp.mpf('1.4')):
    for eps in (1, -1):
        pa, pb = roots_good(a, eps)
        s = good(a, eps)
        mu1, mu2 = (s['k'] - s['v']/r3)/2, (-s['k'] - s['v']/r3)/2          # Phase 1: mu1-mu2=k, mu1+mu2=-v/sqrt3
        lam = mu1**3 - mu1/4
        worst = max(worst, abs(pa**2 + pa*pb + pb**2 - mp.mpf(1)/4), abs(ellp(pa) - s['l']), abs(ellp(pb) - s['l']),
                    abs(mu1 + pb), abs(mu2 + pa), abs(lam + s['l']/(12*r3)), abs(mu2**3 - mu2/4 - lam))
check('4 good KP roots: cubic, ell(p), mu = -p, lambda = -ell/(12 sqrt3) (err %s)' % mp.nstr(worst, 3), worst < TOL)
worst = 0
for (a1, a2) in [(mp.mpf('0.3'), mp.mpf('0.7')), (mp.mpf('0.2'), mp.mpf('0.5'))]:
    for e2 in (1, -1):
        p1a, p1b = roots_good(a1, 1); p2a, p2b = roots_good(a2, e2)
        cr = (p1a - p2a)*(p1b - p2b)/((p1a - p2b)*(p1b - p2a))
        worst = max(worst, abs(cr - Aof(good(a1, 1), good(a2, e2), False)))
check('4 good A_12 = KP cross-ratio, both orientations (err %s)' % mp.nstr(worst, 3), worst < TOL)

# ------------------------------------------------------------------ 5. dictionary alpha = i beta
worst = 0
for (b1, b2) in [(mp.mpf('0.4'), mp.mpf('1.1')), (mp.mpf('0.9'), mp.mpf('0.25'))]:
    for e2 in (1, -1):
        g1, g2 = good(1j*b1, 1), good(1j*b2, e2)
        h1, h2 = bad(b1, 1), bad(b2, e2)
        worst = max(worst, abs(g1['k'] - 1j*h1['k']), abs(g1['M'] - 1j*h1['M']), abs(g1['Pi'] - 1j*h1['Pi']),
                    abs(g1['P'] + 1j*h1['P']), abs(g1['E'] + 1j*h1['E']), abs(g1['l'] + h1['l']),
                    abs(g1['v'] - h1['v']), abs(g2['l'] + h2['l']),
                    abs(Aof(g1, g2, False) - Aof(h1, h2, True)),
                    abs(delta(g1, g2, False) + delta(h1, h2, True)))
check('5 alpha = i beta maps good data to bad data; delta_good -> -delta_bad (err %s)' % mp.nstr(worst, 3), worst < TOL)

# ------------------------------------------------------------------ 6. unitarity and crossing
worst_u = worst_cb = worst_cg = 0
for (x1, x2) in [(mp.mpf('1.1'), mp.mpf('0.4')), (mp.mpf('0.3'), mp.mpf('0.2')), (mp.mpf('2.0'), mp.mpf('0.9'))]:
    s1, s2 = bad(x1, 1), bad(x2, 1)
    worst_u = max(worst_u, abs(delta(s1, s2, True) + delta(s2, s1, True)))
    lhs = delta(bad(x1, 1), bad(x2 + 1j*mp.pi, 1), True)
    worst_cb = max(worst_cb, abs(lhs + delta(bad(x1, 1), bad(x2, -1), True)))
# good: head-on points must be regular (alpha1+alpha2 < pi/3 or |alpha1-alpha2| > pi/3); a first choice
# (0.9, 0.15) lay in the singular head-on region A_ho < 0, where the identity holds only modulo the
# log branch (recorded in the paper's failure list).
for (x1, x2) in [(mp.mpf('0.5'), mp.mpf('0.3')), (mp.mpf('0.2'), mp.mpf('0.1')), (mp.mpf('1.3'), mp.mpf('0.2'))]:
    s1, s2 = good(x1, 1), good(x2, 1)
    worst_u = max(worst_u, abs(delta(s1, s2, False) + delta(s2, s1, False)))
    lhs = delta(good(x1, 1), good(x2 + mp.pi, 1), False)
    worst_cg = max(worst_cg, abs(lhs + delta(good(x1, 1), good(x2, -1), False)))
check('6 unitarity delta(1,2) = -delta(2,1), both equations (err %s)' % mp.nstr(worst_u, 3), worst_u < TOL)
check('6 crossing bad: delta_ov(b1, b2 + i pi) = -delta_ho(b1, b2) (err %s)' % mp.nstr(worst_cb, 3), worst_cb < TOL)
check('6 crossing good: delta_ov(a1, a2 + pi) = -delta_ho(a1, a2) (err %s)' % mp.nstr(worst_cg, 3), worst_cg < TOL)

# ------------------------------------------------------------------ 7. small-amplitude limits
lam_ = mp.mpf(10)**-6
b1, b2 = mp.mpf('1.7')*lam_, lam_
s1, s2 = bad(b1, 1), bad(b2, 1)
A = Aof(s1, s2, True)
dd = mp.diff(lambda y: delta(bad(y, 1), s2, True), b1)
ratio = dd/(2*b1*mp.log(A))
AK = ((b1 - b2)/(b1 + b2))**2
pred = 1 + 2*b2/(b1*mp.log(AK))
cross_part = -mp.diff(lambda y: bad(y, 1)['Pi'], b1)*s2['M']
check('7 overtaking: ratio %s -> 1 + 2b2/(b1 log A_KdV) = %s' % (mp.nstr(ratio, 8), mp.nstr(pred, 8)),
      abs(ratio - pred) < 1e-5 and abs(pred - mp.mpf('0.5642')) < 1e-4)
check('7 cross-term part -(dPi_1/db1) M_2 / (4 b2) -> 1 (%s)' % mp.nstr(cross_part/(4*b2), 8), abs(cross_part/(4*b2) - 1) < 1e-6)
for isbad, mk in ((True, bad), (False, good)):
    h1, h2 = mk(b1, 1), mk(b2, -1)
    top = mp.mpf(2)/3*h1['M']*h2['M']*(h1['v'] - h2['v'])
    dyn = mp.mpf(2)/9*(h1['l'] - h2['l'])*mp.log(Aof(h1, h2, isbad))
    e = max(abs(top/(b1*b2) - mp.mpf(16)/3), abs(dyn/(b1*b2) + mp.mpf(4)/3))
    check('7 head-on %s: delta -> (16/3 - 4/3) b1 b2 (err %s)' % ('bad' if isbad else 'good', mp.nstr(e, 3)), e < 1e-5)

# ------------------------------------------------------------------ 8. Bogdan-Charkina invariant
worst = 0
for a, b in [(mp.mpf('0.8'), mp.mpf('1.2')), (mp.mpf('0.3'), mp.mpf('1.1'))]:
    h1, h2 = bad(a + 1j*b, 1), bad(a - 1j*b, 1)
    J = 1j*mp.mpf(2)/9*(h1['l'] - h2['l'])
    worst = max(worst, abs(J + mp.mpf(4)/9*mp.sinh(3*a)*mp.sin(3*b)))
check('8 i(2/9)(l1 - l2) = -(4/9) sinh 3a sin 3b (err %s)' % mp.nstr(worst, 3), worst < TOL)

# ------------------------------------------------------------------ 10. defect factor conventions
worst = 0; worst2 = 0
for (a, eps, s, eta) in [(mp.mpf('0.4'), 1, mp.mpf('0.7'), 1), (mp.mpf('0.4'), 1, mp.mpf('0.7'), -1),
                         (mp.mpf('0.9'), -1, mp.mpf('-1.5'), 1), (mp.mpf('0.9'), -1, mp.mpf('-1.5'), -1)]:
    kk, v = mp.sin(a), eps*mp.cos(a)
    pa, pb = roots_good(a, eps)
    q = -s/(2*r3)
    Acz = (v + s)/kk
    zcz = (Acz - eta*r3)/(Acz + eta*r3)
    root = ((pb - q)/(pa - q))**eta
    mid = ((kk - v/r3 - s/r3)/(kk + v/r3 + s/r3))**(-eta)
    worst = max(worst, abs(zcz - root), abs(ellp(q) - (3*s - s**3)/2))
    worst2 = max(worst2, abs(mid + 1/root))
check('10 Phase 2c: z_CZ = ((p_b-q)/(p_a-q))^eta, ell(q) = (3s - s^3)/2 (err %s)' % mp.nstr(worst, 3), worst < TOL)
check('10 oneloop_defect_phase.tex (z1_sol) middle form = -1/(root form) (err %s)' % mp.nstr(worst2, 3), worst2 < TOL)

print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

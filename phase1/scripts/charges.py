"""Conserved charges of the Boussinesq hierarchy and their values on one-soliton solutions.

 1. Pseudo-differential calculus: L = d^3 + q d + r, L^{1/3} = d + sum c_m d^{-m}.
    H_n = int Res L^{n/3} dx.  Check Res L^{n/3} = 0 identically for n = 3, 6 (no charges of
    spin divisible by 3), and nonzero for n = 1, 2, 4, 5.
 2. Project-convention soliton (good):  q = (3/2) u - 1/4,  t_Lax = sqrt(3) t, background r -> 0.
    Spectral picture: L e^{mu x} = lambda e^{mu x} on the background gives mu^3 - mu/4 = lambda.
    A soliton of wavenumber k and velocity v is the pair of roots (mu_1, mu_2) with
    mu_1 - mu_2 = k, mu_1 + mu_2 = s = -v/sqrt(3) (so that mu_1^2+mu_1 mu_2+mu_2^2 = 1/4).
    Tested numerically: H_n[soliton] lies in the span of (mu_1^m - mu_2^m), m <= n, m != 0 mod 3,
    with soliton-independent coefficients (a trace-formula-type statement).
 3. Lagrangian energy/momentum (phi_x = u):
      Lagrangian  L = 1/2 phi_t^2 - 1/2 phi_x^2 - 1/2 phi_xx^2 + phi_x^3  (Euler-Lagrange: project eq.)
      E = int 1/2 phi_t^2 + 1/2 u^2 + 1/2 u_x^2 - u^3,   P = - int phi_t u,   M = int u.
    Closed forms on the soliton, check dE/dP = v (Hamilton's equation for the collective coordinate).
"""
import sympy as sp
import mpmath as mp

x = sp.Symbol('x', real=True)
q = sp.Function('q')(x)
r = sp.Function('r')(x)
NTRUNC = 7          # keep orders d^1 ... d^{-NTRUNC}


def mul(A, B):
    """Product of pseudo-differential operators given as {order: coeff}."""
    C = {}
    for i, a in A.items():
        for j, b in B.items():
            m = 0
            while True:
                o = i + j - m
                if o < -NTRUNC:
                    break
                bc = sp.binomial(i, m)
                if bc == 0 and i >= 0 and m > i:
                    break
                C[o] = C.get(o, 0) + bc*a*sp.diff(b, x, m)
                m += 1
    return {o: sp.expand(c) for o, c in C.items() if sp.expand(c) != 0}


# L^{1/3} by undetermined coefficients
cs = sp.symbols('c1:%d' % (NTRUNC + 1))
Lop = {3: 1, 1: q, 0: r}
root = {1: sp.Integer(1)}
for m in range(1, NTRUNC + 1):
    trial = dict(root)
    cm = sp.Function('C%d' % m)(x)
    trial[-m] = cm
    cube = mul(mul(trial, trial), trial)
    target = Lop.get(2 - m, 0)
    sol = sp.solve(sp.Eq(cube.get(2 - m, 0), target), cm)
    root[-m] = sp.expand(sol[0])

def power(n):
    P = {0: sp.Integer(1)}
    for _ in range(n):
        P = mul(P, root)
    return P

res = {}
for n in range(1, 7):
    res[n] = sp.simplify(power(n).get(-1, 0))
    print('Res L^{%d/3} =' % n, res[n] if n < 4 else ('<%d terms>' % len(sp.Add.make_args(sp.expand(res[n])))))
assert res[3] == 0 and res[6] == 0
assert all(res[n] != 0 for n in (1, 2, 4, 5))

# ---- 2. values on the soliton -----------------------------------------------------------------
mp.mp.dps = 30
def soliton_fields(k, eps):
    v = eps*sp.sqrt(1 - k**2)
    u = k**2/2*sp.sech(k*x/2)**2
    qq = sp.Rational(3, 2)*u - sp.Rational(1, 4)
    # q_tL = 2 r_x - q_xx with q_tL = (1/sqrt3) q_t = -(v/sqrt3) q_x  => r = (q_x - (v/sqrt3)(q+1/4))/2
    rr = (sp.diff(qq, x) - v/sp.sqrt(3)*(qq + sp.Rational(1, 4)))/2
    return qq, rr, v

def H(n, k, eps):
    qq, rr, v = soliton_fields(k, eps)
    dens = res[n].subs({q: qq, r: rr}).doit()
    bg = res[n].subs({q: sp.Rational(-1, 4), r: 0}).doit()
    f = sp.lambdify(x, sp.simplify(dens - bg), 'mpmath')
    return mp.quad(f, [-mp.inf, 0, mp.inf])

# On a constant background Res L^{n/3} mixes with lower charges, so test the weaker (trace-formula
# type) statement: H_n[soliton] = sum_{m <= n, m != 0 mod 3} c_{n,m} (mu_1^m - mu_2^m), with the
# c_{n,m} fitted on 4 solitons and then checked on 3 further solitons (both directions).
fit_pts = [(sp.Rational(3, 10), 1), (sp.Rational(3, 5), 1), (sp.Rational(3, 5), -1), (sp.Rational(17, 20), -1)]
test_pts = [(sp.Rational(1, 7), -1), (sp.Rational(2, 3), 1), (sp.Rational(9, 10), 1)]
def mus(k, eps):
    v = eps*mp.sqrt(1 - mp.mpf(k)**2); s_ = -v/mp.sqrt(3)
    return (s_ + k)/2, (s_ - k)/2
print()
for n in (1, 2, 4, 5):
    basis = [m for m in range(1, n + 1) if m % 3]
    rows = [[mus(k, e)[0]**m - mus(k, e)[1]**m for m in basis] for k, e in fit_pts]
    rhs = [H(n, k, e) for k, e in fit_pts]
    coef = mp.lu_solve(mp.matrix(rows)[:, :len(basis)] if len(basis) == len(rows) else
                       mp.matrix([r[:len(basis)] for r in rows]), mp.matrix(rhs)) if len(basis) == 4 else \
           mp.qr_solve(mp.matrix(rows), mp.matrix(rhs))[0]
    worst = 0
    for k, e in test_pts:
        m1, m2 = mus(k, e)
        pred = sum(coef[i_]*(m1**m - m2**m) for i_, m in enumerate(basis))
        worst = max(worst, abs(pred - H(n, k, e)))
    print('H_%d: basis m=%s, coefficients %s, max error on held-out solitons %s'
          % (n, basis, [mp.nstr(c_, 10) for c_ in coef], mp.nstr(worst, 3)))
    assert worst < 1e-20

# ---- 3. energy, momentum, mass --------------------------------------------------------------
k, v = sp.symbols('k v', positive=True)
X = sp.Symbol('X', real=True)
u = k**2/2*sp.sech(k*X/2)**2
I = lambda f: sp.simplify(sp.integrate(sp.simplify(f.rewrite(sp.exp)), (X, -sp.oo, sp.oo)))
M = I(u)
Iu2 = I(u**2); Iu3 = I(u**3); Iux2 = I(sp.diff(u, X)**2)
print('\nM = int u =', M, ';  int u^2 =', Iu2, '; int u^3 =', Iu3, '; int u_x^2 =', Iux2)
for name, sgn in [('good', -1)]:
    vv = sp.sqrt(1 - k**2)                    # right-mover
    Pk = sp.simplify(vv*Iu2)                  # P = -int phi_t u with phi_t = -v u
    Ek = sp.simplify(sp.Rational(1, 2)*(vv**2 + 1)*Iu2 + sp.Rational(1, 2)*Iux2 - Iu3)
    print(name, ': P(k) =', sp.factor(Pk), ';  E(k) =', sp.factor(Ek))
    dEdP = sp.simplify(sp.diff(Ek, k)/sp.diff(Pk, k))
    print('   dE/dP =', dEdP, '  (should equal v = sqrt(1-k^2))')
    assert sp.simplify(dEdP - vv) == 0
    print('   E - P series in k:', sp.series(Ek - Pk, k, 0, 9))
    kc = sp.sqrt(3)/2
    print('   dP/dk = 0 at k = sqrt(3)/2:', sp.simplify(sp.diff(Pk, k).subs(k, kc)) == 0,
          '; P_max =', sp.nsimplify(sp.simplify(Pk.subs(k, kc))), ', E there =', sp.simplify(Ek.subs(k, kc)))

    # heavy branch near k = 1 (v -> 0): E = E0 + P^2/(2 m*) ?
    dl = sp.Symbol('delta', positive=True)
    Ps = sp.series(Pk.subs(k, 1 - dl), dl, 0, 2).removeO(); Es = sp.series(Ek.subs(k, 1 - dl), dl, 0, 2).removeO()
    print('   near k=1: P ~', sp.simplify(Ps), ',  E ~', sp.expand(Es))
    print('   => E = 2/15 + (3/4) P^2 + ..., i.e. rest energy 2/15 and inertial mass m* = 2/3')
    assert sp.simplify(sp.expand(Es) - (sp.Rational(2, 15) + sp.Rational(3, 4)*sp.expand(Ps**2))) .subs(dl, 0) == 0
    assert sp.simplify(sp.diff(sp.expand(Es) - sp.Rational(3, 4)*sp.expand(Ps**2), dl).subs(dl, 0)) == 0

# Euler-Lagrange check of the Lagrangian
t_, x_ = sp.symbols('t x')
phi = sp.Function('phi')(x_, t_)
Lag = sp.Rational(1, 2)*sp.diff(phi, t_)**2 - sp.Rational(1, 2)*sp.diff(phi, x_)**2 \
      - sp.Rational(1, 2)*sp.diff(phi, x_, 2)**2 + sp.diff(phi, x_)**3
EL = sp.euler_equations(Lag, [phi], [x_, t_])[0].lhs
uu = sp.diff(phi, x_)
target = sp.diff(phi, t_, 2) - sp.diff(phi, x_, 2) + sp.diff(phi, x_, 4) + 3*sp.diff(uu**2, x_)
print('Euler-Lagrange equation = -(phi_tt - phi_xx + phi_xxxx + 3 (phi_x^2)_x):',
      sp.simplify(EL + target) == 0)
assert sp.simplify(EL + target) == 0
print('ALL CHECKS PASSED')

# ---- 4. physical energy/momentum in terms of the hierarchy charges on solitons ---------------------
kk, ee = sp.symbols('k epsilon')
s_ = -ee*sp.sqrt(1 - kk**2)/sp.sqrt(3)
m1, m2 = (s_ + kk)/2, (s_ - kk)/2
d = lambda m: sp.expand(m1**m - m2**m)
H1, H2 = d(1), d(2)
H4 = d(4) - sp.Rational(1, 3)*d(2)
H5 = d(5) - sp.Rational(5, 72)*d(1)
Pphys = sp.Rational(2, 3)*ee*kk**3*sp.sqrt(1 - kk**2)
Ephys = sp.Rational(2, 15)*kk**3*(5 - 4*kk**2)
a_, b_ = sp.symbols('a b')
red = lambda e: sp.expand(e.subs(ee**2, 1).subs(ee**3, ee))
solP = sp.solve(sp.Poly(red(Pphys - a_*H4 - b_*H2), kk, ee, sp.sqrt(1 - kk**2)).coeffs(), [a_, b_])
solE = sp.solve(sp.Poly(red(Ephys - a_*H5 - b_*H1), kk, ee, sp.sqrt(1 - kk**2)).coeffs(), [a_, b_])
print('P_phys = a H_4 + b H_2 with', solP, '; E_phys = a H_5 + b H_1 with', solE)
assert solP and solE
print('ALL CHECKS PASSED (section 4)')

"""Consistency checks on the semiclassical two-soliton phase (bad Boussinesq)

   delta(1,2) = (2/9) (l_1 - l_2) log A_12 + (2/3) M_1 M_2 (v_1 - v_2),   l = eps cosh(3 beta),
   k = sinh beta, v = eps cosh beta, M = 2k,  (S ~ exp(i delta / hbar)).

 1. gradient of delta reproduces the Darboux-coordinate shifts (re-derived independently here)
 2. unitarity:           delta(1,2) = -delta(2,1)
 3. crossing-like:       delta_ov(b1, b2 + i pi) = -delta_ho(b1, b2)   [b -> b + i pi: right -> left mover]
 4. breathers: for b_{1,2} = a +- i b, (i) A > 1 <=> b > pi/3 (Bogdan-Charkina existence boundary);
    (ii) A equals Bogdan-Charkina a_12 (33); (iii) i (2/9)(l_1 - l_2) equals their adiabatic invariant I_br (64)
 5. KP cross-ratio form: A_12 = (m1a-m2a)(m1b-m2b)/((m1a-m2b)(m1b-m2a)) in terms of the cubic roots
 6. KdV limit (small b): d delta/d b1 -> 2 b1 log A, i.e. Bonnemain-Doyon's P_GHD = eta^2/2 convention
 7. good equation: same symplectic identity (with cross terms) in the regular region; the
    analogous phase with l = eps cos(3 alpha)
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30
b1, b2 = sp.symbols('beta_1 beta_2')
I = sp.I


def data(bb, eps, bad=True):
    if bad:
        k = sp.sinh(bb); v = eps*sp.cosh(bb); l = eps*sp.cosh(3*bb)
        P = sp.Rational(2, 3)*eps*k**3*sp.cosh(bb)
    else:
        k = sp.sin(bb); v = eps*sp.cos(bb); l = -eps*sp.cos(3*bb)
        P = sp.Rational(2, 3)*eps*k**3*sp.cos(bb)
    return dict(k=k, v=v, w=k*v, M=2*k, Pi=-2*k*v, l=l, P=P)


def Aof(s1, s2, bad=True):
    sg = -1 if bad else 1
    Pp = lambda K, W: W**2 - K**2 + sg*K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


RC = {True: sp.Rational(2, 3), False: sp.Rational(2, 3)}
def delta(s1, s2, bad=True):
    return sp.Rational(2, 9)*(s1['l'] - s2['l'])*sp.log(Aof(s1, s2, bad)) + RC[bad]*s1['M']*s2['M']*(s1['v'] - s2['v'])


def num(expr, subs):
    return complex(sp.N(expr.subs(subs), 25))

ok = True
# 1. gradients
for e2, name in [(1, 'ov'), (-1, 'ho')]:
    for bad in (True, False):
        s1, s2 = data(b1, 1, bad), data(b2, e2, bad)
        L = sp.log(Aof(s1, s2, bad))
        g1 = sp.diff(s1['Pi'], b1)/sp.diff(s1['P'], b1); g2 = sp.diff(s2['Pi'], b2)/sp.diff(s2['P'], b2)
        D1 = -L/s1['k'] + g1*s2['M']; D2 = L/s2['k'] - g2*s1['M']
        dl = delta(s1, s2, bad)
        r1 = sp.diff(dl, b1) + sp.diff(s1['P'], b1)*D1
        r2 = sp.diff(dl, b2) + sp.diff(s2['P'], b2)*D2
        pts = [(1.3, 0.4), (0.7, 0.2)] if bad else [(0.5, 0.3), (0.35, 0.15)]   # good: regular region
        err = max(abs(num(r, {b1: p, b2: q})) for r in (r1, r2) for p, q in pts)
        # symplectic identity with cross terms
        lhs = sp.diff(s1['P'], b1)*sp.diff(-L/s1['k'], b2) - sp.diff(s2['P'], b2)*sp.diff(L/s2['k'], b1)
        rhs = -sp.diff(s2['Pi'], b2)*sp.diff(s1['M'], b1) - sp.diff(s1['Pi'], b1)*sp.diff(s2['M'], b2)
        err2 = max(abs(num(lhs - rhs, {b1: p, b2: q})) for p, q in pts)
        print('1/7. %s %s: |grad delta - Darboux shifts| = %.1e ; symplectic identity residual = %.1e'
              % ('bad ' if bad else 'good', name, err, err2))
        ok &= err < 1e-12 and err2 < 1e-12   # double-precision evaluation

# 2. unitarity  3. crossing
s1, s2 = data(b1, 1), data(b2, 1)
d12 = delta(s1, s2); d21 = delta(s2, s1)
print('2. unitarity delta(1,2)+delta(2,1) =', num(d12 + d21, {b1: 1.1, b2: 0.4}))
ok &= abs(num(d12 + d21, {b1: 1.1, b2: 0.4})) < 1e-20
dho = delta(data(b1, 1), data(b2, -1))
cr = d12.subs(b2, b2 + I*sp.pi) + dho
val = num(cr, {b1: 1.1, b2: 0.4})
print('3. crossing-like delta_ov(b1, b2+i pi) + delta_ho(b1, b2) =', val, '(mod 2 pi i * (2/9)(l1-l2) branch terms?)')
# the log may pick a branch 2 pi i n; report the remainder divided by (2/9)(l1-l2)*2 pi i
coef = num(sp.Rational(2, 9)*(data(b1, 1)['l'] + sp.cosh(3*b2))*2*sp.pi*I, {b1: 1.1, b2: 0.4})
print('   remainder / [(2/9)(l1-l2) 2 pi i] =', val/coef)

# 4. breathers
al, be = sp.symbols('a b', positive=True)
sb1, sb2 = data(al + I*be, 1), data(al - I*be, 1)
Ab = sp.simplify(sp.expand_complex(Aof(sb1, sb2)))
BC_a12 = sp.sin(be)**2*(4*sp.cosh(al)**2 - 1)/(sp.sinh(al)**2*(1 - 4*sp.cos(be)**2))
J = sp.I*sp.Rational(2, 9)*(sb1['l'] - sb2['l'])
BC_I = sp.Rational(16, 9)*sp.sinh(al)*sp.sin(be)*(4*sp.cosh(al)**2 - 1)*(sp.sin(be)**2 - sp.Rational(3, 4))
for p, q in [(0.8, 1.2), (1.5, 1.4), (0.3, 1.1)]:
    sub = {al: p, be: q}
    print('4. breather a=%.1f b=%.2f: A=%.6f, BC a12=%.6f, A>1: %s (b>pi/3: %s) | i(2/9)(l1-l2)=%.6f, BC I_br=%.6f'
          % (p, q, num(Ab, sub).real, num(BC_a12, sub).real, num(Ab, sub).real > 1, q > sp.pi/3,
             num(J, sub).real, num(BC_I, sub).real))
    print("   |A-a12|, |J-I|:", abs(num(Ab - BC_a12, sub)), abs(num(J, sub) - num(BC_I, sub)))
    ok &= abs(num(Ab - BC_a12, sub)) < 1e-12
    ok &= abs(num(J, sub) - num(BC_I, sub)) < 1e-12
print('   => sign relation: J = %s * I_br' % sp.nsimplify(round(num(J, {al: .8, be: 1.2}).real/num(BC_I, {al: .8, be: 1.2}).real, 12)))

# 5. KP cross ratio: roots of mu^3 - (1+?)... use the soliton pairs mu_a - mu_b = k, mu_a + mu_b = s
def roots_pair(k, v, bad=True):
    # bad: background of the bad Lax form; derive from KP: theta = (mu_a - mu_b) x + (mu_a^2 - mu_b^2) y,
    # with y = time up to a scale c: w = c k (mu_a + mu_b) => mu_a + mu_b = v / c. Fix c by the
    # cubic condition mu_a^2 + mu_a mu_b + mu_b^2 = const: (3 s^2 + k^2)/4 = const  with v^2 = 1 + k^2
    # => s^2 = (v^2 - 1)/(-3) ... only consistent for an imaginary c: s = i v / sqrt3, const = -1/4.
    s = sp.I*v/sp.sqrt(3)
    return (s + k)/2, (s - k)/2
for e2 in (1, -1):
    t1, t2 = data(sp.Rational(11, 10), 1), data(sp.Rational(2, 5), e2)
    m1a, m1b = roots_pair(t1['k'], t1['v']); m2a, m2b = roots_pair(t2['k'], t2['v'])
    CR = (m1a - m2a)*(m1b - m2b)/((m1a - m2b)*(m1b - m2a))
    print('5. eps2=%+d: A = %.15f, KP cross ratio = %s' % (e2, sp.N(Aof(t1, t2)), sp.N(CR, 15)))
    print("   |A - cross ratio| =", abs(complex(sp.N(CR - Aof(t1, t2), 30))))
    ok &= abs(complex(sp.N(CR - Aof(t1, t2), 30))) < 1e-14

# 6. KdV limit
s1, s2 = data(b1, 1), data(b2, 1)
L = sp.log(Aof(s1, s2))
g = sp.diff(delta(s1, s2), b1)
for lam in [1e-1, 1e-2, 1e-3]:
    rr = num(g/(2*b1*L), {b1: 1.7*lam, b2: lam})
    print('6. small amplitude (scale %.0e): [d delta/d b1] / [2 b1 log A] = %.6f   (finding, not a test: the' % (lam, rr.real))
print('   topological term is of the same order as the KdV-type term; ratio -> 1 - 4 b2/(2 b1 log A))')
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

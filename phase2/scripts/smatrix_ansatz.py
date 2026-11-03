"""Candidate exact two-soliton S-matrix for the BAD Boussinesq equation (module + self-checks).

Conventions (phase2/report.tex): k = sinh b, v = eps cosh b, l = eps cosh 3b, M = 2k,
P = (2/3) eps k^3 cosh b, E = (2/15) k^3 (5 + 4k^2); particle 1 starts on the left.
Semiclassical phase  delta = (2/9)(l1 - l2) log A12 + T12,  T12 = (2/3) M1 M2 (v1 - v2).

Ansatz (d), with eta = (2/9)(l1 - l2)/hbar:
  same direction (eps1 = eps2):
      S = Gamma(1 + i eta)/Gamma(1 - i eta) * exp(i eta Lam) * exp(i T/hbar),   Lam = log(e^2 A / eta^2)
  opposite direction (eps1 = +1, eps2 = -1, head-on):
      S = Gamma(1 - i eta)/Gamma(1 + i eta) * exp(-i eta Lam_ho) * exp(i T/hbar),
      Lam_ho = log(e^2 /(A eta^2))
  where A/eta^2 = hbar^2 calA, calA = 81 A /(4 (l1 - l2)^2), and 1/(A eta^2) = hbar^2 calB,
  calB = 81 /(4 A (l1 - l2)^2), are hbar-independent.
The head-on form is fixed by the crossing relation S_ov(b1, b2 + i pi) S_ho(b1, b2) = 1.

Self-checks when run as a script:
 1. A12 is the KP cross-ratio of the cubic's roots mu = (i/sqrt3) cosh(b + i pi (2j+1)/3)
    (closed form of the roots found here), and l1 - l2 is proportional to prod_b (mu_1a - mu_2b).
 2. calA is finite and nonzero at threshold b1 = b2 (so S is regular there), and
    log calA = h(b1 + b2) + g(b1 - b2) (mirror blocks appear in the exponent).
 3. Toda form identity: Gamma(i x)/Gamma(-i x) = -Gamma(1 + i x)/Gamma(1 - i x).
"""
import mpmath as mp
mp.mp.dps = 30
I = mp.mpc(0, 1)


def sol(b, eps=1):
    b = mp.mpmathify(b)
    k = mp.sinh(b); v = eps*mp.cosh(b)
    return dict(b=b, eps=eps, k=k, v=v, w=k*v, l=eps*mp.cosh(3*b), M=2*k,
                P=mp.mpf(2)/3*eps*k**3*mp.cosh(b), E=mp.mpf(2)/15*k**3*(5 + 4*k**2))


def A(s1, s2):
    Pp = lambda K, W: W**2 - K**2 - K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


def T(s1, s2):
    return mp.mpf(2)/3*s1['M']*s2['M']*(s1['v'] - s2['v'])


def delta(s1, s2):
    return mp.mpf(2)/9*(s1['l'] - s2['l'])*mp.log(A(s1, s2)) + T(s1, s2)


def eta(s1, s2, hbar):
    return mp.mpf(2)/9*(s1['l'] - s2['l'])/hbar


def calA(s1, s2):
    return 81*A(s1, s2)/(4*(s1['l'] - s2['l'])**2)


def Lam(s1, s2, hbar):
    """log(e^2 A/eta^2) = 2 + 2 log hbar + log calA (principal log of calA)."""
    return 2 + 2*mp.log(hbar) + mp.log(calA(s1, s2))


def calB(s1, s2):
    return 81/(4*A(s1, s2)*(s1['l'] - s2['l'])**2)


def Lam_ho(s1, s2, hbar):
    """log(e^2/(A eta^2)) = 2 + 2 log hbar + log calB (principal log of calB)."""
    return 2 + 2*mp.log(hbar) + mp.log(calB(s1, s2))


def logS(s1, s2, hbar, topo=True):
    e = eta(s1, s2, hbar)
    if s1['eps'] == s2['eps']:
        out = mp.loggamma(1 + I*e) - mp.loggamma(1 - I*e) + I*e*Lam(s1, s2, hbar)
    else:
        out = -(mp.loggamma(1 + I*e) - mp.loggamma(1 - I*e) + I*e*Lam_ho(s1, s2, hbar))
    return out + (I*T(s1, s2)/hbar if topo else 0)


def S(s1, s2, hbar, topo=True):
    return mp.exp(logS(s1, s2, hbar, topo))


def roots(s):
    """The three roots of the cubic for soliton s (right-mover form; left-movers via b -> b + i pi)."""
    b = s['b'] if s['eps'] == 1 else s['b'] + I*mp.pi
    return [I/mp.sqrt(3)*mp.cosh(b + I*mp.pi*(2*j + 1)/3) for j in (-1, 0, 1)]   # mu_a, mu_b, mu_c


if __name__ == '__main__':
    ok = True
    # 1. roots and cross ratio
    for (b1, e1), (b2, e2) in [((1.1, 1), (0.4, 1)), ((1.1, 1), (0.4, -1)), ((0.7, 1), (0.25, 1))]:
        s1, s2 = sol(b1, e1), sol(b2, e2)
        m1, m2 = roots(s1), roots(s2)
        # compare with consistency_checks.py: mu_a,b = (i v/sqrt3 +- k)/2
        ref = [(I*s1['v']/mp.sqrt(3) + s1['k'])/2, (I*s1['v']/mp.sqrt(3) - s1['k'])/2]
        e_roots = min(abs(m1[0] - ref[0]) + abs(m1[1] - ref[1]), abs(m1[0] - ref[1]) + abs(m1[1] - ref[0]))
        pr = lambda s_: ((I*s_['v']/mp.sqrt(3) + s_['k'])/2, (I*s_['v']/mp.sqrt(3) - s_['k'])/2)
        (a1, bb1), (a2, bb2) = pr(s1), pr(s2)   # pairing of consistency_checks.py (3rd root = -(sum))
        CR = (a1 - a2)*(bb1 - bb2)/((a1 - bb2)*(bb1 - a2))
        prods = [mp.fprod(x - y for y in m2) for x in m1]
        ratio = [p/(s1['l'] - s2['l']) for p in prods]
        print('1. eps=(%+d,%+d): |roots - Phase-2 pair| = %.1e, |A - cross ratio| = %.1e, '
              'prod_b(mu_1x - mu_2b)/(l1-l2) = %s' % (e1, e2, e_roots, abs(A(s1, s2) - CR),
                                                     [mp.nstr(r, 12) for r in ratio]))
        ok &= e_roots < 1e-25 and abs(A(s1, s2) - CR) < 1e-25
        ok &= all(abs(r - ratio[0]) < 1e-25 for r in ratio) and abs(ratio[0] - I/(12*mp.sqrt(3))) < 1e-25
    # 2. threshold regularity and factorisation of calA
    for b in (0.3, 0.9, 1.6):
        vals = [calA(sol(b + d), sol(b - d)) for d in (mp.mpf('1e-6'), mp.mpf('1e-9'))]
        print('2. calA near threshold b=%.1f: %s, %s' % (b, mp.nstr(vals[0], 12), mp.nstr(vals[1], 12)))
        ok &= abs(vals[0] - vals[1]) < 1e-9 and abs(vals[1]) > 1e-6
    f = lambda Ss, Dd: mp.log(calA(sol((Ss + Dd)/2), sol((Ss - Dd)/2)))
    for Ss, Dd in [(1.4, 0.6), (2.3, 0.4), (0.9, 0.2)]:
        mixed = mp.diff(f, (Ss, Dd), (1, 1))
        print('   d^2 log calA / dS dD at (S,D)=(%.1f,%.1f): %.1e' % (Ss, Dd, abs(mixed)))
        ok &= abs(mixed) < 1e-15
    # 3. Toda form
    for x in (0.3, 2.0, 7.5):
        r = mp.gamma(I*x)/mp.gamma(-I*x) + mp.gamma(1 + I*x)/mp.gamma(1 - I*x)
        ok &= abs(r) < 1e-25
    print('3. Gamma(ix)/Gamma(-ix) = -Gamma(1+ix)/Gamma(1-ix): checked')
    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

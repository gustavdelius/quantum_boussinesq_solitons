"""One-loop (DHN) energy of the bad-Boussinesq soliton in the Charlier-Lenells-reduced theory.

Box quantisation q L + theta(q) = 2 pi n, theta = arg A(soliton, phonon(q)) (oneloop_phase_shifts.py),
one oscillator per q in (-1, 1), w(q) = |q| sqrt(1 - q^2). Then at fixed classical beta
    E_1(beta) = (hbar/4 pi) int_{-1}^{1} w(q) theta'(q) dq   (+ c M, scheme term)
is unambiguous: w = 0 at q = 0, +-1, so 2 pi jumps of theta and the band edges cost nothing.

Checks / findings:
 1. E_1 is finite; table for 0.01 <= beta <= 2.
 2. Small beta: E_1 -> -(1/2) hbar w(q*), q* = -sqrt3/2: the phonon on the short-wave part
    |q| > 1/sqrt2 of the band has REVERSED group velocity, and at q* it co-moves with the (sonic)
    soliton; its phase shift winds by -2 pi in a window of width O(beta) (one mode captured;
    tails converge only linearly in beta).
    In Toda (w = 2|sin q/2|, v_g <= 1 < V) no such resonance exists.
 3. Scheme term: normal ordering of phi_x^3 w.r.t. the reduced vacuum adds 3<phi_x^2> M,
    <phi_x^2> = hbar/(2 pi) (finite), i.e. + (3 hbar/pi) sinh beta.
 4. The zero-point momentum (hbar/4pi) int q theta' dq is NOT well defined (edge terms q theta at
    q = +-1 oscillate with L), so E_1 is given at fixed classical beta, not at fixed P.
"""
import mpmath as mp
from smatrix_ansatz import sol
from oneloop_phase_shifts import phonon, theta_prime
mp.mp.dps = 20
ok = True
w = lambda q: abs(q)*mp.sqrt(1 - q**2)


def e1(b, pts_extra=()):
    s = sol(mp.mpf(b))
    qs = sorted(set([-1, -0.99, -0.9, -0.8, -0.5, 0, 0.5, 0.9, 0.99, 1] + [p for p in pts_extra if -1 < p < 1]))
    return mp.quad(lambda q: w(q)*theta_prime(s, q), qs)/(4*mp.pi)


def qstar(V):
    # left band: w = -q sqrt(1-q^2), group velocity dw/dq = (2q^2 - 1)/sqrt(1 - q^2) = V
    return mp.findroot(lambda q: (2*q**2 - 1)/mp.sqrt(1 - q**2) - V, (-1 + mp.mpf('1e-12'), -1/mp.sqrt(2)), solver='anderson')


if __name__ == '__main__':
    print('1. E_1/hbar (beta):')
    tab = {}
    for b in ('0.01', '0.03', '0.1', '0.3', '0.7', '1.1', '1.5', '2.0'):
        V = mp.cosh(mp.mpf(b)); qs_ = qstar(V)
        tab[b] = e1(b, (float(qs_) - 0.01, float(qs_), float(qs_) + 0.01, float(mp.mpf(b)), -float(mp.mpf(b))))
        print('   beta=%-5s E1/hbar=%+.10f   (q*=%.6f, -w(q*)/2=%+.6f)' % (b, tab[b], qs_, -w(qs_)/2))
        ok &= mp.isfinite(tab[b])
    print('2. resonance: q*(V=1) = %s, -w(q*)/2 = %s' % (mp.nstr(qstar(1), 10), mp.nstr(-w(qstar(1))/2, 10)))
    ok &= abs(qstar(1) + mp.sqrt(3)/2) < 1e-15
    for b in ('0.01', '0.003', '0.001'):
        s = sol(mp.mpf(b)); qs_ = qstar(mp.cosh(mp.mpf(b)))
        lo, hi = qs_ - mp.mpf('0.08'), qs_ + mp.mpf('0.08')
        wind = mp.quad(lambda q: theta_prime(s, q), [lo, qs_, hi])
        contrib = mp.quad(lambda q: w(q)*theta_prime(s, q), [lo, qs_, hi])/(4*mp.pi)
        print('   beta=%s: theta winding across |q-q*| < 0.08: %s x 2pi ; E1 contribution %s'
              % (b, mp.nstr(wind/(2*mp.pi), 8), mp.nstr(contrib, 8)))
    ok &= abs(wind/(2*mp.pi) + 1) < 0.02 and abs(contrib + w(qstar(1))/2) < 0.005   # at beta = 0.001
    print('   E1(0.01) + w(q*)/2 = %s (-> 0 as beta -> 0, slowly)' % mp.nstr(tab['0.01'] + w(qstar(1))/2, 6))
    ok &= abs(tab['0.01'] + w(qstar(1))/2) < 0.02
    # 3. normal-ordering constant
    phx2 = 2*mp.quad(lambda q: q**2/(2*w(q)), [0, 1])/(2*mp.pi)
    print('3. <phi_x^2>/hbar = %s (1/(2 pi) = %s)' % (mp.nstr(phx2, 12), mp.nstr(1/(2*mp.pi), 12)))
    ok &= abs(phx2 - 1/(2*mp.pi)) < 1e-12
    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

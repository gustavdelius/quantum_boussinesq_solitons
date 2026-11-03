"""Step 0 of the one-loop problem (bad Boussinesq): is a phase-shift (DHN) sum definable?

A phonon e^{i q x - i w t} is the analytic continuation of a soliton to K = sinh(beta) = i q:
  |q| < 1: beta = i asin q        (stable, w^2 = q^2 - q^4 >= 0)
  |q| > 1: beta = i pi/2 + acosh q (unstable, w imaginary)
The soliton-phonon interaction constant A(soliton, phonon) then gives the phonon's asymptotic
phase shift around the soliton (the standard DHN route for integrable models; verified against
the linearised equation in a later script).

Checks:
 1. dispersion: the continued tau-exponent gives w^2 = q^2 - q^4 on both branches eps = +-1.
 2. stable branch: |A| = 1 exactly, so theta(q) = arg A is a real phase shift; spectral parameter
    l = eps cosh 3 beta is real, in [-1, 1].
 3. unstable branch: |A| != 1 (the soliton amplifies/attenuates the mode: no phase shift exists),
    and l is purely imaginary -- i.e. exactly the part of the spectral contour (l in i R, the
    image of Charlier-Lenells' k in [0, i]) on which r_1 = 0 is imposed in their invariant class.
 4. finiteness on the reduced (Charlier-Lenells) phase space: theta(q) is bounded on (-1, 1)
    and the DHN integrand w(q) theta'(q) is integrable (w -> 0 at q -> +-1 and at q -> 0).
"""
import mpmath as mp
from smatrix_ansatz import sol, A, I
mp.mp.dps = 25
ok = True


def phonon(q, eps):
    q = mp.mpf(q)
    return sol(I*mp.asin(q) if abs(q) < 1 else I*mp.pi/2 + mp.acosh(q), eps)


for bs in (0.3, 0.7, 1.4):
    s = sol(bs)
    for eps in (1, -1):
        for q in map(mp.mpf, ("0.02", "0.3", "0.7", "0.95", "0.9999")):
            ph = phonon(q, eps)
            Om = ph['w']                      # tau ~ exp(K x - Om t), K = i q, Om = i w
            ok &= abs(Om**2 - (ph['k']**2 + ph['k']**4)) < 1e-20     # bad-soliton relation
            ok &= abs(-Om**2 - (q**2 - q**4)) < 1e-20                 # => w^2 = q^2 - q^4
            ok &= abs(abs(A(s, ph)) - 1) < 1e-20 and abs(mp.im(ph['l'])) < 1e-20 and abs(ph['l']) <= 1
        for q in (1.01, 1.3, 2.0, 5.0):
            ph = phonon(q, eps)
            ok &= abs(abs(A(s, ph)) - 1) > 1e-3 and abs(mp.re(ph['l'])) < 1e-15
    print('beta_s=%.1f: stable |A|=1 & l real in [-1,1]; unstable |A|!=1 & l imaginary: %s' % (bs, ok))

# 4. DHN integrand on (-1, 1): theta continuous, w theta' integrable
s = sol(0.7)
for eps in (1, -1):
    th = lambda q: mp.im(mp.log(A(s, phonon(q, eps))))
    w = lambda q: mp.sqrt(q**2 - q**4)
    vals = [w(q)*mp.diff(th, q) for q in (mp.mpf('0.001'), mp.mpf('0.5'), mp.mpf('0.99'), mp.mpf('0.99999'))]
    print('eps=%+d: w theta\' at q = 1e-3, 0.5, 0.99, 0.99999: %s' % (eps, [mp.nstr(v, 6) for v in vals]))
    ok &= all(abs(v) < 10 for v in vals)
    for qq in (mp.mpf('0.999'), mp.mpf('0.999999')):
        print('   theta(%s) = %s' % (qq, mp.nstr(th(qq), 10)))
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

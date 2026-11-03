"""Tests of the candidate S-matrix of smatrix_ansatz.py at real rapidities.

 1. semiclassical limit: (hbar/i) log S -> delta + hbar*(pi/2)*sgn_eff + O(hbar^2), i.e. the phase
    of S minus delta/hbar tends to a constant +-pi/2 (Maslov-type, sign = sign of the Gamma ratio
    times sgn eta), with remainder O(hbar). Dropping the topological factor exp(iT/hbar) spoils it.
 2. unitarity: |S| = 1 and S(1,2) S(2,1) = 1 (overtaking and head-on).
 3. crossing: S_ov(b1, b2 + i pi) S_ho(b1, b2) = 1 exactly (this fixes the head-on form), and
    the Phase-2 relation delta_ov(b1, b2 + i pi) = -delta_ho(b1, b2) is recovered at leading order.
 4. real analyticity: S(conj b1, conj b2) = 1/conj S(b1, b2) at generic complex rapidities.
"""
import mpmath as mp
from smatrix_ansatz import sol, S, logS, delta, eta, T, A, I
mp.mp.dps = 30
ok = True
wrap = lambda x: (x + mp.pi) % (2*mp.pi) - mp.pi

pts = [((1.1, 1), (0.4, 1)), ((0.7, 1), (0.25, 1)), ((1.3, 1), (0.9, 1)),
       ((1.1, 1), (0.4, -1)), ((0.5, 1), (0.8, -1)), ((0.3, 1), (0.2, -1))]
print('1. semiclassical limit: r(hbar) = arg S - delta/hbar - (pi/2) sgn(eta) sgn_ch  (mod 2 pi)')
for (b1, e1), (b2, e2) in pts:
    s1, s2 = sol(b1, e1), sol(b2, e2)
    sg = 1 if e1 == e2 else -1
    rs, rnt = [], []
    for h in (mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')):
        c = sg*mp.sign(eta(s1, s2, h))*mp.pi/2
        rs.append(wrap(mp.im(logS(s1, s2, h)) - delta(s1, s2)/h - c))
        rnt.append(wrap(mp.im(logS(s1, s2, h, topo=False)) - delta(s1, s2)/h - c))
    print('   %s b=(%.2f,%.2f): r = %s ; without topological factor: %s'
          % ('ov' if sg == 1 else 'ho', b1, b2, [mp.nstr(r, 3) for r in rs], [mp.nstr(r, 3) for r in rnt]))
    ok &= abs(rs[1]) < 2e-2 and abs(rs[2]/rs[1]) < 0.2        # O(hbar) remainder
    ok &= abs(rnt[2]) > 1e-2                                   # topological term is required
    # leading coefficient: hbar * r -> 0 and r/hbar finite (Stirling 1/(6 eta) term + Lam)
print('2. unitarity')
for (b1, e1), (b2, e2) in pts:
    s1, s2 = sol(b1, e1), sol(b2, e2)
    for h in (mp.mpf('0.3'), mp.mpf('0.05')):
        u1 = abs(abs(S(s1, s2, h)) - 1); u2 = abs(S(s1, s2, h)*S(s2, s1, h) - 1)
        ok &= u1 < 1e-25 and u2 < 1e-25
    print('   b=(%.2f,%+d),(%.2f,%+d): ||S|-1| = %.1e, |S12 S21 - 1| = %.1e' % (b1, e1, b2, e2, u1, u2))
print('3. crossing')
for b1, b2 in [(1.1, 0.4), (0.5, 0.8), (0.3, 0.2), (1.7, 1.2)]:
    for h in (mp.mpf('0.3'), mp.mpf('0.05')):
        sx = sol(mp.mpf(b2) + I*mp.pi, 1)
        c = S(sol(b1), sx, h)*S(sol(b1), sol(b2, -1), h)
        lead = delta(sol(b1), sx) + delta(sol(b1), sol(b2, -1))
        ok &= abs(c - 1) < 1e-20
    ok &= abs(mp.im(lead)) < 1e-20 and abs(wrap(mp.re(lead))) < 1e-20 or abs(lead) < 1e-20
    print('   b=(%.1f,%.1f): |S_ov(b1,b2+i pi) S_ho(b1,b2) - 1| = %.1e ; delta_ov(..+i pi)+delta_ho = %s'
          % (b1, b2, abs(c - 1), mp.nstr(lead, 5)))
print('4. real analyticity')
for z1, z2, e2 in [(0.8 + 0.3j, 0.2 - 0.1j, 1), (1.2 + 0.5j, 0.4 + 0.2j, -1)]:
    h = mp.mpf('0.2')
    a = S(sol(z1), sol(z2, e2), h); b = S(sol(mp.conj(z1)), sol(mp.conj(z2), e2), h)
    print('   |S(conj) conj S - 1| = %.1e' % abs(b*mp.conj(a) - 1)); ok &= abs(b*mp.conj(a) - 1) < 1e-20
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

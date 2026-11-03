"""Pole structure and breather-soliton bootstrap of the candidate S-matrix (smatrix_ansatz.py).

Breather = two right-movers at b1 = a - i b, b2 = a + i b, pi/3 < b < pi/2 (right-moving breathers;
left-moving ones are their parity images).
 1. physical sheet: Im(P1 - P2) > 0 for this labelling (relative momentum k = (P1-P2)/(2 hbar) in
    the upper half plane, particle 1 on the left initially), so bound states = poles of S(1,2) here.
 2. poles: S(1,2) has simple poles exactly where i eta = -N, N = 1, 2, ..., i.e.
    I_br(a, b) = -(4/9) sinh 3a sin 3b = N hbar  (Bogdan-Charkina Bohr-Sommerfeld condition, Phase 2
    Prop. 3.4); no pole at N = 0 (threshold b = pi/3); the residue modulus is independent of the
    branch of Lam because exp(i eta * 2 pi i) = 1 at i eta = -N. The pure exponential exp(i delta/hbar)
    is finite there (no bound states) -- that candidate is rejected.
 3. bootstrap: S_{B_N,3} = S(1,3) S(2,3) at the pole:
    (i) |S_{B_N,3}| = 1 for real b3, both directions; (ii) semiclassical: arg S_{B_N,3} - (delta13+delta23)/hbar
    -> constant, with delta13 + delta23 real (classical breather-soliton phase, Hirota additivity);
    (iii) Gamma part = G(1+x_c-N/2)G(1+x_c+N/2)/(G(1-x_c+N/2)G(1-x_c-N/2)), x_c = i(eta13+eta23)/2;
    its poles: simple at x_c = -N/2, double at x_c = -N/2-1-m; zeros mirrored (numerical order test).
"""
import mpmath as mp
from smatrix_ansatz import sol, S, delta, eta, I
mp.mp.dps = 30
ok = True
Ibr = lambda a, b: -mp.mpf(4)/9*mp.sinh(3*a)*mp.sin(3*b)

print('1. physical sheet')
for a in (0.3, 0.8, 1.5):
    for b in mp.linspace(mp.pi/3 + 0.01, mp.pi/2 - 0.01, 5):
        s1, s2 = sol(a - I*b), sol(a + I*b)
        ok &= mp.im(s1['P'] - s2['P']) > 0 and abs(mp.im(s1['E'] + s2['E'])) < 1e-25 and abs(mp.im(s1['P'] + s2['P'])) < 1e-25
print('   Im(P1-P2) > 0, total E and P real on pi/3 < b < pi/2: %s' % ok)

print('2. poles at the Bohr-Sommerfeld points')
hbar = mp.mpf('0.1')
for a in (mp.mpf('0.8'), mp.mpf('0.5')):
    Nmax = int(Ibr(a, mp.pi/2)/hbar)
    for N in range(1, min(Nmax, 4) + 1):
        bN = mp.findroot(lambda b: Ibr(a, b) - N*hbar, (mp.pi/3 + 1e-9, mp.pi/2), solver='anderson')
        s1, s2 = sol(a - I*bN), sol(a + I*bN)
        ieta = I*eta(s1, s2, hbar)
        res = [abs(S(sol(a - I*(bN + d)), sol(a + I*(bN + d)), hbar))*abs(d) for d in (mp.mpf('1e-12'), mp.mpf('-1e-14'))]
        pe = [abs(mp.exp(I*delta(sol(a - I*(bN + d)), sol(a + I*(bN + d)))/hbar))*abs(d) for d in (mp.mpf('1e-8'), mp.mpf('1e-10'))]
        pexp = pe[1]/pe[0]   # -> 1/100 if exp(i delta/hbar) is regular at the BS point
        E = s1['E'] + s2['E']; P = s1['P'] + s2['P']
        print('   a=%.1f N=%d: b_N=%.10f, i eta=%s, |S|*|db| = %s, %s ; |e^{i delta/h}||db| ratio(1e-10/1e-8) = %.3f ; E=%.8f P=%.8f'
              % (a, N, bN, mp.nstr(ieta, 12), mp.nstr(res[0], 10), mp.nstr(res[1], 10), pexp, mp.re(E), mp.re(P)))
        ok &= abs(ieta + N) < 1e-20 and abs(res[0]/res[1] - 1) < 1e-6 and res[0] > 1e-8 and abs(pexp - mp.mpf('0.01')) < 1e-4
    # no pole at threshold
    v = [abs(S(sol(a - I*(mp.pi/3 + d)), sol(a + I*(mp.pi/3 + d)), hbar)) for d in (mp.mpf('1e-6'), mp.mpf('1e-9'))]
    print('   a=%.1f: |S| at b = pi/3 + 1e-6, 1e-9: %s, %s (finite: no N=0 pole)' % (a, mp.nstr(v[0], 8), mp.nstr(v[1], 8)))
    ok &= abs(v[1]/v[0] - 1) < 1e-2   # slowly varying (eta log eta), no 1/db growth

print('3. breather-soliton bootstrap')
a, N = mp.mpf('0.8'), 2
bN = mp.findroot(lambda b: Ibr(a, b) - N*hbar, (mp.pi/3 + 1e-9, mp.pi/2), solver='anderson')
c1, c2 = sol(a - I*bN), sol(a + I*bN)
for b3, e3 in [(0.4, 1), (1.4, 1), (0.6, -1)]:
    s3 = sol(b3, e3)
    # breather on the left of soliton 3 for overtaking: right-moving soliton 3 slower/faster both allowed
    SB = S(c1, s3, hbar)*S(c2, s3, hbar)
    dcl = delta(c1, s3) + delta(c2, s3)
    print('   b3=%.1f eps3=%+d: ||S_B3|-1| = %.1e, Im(delta13+delta23) = %.1e' % (b3, e3, abs(abs(SB) - 1), abs(mp.im(dcl))))
    ok &= abs(abs(SB) - 1) < 1e-20 and abs(mp.im(dcl)) < 1e-20
    # semiclassics of the fused amplitude: string at fixed I_br = N hbar_0 while hbar -> 0 means N -> infinity;
    # test at fixed (a, b) with hbar = I_br/N, N = 20, 40, 80
    rs = []
    for NN in (20, 40, 80):
        h = Ibr(a, bN)/NN
        val = mp.im(mp.log(S(c1, s3, h)) + mp.log(S(c2, s3, h))) - mp.re(dcl)/h
        rs.append(val % (2*mp.pi))
    print('      arg S_B3 - delta_cl/hbar (mod 2pi) for hbar = I/20, I/40, I/80: %s' % [mp.nstr(r, 6) for r in rs])
    d1, d2 = abs(rs[1] - rs[0]), abs(rs[2] - rs[1])
    ok &= d2 < 0.6*d1 + 1e-12 and abs(rs[2] - mp.pi) < 0.01   # converges to pi = 2 x (pi/2) Maslov-type constant

def Gpart(x, N):
    g = mp.gamma
    return g(1 + x - mp.mpf(N)/2)*g(1 + x + mp.mpf(N)/2)/(g(1 - x + mp.mpf(N)/2)*g(1 - x - mp.mpf(N)/2))
for N in (1, 2, 3):
    # check identity with S(1,3)S(2,3) Gamma parts
    x = mp.mpf('0.37') + 0.21*I
    e13 = -I*(x - mp.mpf(N)/2); e23 = -I*(x + mp.mpf(N)/2)
    gp = lambda e: mp.gamma(1 + I*e)/mp.gamma(1 - I*e)
    ok &= abs(gp(e13)*gp(e23) - Gpart(x, N)) < 1e-20
    orders = {}
    for x0 in [-mp.mpf(N)/2 + j for j in range(-3, 4)] + [mp.mpf(N)/2 + j for j in range(-3, 4)]:
        f1, f2 = abs(Gpart(x0 + mp.mpf('1e-6'), N)), abs(Gpart(x0 + mp.mpf('1e-8'), N))
        orders[float(x0)] = int(mp.nint(mp.log(f2/f1)/mp.log(100)))
    print('   N=%d: order of pole (+) / zero (-) of the fused Gamma part at x_c: %s' % (N, orders))
    exp_ = {}
    for x0 in orders:
        o = 0
        if x0 == -N/2: o = 1
        elif x0 < -N/2 and float(x0 + N/2).is_integer(): o = 2
        elif x0 == N/2: o = -1
        elif x0 > N/2 and float(x0 - N/2).is_integer(): o = -2
        exp_[x0] = o
    ok &= orders == exp_
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

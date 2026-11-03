"""One-loop quantisation of the bad-Boussinesq breather, CL-reduced theory, vs the S-matrix ansatz.

Breather: b1,2 = a -+ i b, pi/3 < b < pi/2 (labelling of smatrix_poles_bootstrap.py).
 0. Kinematics: envelope velocity V = Re w / Re k = cosh a cos 2b / cos b < -cosh a: every breather
    is supersonic and (for this half of the b-range) LEFT-moving; there is no breather at rest.
    (Checked against the tau function: the envelope maximum moves with V.)
 1. |A(b1, ph) A(b2, ph)| = 1 on the band, so theta_br = theta(b1) + theta(b2) is a real phase shift.
 2. One-loop energy and zero-point momentum of a one-soliton state at classical beta:
        e(beta) = (1/4pi) int w theta' dq,     p(beta) = (1/4pi) int q theta' dq
    (p with the edge terms at q = +-1 averaged out over box lengths L -- a prescription, because
    q theta does not vanish at the band edges). Both continue HOLOMORPHICALLY from real beta to
    b1, b2 (Cauchy-Riemann along beta = a - i t, 0 <= t <= b: no zero/pole of A crosses the band).
 3. Hence the semiclassical breather (Gutzwiller/DHN for the relative periodic orbit, stability
    angles T (w - V q)) has E = E_cl + hbar[e(b1)+e(b2)], P = P_cl + hbar[p(b1)+p(b2)] at classical
    I_br = (N + mu/4) hbar: one-loop breather data = sums of continued one-soliton data.
 4. Consequence for the ansatz: labelling one-soliton states by their classical beta (one-loop
    dispersion E_cl + hbar e, P_cl + hbar p), the poles at I_br = N hbar reproduce these breather
    states. No O(hbar^0) shift of the string condition and no hbar renormalisation are required at
    this order (mu = 0 is taken from Bogdan-Charkina / DHN, not derived).
"""
import mpmath as mp
from smatrix_ansatz import sol, A, I
from oneloop_phase_shifts import phonon, theta_prime
mp.mp.dps = 15
ok = True
w = lambda q: abs(q)*mp.sqrt(1 - q**2)
QS = [-1, -0.99, -0.9, -0.866, -0.8, -0.5, 0, 0.5, 0.9, 0.99, 1]
dlogA = lambda s, q: mp.diff(lambda qq: mp.log(A(s, phonon(qq))), q)


def ep(z):
    s = sol(z)
    e = mp.quad(lambda q: w(q)*(-I)*dlogA(s, q), QS)/(4*mp.pi)
    p = mp.quad(lambda q: q*(-I)*dlogA(s, q), QS)/(4*mp.pi)
    return e, p


# 0. kinematics
for a, b in [(0.8, 1.2), (0.5, 1.1), (1.5, 1.4)]:
    a, b = mp.mpf(a), mp.mpf(b)
    s1, s2 = sol(a - I*b), sol(a + I*b)
    V = mp.re(s1['w'])/mp.re(s1['k'])
    Vf = mp.cosh(a)*mp.cos(2*b)/mp.cos(b)
    tau = lambda x, t: 1 + mp.exp(s1['k']*x - s1['w']*t) + mp.exp(s2['k']*x - s2['w']*t) + A(s1, s2)*mp.exp((s1['k'] + s2['k'])*x - (s1['w'] + s2['w'])*t)
    vv = lambda x, t: mp.re(2*mp.diff(lambda y: mp.log(tau(y, t)), x, 2))
    # envelope: track the maximum of |v| averaged over a period via the centre of mass of v^2
    T = 2*mp.pi/abs(mp.im(s1['w'] - s1['k']*V))
    com = lambda t: mp.quad(lambda x: x*vv(x, t)**2, [-15 + V*t, V*t, 15 + V*t])/mp.quad(lambda x: vv(x, t)**2, [-15 + V*t, V*t, 15 + V*t])
    Vnum = (com(T) - com(0))/T
    print('0. a=%.1f b=%.2f: V = %s (formula %s), centre-of-mass velocity over one period %s'
          % (a, b, mp.nstr(V, 10), mp.nstr(Vf, 10), mp.nstr(Vnum, 8)))
    ok &= abs(V - Vf) < 1e-12 and abs(Vnum - V) < 1e-6*abs(V) and V < -mp.cosh(a)

for a, b in [(mp.mpf('0.8'), mp.mpf('1.2')), (mp.mpf('0.5'), mp.mpf('1.1'))]:
    s1, s2 = sol(a - I*b), sol(a + I*b)
    m = max(abs(abs(A(s1, phonon(q))*A(s2, phonon(q))) - 1) for q in (-0.95, -0.6, -0.1, 0.2, 0.7, 0.97))
    print('1. a=%.1f b=%.1f: max ||A1 A2| - 1| on the band = %.1e' % (a, b, m)); ok &= m < 1e-12
    for t in (mp.mpf('0.3'), mp.mpf('0.8'), b):
        h = mp.mpf('1e-4')
        fa = [(x - y)/(2*h) for x, y in zip(ep(a + h - I*t), ep(a - h - I*t))]
        ft = [(x - y)/(2*h) for x, y in zip(ep(a - I*(t + h)), ep(a - I*(t - h)))]
        cr = max(abs(ft[j] + I*fa[j])/abs(fa[j]) for j in (0, 1))
        print('2.   t=%.2f: Cauchy-Riemann residual (e and p) = %.1e' % (t, cr)); ok &= cr < 1e-5
    er, pr = ep(a); sr = sol(a)
    E1r = mp.quad(lambda q: w(q)*theta_prime(sr, q), QS)/(4*mp.pi)
    ok &= abs(er - E1r) < 1e-10 and abs(mp.im(pr)) < 1e-12
    e1, p1 = ep(a - I*b); e2, p2 = ep(a + I*b)
    eb = mp.quad(lambda q: w(q)*mp.im(dlogA(s1, q) + dlogA(s2, q)), QS)/(4*mp.pi)
    pb = mp.quad(lambda q: q*mp.im(dlogA(s1, q) + dlogA(s2, q)), QS)/(4*mp.pi)
    print('3. a=%.1f b=%.1f: breather (E1, P1)/hbar direct = (%s, %s) ; continued constituent sums = (%s, %s)'
          % (a, b, mp.nstr(eb, 10), mp.nstr(pb, 10), mp.nstr(e1 + e2, 10), mp.nstr(p1 + p2, 10)))
    ok &= abs(eb - (e1 + e2)) < 1e-10 and abs(pb - (p1 + p2)) < 1e-10

# 4. one-loop breather table at the ansatz poles (hbar = 0.1, a = 0.8)
hbar, a = mp.mpf('0.1'), mp.mpf('0.8')
Ibr = lambda b: -mp.mpf(4)/9*mp.sinh(3*a)*mp.sin(3*b)
print('4. hbar=0.1, a=0.8: N, b_N, E_cl, hbar*E1, P_cl, hbar*P1')
for N in (1, 2, 3):
    bN = mp.findroot(lambda b: Ibr(b) - N*hbar, (mp.pi/3 + 1e-9, mp.pi/2), solver='anderson')
    s1, s2 = sol(a - I*bN), sol(a + I*bN)
    e1, p1 = ep(a - I*bN); e2, p2 = ep(a + I*bN)
    print('   %d  %.8f  %.8f  %+.6f  %+.8f  %+.6f' % (N, bN, mp.re(s1['E'] + s2['E']), hbar*mp.re(e1 + e2),
                                                 mp.re(s1['P'] + s2['P']), hbar*mp.re(p1 + p2)))
    ok &= abs(mp.im(e1 + e2)) < 1e-10 and abs(mp.im(p1 + p2)) < 1e-10
print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
assert ok

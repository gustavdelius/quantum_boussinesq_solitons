"""Fluctuations around the bad-Boussinesq soliton on the Charlier-Lenells (stable-phonon) phase space.

 1. Exact linearised modes: tau = 1 + e^{th1} + eps (e^{th2} + A e^{th1+th2}), th2 = K x - Om t with
    K = i q (phonon continued from a soliton, oneloop_step0_wellposedness.py). Then
    dv = d/d eps [2 (log tau)_xx] at eps = 0 solves the linearisation
        dv_tt = dv_xx + dv_xxxx + 6 (v0 dv)_xx     (residual checked to 25 digits)
    and dv -> 2K^2 e^{th2} (x -> -inf), 2K^2 A e^{th2} (x -> +inf): phase shift theta(q) = arg A.
 2. Physical modes: real field, one oscillator per q in (-1, 1), lab frequency w = |q| sqrt(1-q^2) > 0
    requires eps = sign(q) in the continuation (checked).
 3. Mode counting (Levinson-type): total change of theta over the stable band, per branch.
"""
import mpmath as mp
from smatrix_ansatz import sol, A, I
mp.mp.dps = 30
ok = True


def phonon(q):
    q = mp.mpf(q)
    return sol(I*mp.asin(q), 1 if q > 0 else -1)


def theta_prime(s, q):
    return mp.im(mp.diff(lambda qq: mp.log(A(s, phonon(qq))), q))


if __name__ == '__main__':
    for bs in ('0.4', '1.1'):
        s = sol(mp.mpf(bs))
        for q in ('0.35', '-0.6', '0.9'):
            ph = phonon(q)
            Ak = A(s, ph)
            th1 = lambda x, t: s['k']*x - s['w']*t
            th2 = lambda x, t: ph['k']*x - ph['w']*t
            v0 = lambda x, t: 2*mp.diff(lambda y: mp.log(1 + mp.exp(th1(y, t))), x, 2)
            f = lambda x, t: (mp.exp(th2(x, t)) + Ak*mp.exp(th1(x, t) + th2(x, t)))/(1 + mp.exp(th1(x, t)))
            dv = lambda x, t: 2*mp.diff(lambda y: f(y, t), x, 2)
            res = 0
            for x0, t0 in [(mp.mpf('0.3'), mp.mpf('0.1')), (mp.mpf('-1.2'), mp.mpf('0.5'))]:
                lhs = mp.diff(lambda t: dv(x0, t), t0, 2)
                rhs = (mp.diff(lambda y: dv(y, t0), x0, 2) + mp.diff(lambda y: dv(y, t0), x0, 4)
                       + 6*mp.diff(lambda y: v0(y, t0)*dv(y, t0), x0, 2))
                res = max(res, abs(lhs - rhs)/abs(lhs))
            X = 40/s['k']
            ratio = (dv(X, 0)/mp.exp(th2(X, 0)))/(dv(-X, 0)/mp.exp(th2(-X, 0)))
            w = -I*ph['w']   # tau ~ e^{i q x - i w t}
            print('b_s=%s q=%s: linearised-eq residual %.1e ; dv(+inf)/dv(-inf) / A - 1 = %.1e ; |A|=%s ; w=%s (>0: %s)'
                  % (bs, q, res, abs(ratio/Ak - 1), mp.nstr(abs(Ak), 12), mp.nstr(w, 8), mp.re(w) > 0))
            ok &= res < 1e-15 and abs(ratio/Ak - 1) < 1e-12 and abs(abs(Ak) - 1) < 1e-25
            ok &= abs(mp.im(w)) < 1e-25 and mp.re(w) > 0 and abs(w**2 - (mp.mpf(q)**2 - mp.mpf(q)**4)) < 1e-25
    print('3. Levinson-type count: (1/2pi) * total change of theta over the band')
    for bs in ('0.2', '0.7', '1.5'):
        s = sol(mp.mpf(bs))
        dp = mp.quad(lambda q: theta_prime(s, q), [0, 0.5, 0.9, 0.99, 1])
        dm = mp.quad(lambda q: theta_prime(s, q), [-1, -0.99, -0.9, -0.5, 0])
        print('   b_s=%s: right-moving band %.10f, left-moving band %.10f, total %.10f (x 2 pi)'
              % (bs, dp/(2*mp.pi), dm/(2*mp.pi), (dp + dm)/(2*mp.pi)))
    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

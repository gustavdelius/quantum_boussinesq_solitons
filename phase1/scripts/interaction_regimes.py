"""Two-soliton interaction regimes, resonances, breathers and position shifts (good Boussinesq).

Parametrise k = sin(alpha), sqrt(1-k^2) = cos(alpha), alpha in (0, pi/2); then w = eps sin(a)cos(a),
speed v = cos(alpha).  Claims checked here (derived in this project, label (c)):
 (i)  overtaking (eps1 = eps2):  A = [1 - 3 cot^2(S/2)] / [1 - 3 cot^2(D/2)],  S = a1+a2, D = a1-a2
 (ii) head-on   (eps1 = -eps2):  A = [1 - 3 tan^2(D/2)] / [1 - 3 tan^2(S/2)]
 (iii) consequences: regular iff A > 0; resonance curves; sign of shifts
 (iv) breathers (k2 = conj k1, same branch): tau = 1 + 2 e^R cos I + A e^{2R}, regular iff A > 1;
      test numerically over the complex k plane for good and bad equations
 (v)  at a resonance A = 0 the difference (k1-k2, w1-w2) lies on the soliton dispersion curve
 (vi) asymptotic position shifts, compared with peak tracking of the exact solution
"""
import numpy as np
import sympy as sp

a1, a2 = sp.symbols('alpha_1 alpha_2', positive=True)


def A_sym(k1, w1, k2, w2):
    P = lambda K, W: W**2 - K**2 + K**4
    return -P(k1 - k2, w1 - w2)/P(k1 + k2, w1 + w2)


S, D = a1 + a2, a1 - a2
for eps, closed, name in [(1, (1 - 3*sp.cot(S/2)**2)/(1 - 3*sp.cot(D/2)**2), 'overtaking'),
                          (-1, (1 - 3*sp.tan(D/2)**2)/(1 - 3*sp.tan(S/2)**2), 'head-on')]:
    Aex = A_sym(sp.sin(a1), sp.sin(a1)*sp.cos(a1), sp.sin(a2), eps*sp.sin(a2)*sp.cos(a2))
    pts = [(0.2, 0.7), (1.1, 0.3), (1.4, 1.2), (0.05, 0.6)]
    err = max(abs(float((Aex - closed).subs({a1: p, a2: q}))) for p, q in pts)
    simp = sp.simplify(sp.expand_trig(Aex - closed))
    print('(%s) closed form: max numerical error %.1e; simplify ->' % (name, err), simp)
    assert err < 1e-12

# ---- (iii) classification on a grid ------------------------------------------------------------
def A_num(al1, al2, e1, e2):
    k1, k2 = np.sin(al1), np.sin(al2)
    w1, w2 = e1*k1*np.cos(al1), e2*k2*np.cos(al2)
    P = lambda K, W: W**2 - K**2 + K**4
    return -P(k1 - k2, w1 - w2)/P(k1 + k2, w1 + w2)

g = np.linspace(1e-3, np.pi/2 - 1e-3, 400)
G1, G2 = np.meshgrid(g, g)
Aov, Aho = A_num(G1, G2, 1, 1), A_num(G1, G2, 1, -1)
off = np.abs(G1 - G2) > 1e-2
pred_ov = (G1 + G2) < 2*np.pi/3
pred_ho_reg = ((G1 + G2) < np.pi/3) | (np.abs(G1 - G2) > np.pi/3)
band = lambda Z, c: np.abs(Z - c) > 2e-2
m_ov = off & band(G1 + G2, 2*np.pi/3)
m_ho = off & band(G1 + G2, np.pi/3) & band(np.abs(G1 - G2), np.pi/3)
assert np.all((Aov > 0)[m_ov] == pred_ov[m_ov]); assert np.all((Aov < 1)[m_ov & pred_ov])
assert np.all((Aho > 0)[m_ho] == pred_ho_reg[m_ho])
assert np.all((Aho > 1)[m_ho & ((G1 + G2) < np.pi/3)])
assert np.all((Aho < 1)[m_ho & (np.abs(G1 - G2) > np.pi/3)])
print('(iii) overtaking: regular & 0<A<1 iff a1+a2 < 2pi/3, singular beyond (resonance A=0 on the line)')
print('      head-on: A>1 (attractive) iff a1+a2 < pi/3; singular iff a1+a2 > pi/3 and |a1-a2| < pi/3;')
print('               0<A<1 (repulsive) iff |a1-a2| > pi/3; A=inf on a1+a2=pi/3, A=0 on |a1-a2|=pi/3')

# ---- (v) resonance = third soliton on the dispersion curve --------------------------------------
for al1, e in [(1.3, 1), (0.9, -1)]:
    if e == 1:
        al2 = 2*np.pi/3 - al1                  # overtaking resonance, A = 0
    else:
        al2 = al1 - np.pi/3                    # head-on resonance |a1-a2| = pi/3, A = 0
    k1, k2 = np.sin(al1), np.sin(al2)
    w1, w2 = k1*np.cos(al1), e*k2*np.cos(al2)
    K, W = k1 - k2, w1 - w2
    print('(v) eps2=%+d: A=%.2e, (K,W)=(%.4f,%.4f), W^2-K^2+K^4 = %.2e, K<1: %s'
          % (e, A_num(al1, al2, 1, e), K, W, W**2 - K**2 + K**4, abs(K) < 1))
    assert abs(W**2 - K**2 + K**4) < 1e-12

# ---- (iv) breathers ----------------------------------------------------------------------------
def breather_A(k, sign_disp):
    # w = k sqrt(1 + sign_disp k^2): sign_disp = -1 good, +1 bad; principal branch
    w = k*np.sqrt(1 + sign_disp*k**2 + 0j)
    P = lambda K, W: W**2 - K**2 - sign_disp*K**4   # good: W^2-K^2+K^4 ; bad: W^2-K^2-K^4
    return -P(k - np.conj(k), w - np.conj(w))/P(k + np.conj(k), w + np.conj(w))

re = np.linspace(0.01, 2.0, 500); im = np.linspace(0.01, 2.0, 500)
R, I = np.meshgrid(re, im); Kc = R + 1j*I
for sd, name in [(-1, 'good'), (1, 'bad')]:
    Ab = breather_A(Kc, sd)
    assert np.max(np.abs(Ab.imag)) < 1e-9
    frac = np.mean(Ab.real > 1)
    print('(iv) %s Boussinesq breathers: fraction of sampled complex k with A>1 (regular): %.3f;'
          ' max A = %.4f' % (name, frac, np.nanmax(Ab.real)))

# analytic statement for the good case: with alpha = a + i b, A = [1-3cot^2 a]/[1+3coth^2 b] < 1
a_, b_ = 0.7, 0.4
al = a_ + 1j*b_
k = np.sin(al); w = np.sin(al)*np.cos(al)
P = lambda K, W: W**2 - K**2 + K**4
Ab = -P(k - np.conj(k), w - np.conj(w))/P(k + np.conj(k), w + np.conj(w))
print('    good, alpha=0.7+0.4i: A =', Ab, ' closed form:', (1 - 3/np.tan(a_)**2)/(1 + 3/np.tanh(b_)**2))

# grid test of the good-breather closed form and of the bound A < 1/4
aa, bb = np.meshgrid(np.linspace(0.02, np.pi - 0.02, 200), np.linspace(-2, 2, 201))
bb = np.where(np.abs(bb) < 1e-3, 1e-3, bb)
al = aa + 1j*bb
k = np.sin(al); w = np.sin(al)*np.cos(al)
Ag = (-P(k - np.conj(k), w - np.conj(w))/P(k + np.conj(k), w + np.conj(w))).real
Acl = (1 - 3/np.tan(aa)**2)/(1 + 3/np.tanh(bb)**2)
print('    good breather closed form on grid: max error %.1e, max A = %.4f (< 1/4)'
      % (np.max(np.abs(Ag - Acl)/np.maximum(1, np.abs(Acl))), Ag.max()))
assert np.max(np.abs(Ag - Acl)/np.maximum(1, np.abs(Acl))) < 1e-8 and Ag.max() < 0.25

# ---- (vi) position shifts vs peak tracking -------------------------------------------------------
def u_two(xg, t, k1, w1, k2, w2, A):
    th1, th2 = k1*xg - w1*t, k2*xg - w2*t
    # u = 2 (log tau)_xx computed stably
    m = np.maximum.reduce([np.zeros_like(xg), th1, th2, th1 + th2 + np.log(A)])
    e0, e1, e2, e12 = np.exp(-m), np.exp(th1 - m), np.exp(th2 - m), A*np.exp(th1 + th2 - m)
    tau = e0 + e1 + e2 + e12
    tx = k1*e1 + k2*e2 + (k1 + k2)*e12
    txx = k1**2*e1 + k2**2*e2 + (k1 + k2)**2*e12
    return 2*(txx/tau - (tx/tau)**2)

def track(k1, w1, k2, w2, A, which, T=400.0):
    xg = np.linspace(-1500, 1500, 600001)
    out = []
    for t in (-T, T):
        u = u_two(xg, t, k1, w1, k2, w2, A)
        v = w1/k1 if which == 1 else w2/k2
        sel = np.abs(xg - v*t) < 60
        out.append(xg[sel][np.argmax(u[sel])] - v*t)
    return out[1] - out[0]

for (al1, al2, e2) in [(0.5, 0.9, 1), (0.3, 0.6, -1), (0.2, 1.3, -1)]:   # all regular (A>0)
    k1, k2 = np.sin(al1), np.sin(al2)
    w1, w2 = k1*np.cos(al1), e2*k2*np.cos(al2)
    A = A_num(al1, al2, 1, e2)
    v1, v2 = w1/k1, w2/k2
    # theory: the soliton that ends up on the side where the other's phase is +infinity shifts by -log A / k
    # soliton 1 at late times: theta_2 -> +inf at x ~ v1 t iff k2 (v1 - v2) > 0
    d1 = -np.log(A)/k1*(1 if k2*(v1 - v2) > 0 else -1)
    d2 = -np.log(A)/k2*(1 if k1*(v2 - v1) > 0 else -1)
    n1, n2 = track(k1, w1, k2, w2, A, 1), track(k1, w1, k2, w2, A, 2)
    print('(vi) a=(%.1f,%.1f) eps2=%+d A=%.4f: shift1 theory %.5f numeric %.5f | shift2 theory %.5f numeric %.5f'
          % (al1, al2, e2, A, d1, n1, d2, n2))
    assert abs(d1 - n1) < 2e-2 and abs(d2 - n2) < 2e-2
# ---- (vii) bad equation: k = sinh(b), speed cosh(b); closed forms by continuation alpha -> i b ----
def A_bad(b1, b2, e1, e2):
    k1, k2 = np.sinh(b1), np.sinh(b2)
    w1, w2 = e1*k1*np.cosh(b1), e2*k2*np.cosh(b2)
    P = lambda K, W: W**2 - K**2 - K**4
    return -P(k1 - k2, w1 - w2)/P(k1 + k2, w1 + w2)
B1, B2 = np.meshgrid(np.linspace(0.01, 4, 300), np.linspace(0.013, 4.1, 300))
Sb, Db = B1 + B2, B1 - B2
ov = (1 + 3/np.tanh(Sb/2)**2)/(1 + 3/np.tanh(Db/2)**2)
ho = (1 + 3*np.tanh(Db/2)**2)/(1 + 3*np.tanh(Sb/2)**2)
e_ov = np.max(np.abs(A_bad(B1, B2, 1, 1) - ov)); e_ho = np.max(np.abs(A_bad(B1, B2, 1, -1) - ho))
print('(vii) bad: overtaking A = [1+3coth^2(S/2)]/[1+3coth^2(D/2)] (err %.1e), head-on A = '
      '[1+3tanh^2(D/2)]/[1+3tanh^2(S/2)] (err %.1e); both in (0,1): %s'
      % (e_ov, e_ho, bool(np.all((ov > 0) & (ov < 1) & (ho > 0) & (ho < 1)))))
assert e_ov < 1e-9 and e_ho < 1e-9
print('ALL CHECKS PASSED')

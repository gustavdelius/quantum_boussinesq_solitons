"""Two solitons + Corrigan-Zambon defect (good equation): symplecticity of scattering with topological
cross terms, and semiclassical soliton-soliton-defect factorisation.  See phase2/defect_semiclassics.pdf.

Exact solution: CZ (5.6)/(5.11): u = -6 (log tau_u)_x on x<0, v = -6 (log tau_v)_x on x>0 with
  tau_u = 1 + E1 + E2 + S12 E1 E2,  tau_v = 1 + z1 E1 + z2 E2 + S12 z1 z2 E1 E2,  our U = phi_x = -u_x/3.
Checks:
 1. S12 (CZ 5.7) = A12 of the project (good equation) = KP cross-ratio of the roots
 2. asymptotic shifts read off from peak positions of the exact solution, for overtaking and head-on
    collisions, with the soliton-soliton collision on the LEFT or on the RIGHT of the defect:
      Dx_j = (soliton-soliton shift) + (defect shift -log(z_j^eps_j)/k_j), independent of the order
 3. symplecticity: sum_j dP_j ^ dDx_j = cross_in - cross_out (Phase 2 cross terms; the defect adds none),
    40 digits; the naive test (no cross terms) fails
 4. factorisation: the Darboux-frame shifts are -grad_P of delta_12 + delta_1D + delta_2D, for both orders
 5. phases stay finite on resonance (A -> 0, inf) and absorption (z -> 0, inf) lines; behaviour at the cusp
"""
import mpmath as mp
mp.mp.dps = 40
r3 = mp.sqrt(3)
ok = True


def sol(al, eps):
    k = mp.sin(al); v = eps*mp.cos(al)
    return dict(al=al, eps=eps, k=k, v=v, w=k*v, M=2*k, Pi=-2*k*v, P=mp.mpf(2)/3*eps*k**3*mp.cos(al),
                ell=-eps*mp.cos(3*al), pa=(k + eps*mp.cos(al)/r3)/2, pb=(eps*mp.cos(al)/r3 - k)/2)


def A12(s1, s2):
    Pp = lambda K, W: W**2 - K**2 + K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


def zt(s1, s, eta):        # transmission factor in the direction of motion, z^eps
    q = -s/(2*r3)
    return ((s1['pb'] - q)/(s1['pa'] - q))**(eta*s1['eps'])


def delta12(s1, s2):       # Phase 2 formula, good equation
    return mp.mpf(2)/9*(s1['ell'] - s2['ell'])*mp.log(A12(s1, s2)) + mp.mpf(2)/3*s1['M']*s2['M']*(s1['v'] - s2['v'])


def deltaD(s1, s, eta):
    q = -s/(2*r3); e = s1['eps']*eta
    return (mp.mpf(2)/9*(s1['ell'] - (3*s - s**3)/2)*mp.log(zt(s1, s, eta))
            - mp.mpf(2)/3*(s1['Pi']*e*q - e*(s**2 - 2)/(2*r3)*s1['M']))


# ---------------------------------------------------------------- 1. S12 = A12 = cross ratio
for (a1, e1), (a2, e2) in [((0.4, 1), (0.9, 1)), ((0.5, 1), (0.3, -1)), ((1.2, 1), (1.0, -1))]:
    s1, s2 = sol(mp.mpf(a1), e1), sol(mp.mpf(a2), e2)
    th1, th2 = mp.atanh(s1['v']), mp.atanh(s2['v'])
    S = (mp.sinh(th1 - th2)**2 - 3*(mp.cosh(th1) - mp.cosh(th2))**2)/(mp.sinh(th1 - th2)**2 - 3*(mp.cosh(th1) + mp.cosh(th2))**2)
    CR = (s1['pa'] - s2['pa'])*(s1['pb'] - s2['pb'])/((s1['pa'] - s2['pb'])*(s1['pb'] - s2['pa']))
    d = max(abs(S - A12(s1, s2)), abs(CR - A12(s1, s2)))
    print('1. S12 - A12, cross ratio - A12: %s' % mp.nstr(d, 3)); ok &= d < 1e-35


# ---------------------------------------------------------------- 2. shifts from the exact solution
def U_of(s1, s2, s, eta, side, t, x2off):
    z1, z2 = (1, 1) if side == 'u' else (zt(s1, s, eta)**s1['eps'], zt(s2, s, eta)**s2['eps'])   # CZ z
    S = A12(s1, s2)
    def U(x):
        f = lambda xx: mp.log(1 + z1*mp.exp(s1['k']*(xx - s1['v']*t)) + z2*mp.exp(s2['k']*(xx - s2['v']*t) + x2off)
                              + S*z1*z2*mp.exp(s1['k']*(xx - s1['v']*t) + s2['k']*(xx - s2['v']*t) + x2off))
        return 2*mp.diff(f, x, 2)            # U = 2 (log tau)_xx = -u_x/3
    return U


def peak(U, x0):
    return mp.findroot(lambda x: mp.diff(U, x), x0, tol=mp.mpf(10)**-45, verify=False)


def side_of(x):
    return 'u' if x < 0 else 'v'


def shifts_by_peaks(s1, s2, s, eta, x2off, T):
    """peak positions at t = -T and t = +T, minus free motion"""
    res = []
    for j, sj, so in ((1, s1, s2), (2, s2, s1)):
        c = 0 if j == 1 else -x2off/s2['k']
        Xs = []
        for t in (-T, T):
            x0 = sj['v']*t + c
            # guesses: try both sides' z-shift and partner shift +-log A/k; keep the true peak of U
            cands = []
            for dz in (0, -mp.log(zt(sj, s, eta)**sj['eps'])/sj['k'], mp.log(zt(sj, s, eta)**sj['eps'])/sj['k']):
                for dA in (0, mp.log(A12(s1, s2))/sj['k'], -mp.log(A12(s1, s2))/sj['k']):
                    g = x0 + dz + dA
                    if abs(g) < 5:
                        continue
                    U = U_of(s1, s2, s, eta, side_of(g), t, x2off)
                    try:
                        xp = peak(U, g + mp.mpf('0.2'))
                    except Exception:
                        continue
                    if abs(xp - g) < 1e-20 and side_of(xp) == side_of(g) and abs(U(xp) - sj['k']**2/2) < 1e-20:
                        cands.append(xp)
            cands = sorted(set([mp.nstr(c_, 30) for c_ in cands]))
            assert len(cands) == 1, cands
            Xs.append(mp.mpf(cands[0]) - (sj['v']*t + c))
        res.append(Xs[1] - Xs[0])
    return res


def predicted(s1, s2, s, eta):
    L = mp.log(A12(s1, s2))
    return [-L/s1['k'] - mp.log(zt(s1, s, eta))/s1['k'], L/s2['k'] - mp.log(zt(s2, s, eta))/s2['k']]


T = mp.mpf(1000)
worst = 0
configs = [('overtaking', (mp.mpf('0.45'), 1), (mp.mpf('1.0'), 1), mp.mpf('2.4'), 1),
           ('overtaking', (mp.mpf('0.6'), 1), (mp.mpf('1.3'), 1), mp.mpf('-2.3'), -1),
           ('head-on', (mp.mpf('0.4'), 1), (mp.mpf('0.6'), -1), mp.mpf('2.2'), 1),
           ('head-on', (mp.mpf('0.3'), 1), (mp.mpf('0.7'), -1), mp.mpf('-2.6'), -1)]
for name, (a1, e1), (a2, e2), s, eta in configs:
    s1, s2 = sol(a1, e1), sol(a2, e2)
    assert A12(s1, s2) > 0 and zt(s1, s, eta) > 0 and zt(s2, s, eta) > 0
    pr = predicted(s1, s2, s, eta)
    for x2off in (mp.mpf(-30)*s2['k'], mp.mpf(30)*s2['k']):   # collision left / right of the defect
        # collision time/place of the free trajectories
        c2 = -x2off/s2['k']; tc = c2/(s1['v'] - s2['v']); xc = s1['v']*tc
        ex = shifts_by_peaks(s1, s2, s, eta, x2off, T)
        d = max(abs(ex[0] - pr[0]), abs(ex[1] - pr[1])); worst = max(worst, d)
        print('2. %-10s s=%5s eta=%+d collision at x=%7.2f (%s of defect): Dx = (%s, %s), |extracted - predicted| = %s'
              % (name, mp.nstr(s, 3), eta, float(xc), 'left ' if xc < 0 else 'right', mp.nstr(ex[0], 12), mp.nstr(ex[1], 12), mp.nstr(d, 3)))
ok &= worst < 1e-25

# ---------------------------------------------------------------- 3. symplecticity
print('3. coefficient of dalpha1^dalpha2:  sum dP^dDx  vs  cross_in - cross_out  (naive: should be 0)')
worst = 0
extra = [('overtaking', (mp.mpf('0.3'), 1), (mp.mpf('0.8'), 1), mp.mpf(s_), eta_) for s_ in ('0.2', '-1.1', '1.6') for eta_ in (1, -1)]
extra = [c for c in extra if zt(sol(*c[1]), c[3], c[4]) > 0 and zt(sol(*c[2]), c[3], c[4]) > 0]
for name, (a1, e1), (a2, e2), s, eta in configs + extra:
    def Dx(x1, x2, j):
        return predicted(sol(x1, e1), sol(x2, e2), s, eta)[j]
    P = lambda x, e: sol(x, e)['P']; Pi = lambda x, e: sol(x, e)['Pi']; M = lambda x, e: sol(x, e)['M']
    dP1 = mp.diff(lambda x: P(x, e1), a1); dP2 = mp.diff(lambda x: P(x, e2), a2)
    lhs = dP1*mp.diff(lambda x: Dx(a1, x, 0), a2) - dP2*mp.diff(lambda x: Dx(x, a2, 1), a1)
    rhs = (-mp.diff(lambda x: Pi(x, e2), a2)*mp.diff(lambda x: M(x, e1), a1)
           - mp.diff(lambda x: Pi(x, e1), a1)*mp.diff(lambda x: M(x, e2), a2))
    worst = max(worst, abs(lhs - rhs))
    print('   %-10s alpha=(%s,%s) s=%5s: %+.15f  %+.15f  diff %s' % (name, mp.nstr(a1, 3), mp.nstr(a2, 3), mp.nstr(s, 3),
                                                                  float(lhs), float(rhs), mp.nstr(abs(lhs - rhs), 3)))
    ok &= abs(lhs) > 1e-3          # cross terms are needed
ok &= worst < 1e-30

# ---------------------------------------------------------------- 4. factorisation
worst = 0
for name, (a1, e1), (a2, e2), s, eta in configs:
    dtot = lambda x1, x2: (delta12(sol(x1, e1), sol(x2, e2)) + deltaD(sol(x1, e1), s, eta) + deltaD(sol(x2, e2), s, eta))
    s1, s2 = sol(a1, e1), sol(a2, e2)
    g1 = mp.diff(lambda x: sol(x, e1)['Pi'], a1)/mp.diff(lambda x: sol(x, e1)['P'], a1)
    g2 = mp.diff(lambda x: sol(x, e2)['Pi'], a2)/mp.diff(lambda x: sol(x, e2)['P'], a2)
    pr = predicted(s1, s2, s, eta)
    D1 = pr[0] + g1*s2['M']; D2 = pr[1] - g2*s1['M']           # Darboux frame (phi normalised at -inf)
    r1 = mp.diff(lambda x: dtot(x, a2), a1) + mp.diff(lambda x: sol(x, e1)['P'], a1)*D1
    r2 = mp.diff(lambda x: dtot(a1, x), a2) + mp.diff(lambda x: sol(x, e2)['P'], a2)*D2
    worst = max(worst, abs(r1), abs(r2))
print('4. |grad(delta_12 + delta_1D + delta_2D) + dP Dx\'| max:', mp.nstr(worst, 3)); ok &= worst < 1e-30

# ---------------------------------------------------------------- 5. resonance, absorption, cusp
# resonance lines: overtaking A12 -> 0 at p_1a = p_2a, i.e. alpha1 = 2 pi/3 - alpha2 (mirror image across
# the cusp, same P); head-on A12 -> inf at p_1b = p_2a.  On both, ell_1 = ell_2 and delta_12 stays finite.
for name, e2, a2, f, a0 in [('overtaking A->0  ', 1, mp.mpf('0.7'), lambda s1, s2: s1['pa'] - s2['pa'], mp.mpf('1.3')),
                            ('head-on    A->inf', -1, mp.mpf('0.9'), lambda s1, s2: s1['pb'] - s2['pa'], mp.mpf('0.6'))]:
    s2 = sol(a2, e2)
    ares = mp.findroot(lambda x: f(sol(x, 1), s2), a0)
    s1r = sol(ares, 1)
    dlim = mp.mpf(2)/3*s1r['M']*s2['M']*(s1r['v'] - s2['v'])
    for h in (mp.mpf(10)**-4, mp.mpf(10)**-8, mp.mpf(10)**-12):
        s1 = sol(ares + h, 1)
        print('5. %s alpha1*=%s: alpha1-alpha1*=%s  log|A|=%s  delta_12=%s' % (name, mp.nstr(ares, 8), mp.nstr(h, 2),
              mp.nstr(mp.log(abs(A12(s1, s2))), 6), mp.nstr(mp.re(delta12(s1, s2)), 12)))
    print('   ell1 - ell2 = %s ; limit (2/3) M1 M2 (v1 - v2) = %s' % (mp.nstr(s1r['ell'] - s2['ell'], 3), mp.nstr(dlim, 12)))
    ok &= abs(mp.re(delta12(sol(ares + mp.mpf(10)**-12, 1), s2)) - dlim) < 1e-9
# absorption line z -> 0: q = p_b
s, eta = mp.mpf('0.7'), 1
aabs = mp.findroot(lambda x: sol(x, 1)['pb'] + s/(2*r3), mp.mpf('0.4'))
for h in (mp.mpf(10)**-4, mp.mpf(10)**-8, mp.mpf(10)**-12):
    s1 = sol(aabs - h, 1)
    print('   absorption: alpha - alpha* = %s  log z = %s  Dx = %s  delta_D = %s' % (mp.nstr(-h, 2), mp.nstr(mp.log(zt(s1, s, eta)), 6),
          mp.nstr(-mp.log(zt(s1, s, eta))/s1['k'], 6), mp.nstr(deltaD(s1, s, eta), 12)))
ok &= abs(deltaD(sol(aabs - mp.mpf(10)**-12, 1), s, eta) - deltaD(sol(aabs - mp.mpf(10)**-11, 1), s, eta)) < 1e-9
# cusp alpha = pi/3: dP/dalpha = 0, d delta_D/dalpha = 0, Dx finite
for s in (mp.mpf('2.4'), mp.mpf('-2.4')):
    c = sol(mp.pi/3, 1)
    print('   cusp s=%s: dP/dalpha = %s, d delta_D/dalpha = %s, Dx = %s, z = %s' % (mp.nstr(s, 2),
          mp.nstr(mp.diff(lambda x: sol(x, 1)['P'], mp.pi/3), 3), mp.nstr(mp.diff(lambda x: deltaD(sol(x, 1), s, 1), mp.pi/3), 3),
          mp.nstr(-mp.log(zt(c, s, 1))/c['k'], 8), mp.nstr(zt(c, s, 1), 8)))
assert ok
print('ALL CHECKS PASSED')

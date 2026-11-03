"""The bad-equation candidate S-matrix (phase2/smatrix_ansatz.pdf) continued to the GOOD equation, and
tested against the Corrigan-Zambon defect transmission phases (phase2/defect_semiclassics.pdf, Sec. 7).

Good data: k = sin a, v = eps cos a, ell = -eps cos 3a, M = 2k, Pi = -M v, P = (2/3) eps k^3 cos a,
A12 = KP cross-ratio, delta = (2/9)(ell1 - ell2) log A12 + T12, T12 = (2/3) M1 M2 (v1 - v2).
Continued candidate (d), eta = (2/9)(ell1 - ell2)/hbar, R(x) = Gamma(1+ix)/Gamma(1-ix):
  eps1 = eps2:            S = R(eta) (e^2 A/eta^2)^{i eta} e^{i T/hbar}
  eps1 = +1, eps2 = -1:   S = R(eta)^{-1} (A eta^2/e^2)^{i eta} e^{i T/hbar};   (-,+) by unitarity.
Checks:
 1. regular region: |S| = 1, S(1,2) S(2,1) = 1, arg S - delta/hbar -> (pi/2) tau sgn(eta), tau = +-1
 2. resonance lines (ell1 = ell2 there): S -> e^{iT/hbar} continuously; beyond them (A < 0) |S| != 1
 3. 'breather' kinematics a -+ i b: Gamma(1+i eta) poles exist (i eta = -N), but Phase 1 says every good
    breather is singular; we record A < 1 there
 4. defect fusion: the defect with root q absorbs the soliton S = {q, q'} and becomes the defect with root q'
    (defect_transmission_phase.py, check 5).  Leading order, exactly:
       delta_D(q')(3) - delta_D(q)(3) = rho [delta(3, S) + M_S Pi_3],  rho = +-1, no constant.
    Around the triangle q -> q' -> q'' -> q (the three solitons of one KP triple, common ell_S) the
    defect phases telescope, so any exact S must satisfy  prod_i [S(3,S_i) e^{i M_i Pi_3/hbar}]^{rho_i} = 1.
    The continued candidate gives [R(eta)(e/eta)^{2 i eta}]^{sum rho_i tau_i} with sum rho_i tau_i = +-1:
    closure fails by e^{+- i pi/2}(1 + O(hbar)).  The pure exponential e^{i delta/hbar} closes exactly.
"""
import mpmath as mp
mp.mp.dps = 40
I = mp.mpc(0, 1); r3 = mp.sqrt(3)
ok = True


def sol(a, eps):
    a = mp.mpmathify(a); k = mp.sin(a); v = eps*mp.cos(a)
    return dict(a=a, eps=eps, k=k, v=v, w=k*v, M=2*k, Pi=-2*k*v, P=mp.mpf(2)/3*eps*k**3*mp.cos(a),
                ell=-eps*mp.cos(3*a), pa=(k + eps*mp.cos(a)/r3)/2, pb=(eps*mp.cos(a)/r3 - k)/2)


def A12(s1, s2):
    Pp = lambda K, W: W**2 - K**2 + K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w'])/Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


def T12(s1, s2):
    return mp.mpf(2)/3*s1['M']*s2['M']*(s1['v'] - s2['v'])


def delta(s1, s2):
    return mp.mpf(2)/9*(s1['ell'] - s2['ell'])*mp.log(A12(s1, s2)) + T12(s1, s2)


def logR(x):
    return mp.loggamma(1 + I*x) - mp.loggamma(1 - I*x)


def logS(s1, s2, hbar, gamma=True):
    """continued candidate (gamma=True) or pure exponential e^{i delta/hbar} (gamma=False)"""
    if not gamma:
        return I*delta(s1, s2)/hbar
    if s1['eps'] == -1 and s2['eps'] == 1:
        return -logS(s2, s1, hbar)
    e = mp.mpf(2)/9*(s1['ell'] - s2['ell'])/hbar
    calA = 81*A12(s1, s2)/(4*(s1['ell'] - s2['ell'])**2)
    if s1['eps'] == s2['eps']:
        out = logR(e) + I*e*(2 + 2*mp.log(hbar) + mp.log(calA))          # (e^2 A/eta^2)^{i eta}
    else:
        out = -logR(e) - I*e*(2 + 2*mp.log(hbar) - mp.log(A12(s1, s2)*(s1['ell'] - s2['ell'])**2*4/81))
    return out + I*T12(s1, s2)/hbar


def wrap(x):
    return x - 2*mp.pi*mp.nint(x/(2*mp.pi))


# ---------------------------------------------------------------- 1. regular region
print('1. regular region (alpha < pi/3, A > 0)')
for (a1, e1), (a2, e2) in [((0.3, 1), (0.8, 1)), ((0.5, 1), (0.2, 1)), ((0.4, 1), (0.6, -1)), ((0.2, 1), (0.6, -1))]:
    s1, s2 = sol(a1, e1), sol(a2, e2)
    assert A12(s1, s2) > 0
    tau = 1 if e1 == e2 else -1
    sg = mp.sign(s1['ell'] - s2['ell'])
    rem = []
    for hb in (mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')):
        L = logS(s1, s2, hb)
        rem.append(wrap(mp.im(L) - delta(s1, s2)/hb - mp.pi/2*tau*sg))
        u = abs(mp.re(L)) + abs(L + logS(s2, s1, hb))
        ok &= u < 1e-25
    print('   alpha=(%s,%s) eps=(%+d,%+d): arg S - delta/hbar - (pi/2) tau sgn eta at hbar=1e-2,1e-3,1e-4: %s'
          % (a1, a2, e1, e2, [mp.nstr(r, 3) for r in rem]))
    ok &= abs(rem[2]) < 2e-3 and abs(rem[2]) < abs(rem[0])
print('   |S| = 1 and S(1,2)S(2,1) = 1 to 1e-25:', ok)

# ---------------------------------------------------------------- 2. resonance lines
print('2. resonance lines: overtaking A -> 0 at alpha1 = 2pi/3 - alpha2 (alpha1 > pi/3); hbar = 0.05')
hb = mp.mpf('0.05')
s2 = sol('0.7', 1); ares = 2*mp.pi/3 - s2['a']
for h in (mp.mpf('1e-3'), mp.mpf('1e-6'), mp.mpf('1e-9'), -mp.mpf('1e-3')):
    s1 = sol(ares - h, 1)
    L = logS(s1, s2, hb)
    print('   alpha1 - alpha1* = %+.0e: A = %s, eta = %s, |S| = %s, |log S - iT/hbar| = %s'
          % (float(-h), mp.nstr(A12(s1, s2), 4), mp.nstr(mp.mpf(2)/9*(s1['ell'] - s2['ell'])/hb, 4),
             mp.nstr(abs(mp.exp(L)), 8), mp.nstr(abs(L - I*T12(s1, s2)/hb), 3)))
s1 = sol(ares - mp.mpf('1e-9'), 1)
ok &= abs(logS(s1, s2, hb) - I*T12(s1, s2)/hb) < 1e-6
s1 = sol(ares + mp.mpf('1e-3'), 1)
ok &= abs(abs(mp.exp(logS(s1, s2, hb))) - 1) > 1e-3          # non-unitary beyond the line
print('   (A < 0 beyond the line: the classical two-soliton solution is singular there; Phase 1)')

# ---------------------------------------------------------------- 3. breather kinematics
print('3. complex pairs alpha_{1,2} = a -+ i b (right-movers), hbar = 0.1')
hb = mp.mpf('0.1')
for a0 in (mp.mpf('0.4'), mp.mpf('0.6')):
    for N in (1, 2):
        f = lambda b: mp.re(I*mp.mpf(2)/9*(sol(a0 - I*b, 1)['ell'] - sol(a0 + I*b, 1)['ell'])/hb) + N
        try:
            b = mp.findroot(f, mp.mpf('0.3'))
        except Exception:
            print('   a=%s N=%d: no pole found' % (a0, N)); continue
        s1, s2 = sol(a0 - I*b, 1), sol(a0 + I*b, 1)
        ie = I*mp.mpf(2)/9*(s1['ell'] - s2['ell'])/hb
        print('   a=%s N=%d: b=%s  i eta=%s  P_tot=%s  Im(P1-P2)=%s  A=%s'
              % (a0, N, mp.nstr(b, 8), mp.nstr(ie, 6), mp.nstr(s1['P'] + s2['P'], 6), mp.nstr(mp.im(s1['P'] - s2['P']), 4),
                 mp.nstr(A12(s1, s2), 6)))
        ok &= abs(ie + N) < 1e-25 and abs(mp.im(s1['P'] + s2['P'])) < 1e-25 and mp.re(A12(s1, s2)) < 1
print('   -> Gamma(1 + i eta) poles at kinematics where every classical breather is singular (Phase 1)')

# ---------------------------------------------------------------- 4. defect fusion
def deltaD(s3, q, eta):         # defect_transmission_phase.py, s = -2 sqrt3 q
    s = -2*r3*q; e = s3['eps']*eta
    z = ((s3['pb'] - q)/(s3['pa'] - q))**e
    return (mp.mpf(2)/9*(s3['ell'] - (3*s - s**3)/2)*mp.log(z)
            - mp.mpf(2)/3*(s3['Pi']*e*q - e*(s**2 - 2)/(2*r3)*s3['M']))


def zpos(s3, q, eta):
    return (s3['pb'] - q)/(s3['pa'] - q) > 0


def sol_from_pair(x, y):
    pa, pb = max(x, y), min(x, y); eps = 1 if pa + pb > 0 else -1
    s_ = sol(mp.acos(r3*abs(pa + pb)), eps)
    assert abs(s_['pa'] - pa) + abs(s_['pb'] - pb) < 1e-30
    return s_


print('4. defect fusion around a KP triple')
worst_lead = 0; worst_exp = 0; ntri = 0
for qv in ('-0.2', '0.1', '0.45'):
    q = mp.mpf(qv); disc = mp.sqrt(1 - 3*q*q); qs = [q, (-q + disc)/2, (-q - disc)/2]
    for eta in (1, -1):
        for e3 in (1, -1):
            sols = [sol_from_pair(qs[i], qs[(i + 1) % 3]) for i in range(3)]
            # regular test-soliton parameters for all three defects and all three solitons
            a3s = [a for a in mp.linspace(0.05, 1.0, 40)
                   if all(zpos(sol(a, e3), x, eta) for x in qs) and all(A12(sol(a, e3), S_) > 0 for S_ in sols)]
            if len(a3s) < 4:
                continue
            rho = []
            for i, S_ in enumerate(sols):
                rr = []
                for a3 in a3s[:4]:
                    s3 = sol(a3, e3)
                    L = deltaD(s3, qs[(i + 1) % 3], eta) - deltaD(s3, qs[i], eta)
                    rr.append(L/(delta(s3, S_) + S_['M']*s3['Pi']))
                assert all(abs(r - rr[0]) < 1e-30 for r in rr) and abs(abs(rr[0]) - 1) < 1e-30
                rho.append(int(mp.nint(rr[0])))
            for a3 in a3s:
                s3 = sol(a3, e3)
                lead = sum(rho[i]*(delta(s3, S_) + S_['M']*s3['Pi']) for i, S_ in enumerate(sols))
                worst_lead = max(worst_lead, abs(lead))
            tau = [1 if S_['eps'] == e3 else -1 for S_ in sols]
            nrt = sum(r*t for r, t in zip(rho, tau))
            s3 = sol(a3s[len(a3s)//2], e3)
            devs = []
            for hb in (mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.001')):
                tot = sum(rho[i]*(logS(s3, S_, hb) + I*S_['M']*s3['Pi']/hb) for i, S_ in enumerate(sols))
                tot0 = sum(rho[i]*(logS(s3, S_, hb, gamma=False) + I*S_['M']*s3['Pi']/hb) for i, S_ in enumerate(sols))
                devs.append(wrap(mp.im(tot))); worst_exp = max(worst_exp, abs(wrap(mp.im(tot0))) + abs(mp.re(tot0)))
            eta3 = mp.sign(s3['ell'] - sols[0]['ell'])
            print('   q=%5s eta=%+d eps3=%+d: rho=%s tau=%s sum rho tau=%+d; candidate closure phase at hbar=0.1,0.01,0.001: %s'
                  ' (pi/2 * sum * sgn eta = %s)' % (qv, eta, e3, rho, tau, nrt, [mp.nstr(d, 6) for d in devs],
                                                    mp.nstr(mp.pi/2*nrt*eta3, 6)))
            ok &= abs(nrt) == 1 and abs(devs[2] - wrap(mp.pi/2*nrt*eta3)) < 1e-2
            ntri += 1
print('   leading-order closure, max |sum rho_i (delta(3,S_i) + M_i Pi_3)| over %d triangles: %s' % (ntri, mp.nstr(worst_lead, 3)))
print('   pure exponential e^{i delta/hbar}: max closure defect %s' % mp.nstr(worst_exp, 3))
ok &= worst_lead < 1e-30 and worst_exp < 1e-25 and ntri >= 6
assert ok
print('ALL CHECKS PASSED')

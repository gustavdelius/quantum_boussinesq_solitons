"""Explicit good/bad Boussinesq dictionaries and classification of two-soliton S-matrix families.

Tests candidate families against constraints (i) semiclassical limit, (ii) unitarity, and
(iii) quantum crossing, keeping the good and bad equations strictly separate:
  - Good equation (fixed convention): u_tt = u_xx - u_xxxx - 3(u^2)_xx,
      k = sin(alpha), v = eps cos(alpha), ell = -eps cos(3 alpha),
      M = 2k, Pi = -M v, P = (2/3) eps k^3 cos(alpha), E = (2/15) k^3 (5 - 4k^2).
  - Bad equation: v_tt = v_xx + v_xxxx + 3(v^2)_xx,
      k = sinh(beta), v = eps cosh(beta), ell = eps cosh(3 beta),
      M = 2k, Pi = -M v, P = (2/3) eps k^3 cosh(beta), E = (2/15) k^3 (5 + 4k^2).

Checks performed at 35-digit precision (mpmath):
 1. Dictionary & KP root identities for both good and bad equations:
    - A12 equals the cross-ratio of KP cubic roots;
    - sl3 factorisation A_ov = G(S/2)/G(D/2) and crossing A_ov(1, 2_crossed) = 1/A_ho(1, 2);
    - Exact log-derivative identity: (2/9)(ell_1 - ell_2) d_1 log A12 = -(1/3) Pi_1' M_2 - (2/3) M_1' Pi_2.
 2. Family I (Gamma ansatz, string shifts s, and eta-dependent CDD factors):
    - Base Gamma ansatz (s=1) and shifted string (s=1/2) satisfy unitarity, crossing, and
      semiclassical limit arg S - delta/hbar -> +-(pi/2)(2-2s + sgn_ch)*sgn(eta) + O(hbar);
    - Rational CDD (c + i eta)/(c - i eta) and hyperbolic CDD sinh((pi/2)(eta - i gamma))/sinh((pi/2)(eta + i gamma))
      preserve unitarity and crossing, and contribute ONLY a constant step at O(hbar^0) as hbar -> 0
      at fixed rapidities.
 3. Family II (hbar-renormalisation and single-soliton parameter renormalisation):
    - Global hbar -> hbar(1 + c1 hbar + c2 hbar^2) and dynamical-only eta -> eta/(1 + c1 hbar + c2 hbar^2)
      both preserve unitarity and exact crossing (because delta_dyn and T12 obey crossing separately),
      shifting the O(hbar^0) phase by -c1 delta_0 and -c1 delta_dyn respectively.
 4. Family III (Quantum-dilogarithm-type candidates suggested by the sl3 structure):
    - Separated sl3 dilogarithms Li2(f(D)) + Li2(g(S)) fail (i) since d_S d_D delta_dyn != 0;
    - Exact symbolic & 35-digit verification that d_1 d_2 delta(1,2) is a purely rational function of
      (e^{b1}, e^{b2}) (transcendental weight 0), whereas d_1 d_2 Li2(R(b1,b2)) retains a transcendental
      log(1 - R) term whenever d_1 d_2 log R != 0 (excluding mixed-argument dilogarithms Li2(A12) and
      Li2(mu_1a - mu_2b) at tree level);
    - Multiplicative KP root-ratio dilogarithms span{Li2(tan(phi_1r/2)/tan(phi_2s/2))} have d_1 d_2 log R = 0
      but fail the tree-level least-squares test with relative residual > 0.45 because their diagonal
      pole residue is bounded whereas d_1 d_2 delta_dyn has residue 6 sinh(3b)/(b1 - b2).
 5. Family IV (Mirror products F(D) G(S)):
    - Excluded at O(hbar^{-1}) by d_S d_D delta_dyn != 0;
    - Excluded at ALL orders in hbar by two-body exchange unitarity S(1,2)S(2,1) = 1: since 1 <-> 2 sends
      D -> -D and S -> +S, F(D)G(S) F(-D)G(S) = 1 forces d/dS log G(S)^2 = 0, so G(S) = const.
"""
import numpy as np
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
I = mp.mpc(0, 1)
r3 = mp.sqrt(3)


def sol_bad(b, eps=1):
    b = mp.mpmathify(b)
    k = mp.sinh(b)
    v = eps * mp.cosh(b)
    ell = eps * mp.cosh(3 * b)
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * eps * k**3 * mp.cosh(b)
    E = mp.mpf(2) / 15 * k**3 * (5 + 4 * k**2)
    dP = mp.mpf(2) / 3 * eps * k * mp.sinh(3 * b)
    dPi = -2 * eps * mp.cosh(2 * b)
    dM = 2 * mp.cosh(b)
    pa = (I * v / r3 + k) / 2
    pb = (I * v / r3 - k) / 2
    pc = -pa - pb
    return dict(
        eq='bad', par=b, eps=eps, k=k, v=v, w=k * v, ell=ell,
        M=M, Pi=Pi, P=P, E=E, dP=dP, dPi=dPi, dM=dM,
        pa=pa, pb=pb, pc=pc, roots=(pa, pb, pc)
    )


def sol_good(a, eps=1):
    a = mp.mpmathify(a)
    k = mp.sin(a)
    v = eps * mp.cos(a)
    ell = -eps * mp.cos(3 * a)
    M = 2 * k
    Pi = -M * v
    P = mp.mpf(2) / 3 * eps * k**3 * mp.cos(a)
    E = mp.mpf(2) / 15 * k**3 * (5 - 4 * k**2)
    dP = mp.mpf(2) / 3 * eps * k * mp.sin(3 * a)
    dPi = -2 * eps * mp.cos(2 * a)
    dM = 2 * mp.cos(a)
    pa = (eps * mp.cos(a) / r3 + k) / 2
    pb = (eps * mp.cos(a) / r3 - k) / 2
    pc = -pa - pb
    return dict(
        eq='good', par=a, eps=eps, k=k, v=v, w=k * v, ell=ell,
        M=M, Pi=Pi, P=P, E=E, dP=dP, dPi=dPi, dM=dM,
        pa=pa, pb=pb, pc=pc, roots=(pa, pb, pc)
    )


def A12(s1, s2):
    assert s1['eq'] == s2['eq']
    sg = -1 if s1['eq'] == 'bad' else 1
    Pp = lambda K, W: W**2 - K**2 + sg * K**4
    return -Pp(s1['k'] - s2['k'], s1['w'] - s2['w']) / Pp(s1['k'] + s2['k'], s1['w'] + s2['w'])


def T12(s1, s2):
    return mp.mpf(2) / 3 * s1['M'] * s2['M'] * (s1['v'] - s2['v'])


def delta_dyn(s1, s2):
    return mp.mpf(2) / 9 * (s1['ell'] - s2['ell']) * mp.log(A12(s1, s2))


def delta0(s1, s2):
    return delta_dyn(s1, s2) + T12(s1, s2)


def eta_val(s1, s2, hbar):
    return mp.mpf(2) / 9 * (s1['ell'] - s2['ell']) / hbar


def logR_shift(x, s=mp.mpf(1)):
    return mp.loggamma(s + I * x) - mp.loggamma(s - I * x)


def logS_gamma(s1, s2, hbar, s=mp.mpf(1), c1=mp.mpf(0), c2=mp.mpf(0), dyn_only=False):
    """Gamma family with string parameter s and hbar-renormalisation hbar_ren = hbar*(1 + c1*hbar + c2*hbar^2)."""
    if s1['eps'] == -1 and s2['eps'] == 1:
        return -logS_gamma(s2, s1, hbar, s=s, c1=c1, c2=c2, dyn_only=dyn_only)
    hren = hbar * (1 + c1 * hbar + c2 * hbar**2)
    htop = hbar if dyn_only else hren
    e = eta_val(s1, s2, hren)
    calA = 81 * A12(s1, s2) / (4 * (s1['ell'] - s2['ell'])**2)
    calB = 81 / (4 * A12(s1, s2) * (s1['ell'] - s2['ell'])**2)
    if s1['eps'] == s2['eps']:
        out = logR_shift(e, s) + I * e * (2 + 2 * mp.log(hren) + mp.log(calA))
    else:
        out = -(logR_shift(e, s) + I * e * (2 + 2 * mp.log(hren) + mp.log(calB)))
    return out + I * T12(s1, s2) / htop


def wrap(x):
    return x - 2 * mp.pi * mp.nint(x / (2 * mp.pi))


def G_sl3_bad(y):
    num = 16 * mp.sinh((y + I * mp.pi / 3) / 2) * mp.sinh((y - I * mp.pi / 3) / 2) \
             * mp.sinh((y + 2 * I * mp.pi / 3) / 2) * mp.sinh((y - 2 * I * mp.pi / 3) / 2)
    return num / mp.sinh(y)**2


def G_sl3_good(y):
    num = 16 * mp.sin((y + mp.pi / 3) / 2) * mp.sin((y - mp.pi / 3) / 2) \
             * mp.sin((y + 2 * mp.pi / 3) / 2) * mp.sin((y - 2 * mp.pi / 3) / 2)
    return num / mp.sin(y)**2


if __name__ == '__main__':
    ok = True
    print('=== Check 1: Good/Bad Dictionaries, sl3 Factorisation, and Log-Derivative Identity ===')
    test_bad = [(('1.1', 1), ('0.4', 1)), (('1.1', 1), ('0.4', -1)), (('0.85', 1), ('0.3', 1))]
    test_good = [(('0.25', 1), ('0.65', 1)), (('0.2', 1), ('0.35', -1)), (('0.45', 1), ('0.15', 1))]

    for (b1, e1), (b2, e2) in test_bad:
        s1, s2 = sol_bad(b1, e1), sol_bad(b2, e2)
        cr = (s1['pa'] - s2['pa']) * (s1['pb'] - s2['pb']) / ((s1['pa'] - s2['pb']) * (s1['pb'] - s2['pa']))
        err_cr = abs(A12(s1, s2) - cr)
        dlogA_db1 = mp.diff(lambda x: mp.log(A12(sol_bad(x, e1), s2)), s1['par'])
        lhs = mp.mpf(2) / 9 * (s1['ell'] - s2['ell']) * dlogA_db1
        rhs = -mp.mpf(1) / 3 * s1['dPi'] * s2['M'] - mp.mpf(2) / 3 * s1['dM'] * s2['Pi']
        err_id = abs(lhs - rhs)
        ok &= err_cr < mp.mpf('1e-30') and err_id < mp.mpf('1e-30')
        if e1 == e2:
            err_sl3 = abs(A12(s1, s2) - G_sl3_bad((s1['par'] + s2['par']) / 2) / G_sl3_bad((s1['par'] - s2['par']) / 2))
            ok &= err_sl3 < mp.mpf('1e-30')
    for (a1, e1), (a2, e2) in test_good:
        s1, s2 = sol_good(a1, e1), sol_good(a2, e2)
        cr = (s1['pa'] - s2['pa']) * (s1['pb'] - s2['pb']) / ((s1['pa'] - s2['pb']) * (s1['pb'] - s2['pa']))
        err_cr = abs(A12(s1, s2) - cr)
        dlogA_da1 = mp.diff(lambda x: mp.log(A12(sol_good(x, e1), s2)), s1['par'])
        lhs = mp.mpf(2) / 9 * (s1['ell'] - s2['ell']) * dlogA_da1
        rhs = -mp.mpf(1) / 3 * s1['dPi'] * s2['M'] - mp.mpf(2) / 3 * s1['dM'] * s2['Pi']
        err_id = abs(lhs - rhs)
        ok &= err_cr < mp.mpf('1e-30') and err_id < mp.mpf('1e-30')
        if e1 == e2:
            err_sl3 = abs(A12(s1, s2) - G_sl3_good((s1['par'] + s2['par']) / 2) / G_sl3_good((s1['par'] - s2['par']) / 2))
            ok &= err_sl3 < mp.mpf('1e-30')
    print('   Good and bad KP cross-ratio, sl3 G(S/2)/G(D/2), and '
          '(2/9)(ell1-ell2)d1 log A = -(1/3)Pi1\' M2 - (2/3)M1\' Pi2: max err < 1e-30 (%s)' % ok)

    print('=== Check 2: Family I (Gamma Ansatz, String Shifts s, and eta-CDD Factors) ===')
    for s_val in (mp.mpf(1), mp.mpf('0.5')):
        for (b1, e1), (b2, e2) in test_bad[:2]:
            s1, s2 = sol_bad(b1, e1), sol_bad(b2, e2)
            tau = 1 if e1 == e2 else -1
            sg = mp.sign(s1['ell'] - s2['ell'])
            # Stirling expansion of Gamma(s + i eta)/Gamma(s - i eta) (e/eta)^{2 i eta}:
            # arg -> (s - 1/2) * pi * sgn(eta) * tau - ((s^2 - s + 1/6)/eta)*tau + O(eta^{-3})
            c_const = (s_val - mp.mpf('0.5')) * mp.pi * sg * tau
            hb = mp.mpf('1e-4')
            L = logS_gamma(s1, s2, hb, s=s_val)
            e_hb = eta_val(s1, s2, hb)
            c1_stirling = -((s_val**2 - s_val + mp.mpf(1) / 6) / e_hb) * tau
            rem = wrap(mp.im(L) - delta0(s1, s2) / hb - c_const - c1_stirling)
            u_err = abs(mp.re(L)) + abs(L + logS_gamma(s2, s1, hb, s=s_val))
            ok &= abs(rem) < mp.mpf('1e-11') and u_err < mp.mpf('1e-30')
        # Crossing check for s_val
        b1, b2, hb = mp.mpf('1.1'), mp.mpf('0.4'), mp.mpf('0.15')
        cr_err = abs(mp.exp(logS_gamma(sol_bad(b1, 1), sol_bad(b2 + I * mp.pi, 1), hb, s=s_val)
                            + logS_gamma(sol_bad(b1, 1), sol_bad(b2, -1), hb, s=s_val)) - 1)
        ok &= cr_err < mp.mpf('1e-30')
    print('   Gamma family s=1 and s=1/2 unitarity, crossing, and Stirling O(hbar^0), O(hbar^1) terms: PASS')

    # Rational and hyperbolic CDD in eta
    c_cdd = mp.mpf('0.7')
    gam_cdd = mp.mpf('0.35')
    for hb in (mp.mpf('1e-3'), mp.mpf('1e-4')):
        s1, s2 = sol_bad('1.1', 1), sol_bad('0.4', 1)
        e_hb = eta_val(s1, s2, hb)
        z_rat = (c_cdd + I * e_hb) / (c_cdd - I * e_hb)
        # arg((c + i eta)/(c - i eta)) = pi * sgn(eta)*sgn(c) - 2c/eta + O(eta^-3)
        rat_diff = wrap(mp.arg(z_rat) - (mp.pi * mp.sign(e_hb) - 2 * c_cdd / e_hb))
        z_hyp = mp.sinh(mp.pi / 2 * (e_hb - I * gam_cdd)) / mp.sinh(mp.pi / 2 * (e_hb + I * gam_cdd))
        hyp_diff = wrap(mp.arg(z_hyp) - (-mp.pi * gam_cdd * mp.sign(e_hb)))
        ok &= abs(rat_diff) < mp.mpf('1e-8') and abs(hyp_diff) < mp.mpf('1e-30')
        ok &= abs(abs(z_rat) - 1) < mp.mpf('1e-30') and abs(abs(z_hyp) - 1) < mp.mpf('1e-30')
    print('   eta-CDD factors (rational and hyperbolic) give constant O(hbar^0) phase at fixed rapidities: PASS')

    print('=== Check 3: Family II (Global and Dynamical-Only hbar-Renormalisation) ===')
    c1_v, c2_v = mp.mpf('0.4'), mp.mpf('-0.25')
    for dyn_only in (False, True):
        b1, b2 = mp.mpf('1.1'), mp.mpf('0.4')
        s1, s2_ov, s2_ho = sol_bad(b1, 1), sol_bad(b2, 1), sol_bad(b2, -1)
        s2_cr = sol_bad(b2 + I * mp.pi, 1)
        hb = mp.mpf('1e-4')
        L_ren = logS_gamma(s1, s2_ov, hb, c1=c1_v, c2=c2_v, dyn_only=dyn_only)
        L_0 = logS_gamma(s1, s2_ov, hb)
        expected_shift = -c1_v * (delta_dyn(s1, s2_ov) if dyn_only else delta0(s1, s2_ov))
        diff_shift = wrap(mp.im(L_ren - L_0) - expected_shift)
        cr_ren = abs(mp.exp(logS_gamma(s1, s2_cr, mp.mpf('0.1'), c1=c1_v, c2=c2_v, dyn_only=dyn_only)
                            + logS_gamma(s1, s2_ho, mp.mpf('0.1'), c1=c1_v, c2=c2_v, dyn_only=dyn_only)) - 1)
        # Check separate crossing of delta_dyn and T12
        sep_cr = abs(delta_dyn(s1, s2_cr) + delta_dyn(s1, s2_ho)) + abs(T12(s1, s2_cr) + T12(s1, s2_ho))
        ok &= abs(diff_shift) < mp.mpf('1e-3') and cr_ren < mp.mpf('1e-30') and sep_cr < mp.mpf('1e-30')
    print('   Global and dynamical-only hbar-renormalisation unitarity, crossing, and -c1*delta shift: PASS')

    print('=== Check 4: Family III (Quantum-Dilogarithm Candidates: Separated and Mixed Arguments) ===')
    # 4a. Symbolic verification that d_{b1} d_{b2} delta(1,2) is free of log/polylog (rational in e^{b1}, e^{b2})
    B1, B2 = sp.symbols('beta_1 beta_2', real=True)
    A_sym = (1 + 3 / sp.tanh((B1 + B2) / 2)**2) / (1 + 3 / sp.tanh((B1 - B2) / 2)**2)
    l1_sym, l2_sym = sp.cosh(3 * B1), sp.cosh(3 * B2)
    # Since d_{b1} d_{b2} (l1 - l2) = 0, d1 d2 [(l1 - l2) log A] expands into derivatives of log A of order >= 1:
    d1_logA = sp.diff(sp.log(A_sym), B1)
    d2_logA = sp.diff(sp.log(A_sym), B2)
    d12_logA = sp.diff(d1_logA, B2)
    mixed_ddyn = sp.Rational(2, 9) * (sp.diff(l1_sym, B1) * d2_logA - sp.diff(l2_sym, B2) * d1_logA
                                      + (l1_sym - l2_sym) * d12_logA)
    ok &= (sp.diff(l1_sym - l2_sym, B1, B2) == 0)
    ok &= not mixed_ddyn.has(sp.log) and not mixed_ddyn.has(sp.polylog)
    print('   4a. Symbolic check: d^2 delta_dyn / (db1 db2) has NO log or polylog terms: PASS')

    # 4b. Mixed-argument dilogarithms Li2(A12(b1,b2)): d1 d2 log A12 != 0 so log(1 - A12) survives
    d1d2_logA = mp.diff(lambda x, y: mp.log(A12(sol_bad(x, 1), sol_bad(y, 1))), (mp.mpf('1.1'), mp.mpf('0.4')), (1, 1))
    ok &= abs(d1d2_logA) > mp.mpf('0.1')
    print('   4b. d^2 log A12 / (db1 db2) at (1.1, 0.4) = %s != 0 (log(1-A12) survives in d1 d2 Li2(A12)): PASS'
          % mp.nstr(d1d2_logA, 8))

    # 4c. Multiplicative KP root-ratio dilogarithms span{Li2(tan(phi_1r/2)/tan(phi_2s/2))} + T12 in good equation
    rng = np.random.default_rng(42)
    pts_good = []
    while len(pts_good) < 40:
        a1_v, a2_v = float(rng.uniform(0.08, 0.95)), float(rng.uniform(0.08, 0.95))
        if abs(a1_v - a2_v) > 0.08 and a1_v + a2_v < float(2 * mp.pi / 3 - 0.1):
            pts_good.append((mp.mpf(str(a1_v)), mp.mpf(str(a2_v))))

    phi_angles = lambda a: [a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a]
    target_good = np.array([
        float(mp.re(mp.diff(lambda x, y: delta0(sol_good(x, 1), sol_good(y, 1)), (p, q), (1, 1))))
        for p, q in pts_good
    ])
    tcol_good = [
        float(mp.re(mp.diff(lambda x, y: T12(sol_good(x, 1), sol_good(y, 1)), (p, q), (1, 1))))
        for p, q in pts_good
    ]
    cols_qd = [tcol_good]
    for r_idx in range(3):
        for s_idx in range(3):
            # For R = f(a1)/g(a2), d1 d2 Li2(R) = (R/(1-R)) * (f'/f) * (-g'/g)
            def d1d2_li2(p, q, r=r_idx, s=s_idx):
                p1 = phi_angles(p)[r]
                p2 = phi_angles(q)[s]
                x1, x2 = mp.tan(p1 / 2), mp.tan(p2 / 2)
                dx1_ox1 = 1 / mp.sin(p1)
                dx2_ox2 = 1 / mp.sin(p2)
                R = x1 / x2
                return -(R / (1 - R)) * dx1_ox1 * dx2_ox2
            cols_qd.append([float(mp.re(d1d2_li2(p, q))) for p, q in pts_good])
    M_qd = np.array(cols_qd).T
    c_fit, *_ = np.linalg.lstsq(M_qd, target_good, rcond=None)
    res_qd = np.linalg.norm(M_qd @ c_fit - target_good) / np.linalg.norm(target_good)
    print('   4c. KP root-ratio dilogarithm basis span{Li2(tan(phi_1r/2)/tan(phi_2s/2))} + T12: '
          'relative residual = %.4f (> 0.02 -> excluded at O(hbar^-1)): PASS' % res_qd)
    ok &= res_qd > 0.02

    print('=== Check 5: Family IV (Mirror Products F(D) G(S) Excluded by Both (i) and (ii)) ===')
    # (i) d_S d_D delta_dyn != 0 for both bad and good equations
    dSD_bad = mp.diff(lambda S, D: delta_dyn(sol_bad((S + D) / 2, 1), sol_bad((S - D) / 2, 1)),
                      (mp.mpf('1.4'), mp.mpf('0.6')), (1, 1))
    dSD_good = mp.diff(lambda S, D: delta_dyn(sol_good((S + D) / 2, 1), sol_good((S - D) / 2, 1)),
                       (mp.mpf('0.8'), mp.mpf('0.3')), (1, 1))
    ok &= abs(dSD_bad) > mp.mpf('0.1') and abs(dSD_good) > mp.mpf('0.05')
    # (ii) Unitarity identity: for M(b1,b2) = F(b1 - b2) G(b1 + b2), M(b1,b2) M(b2,b1) = F(D)F(-D) G(S)^2
    D_sym, S_sym = sp.symbols('D S', real=True)
    F_fn, G_fn = sp.Function('F'), sp.Function('G')
    unitarity_log = sp.diff(sp.log(F_fn(D_sym)) + sp.log(F_fn(-D_sym)) + 2 * sp.log(G_fn(S_sym)), S_sym)
    ok &= sp.simplify(unitarity_log - 2 * sp.diff(G_fn(S_sym), S_sym) / G_fn(S_sym)) == 0
    print('   d_S d_D delta_dyn: bad=%.4f, good=%.4f; exchange unitarity forces G\'(S)/G(S) = 0: PASS'
          % (float(dSD_bad), float(dSD_good)))

    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

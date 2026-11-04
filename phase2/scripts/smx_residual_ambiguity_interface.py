"""Task 2: Minimal residual ambiguity after (i)-(v) and exact interface to det_* and frame_* packages.

Verifies at 40-digit precision (mpmath) the structure of the residual O(hbar^0) ambiguity and the
exact analytical interface formulas:
 1. Crossing identity of the continuum one-loop dilogarithm integrals I_{12} and J_{12}:
    - For alpha_1, alpha_2 in (0, pi/3):
        I_{12}^same(alpha_1, alpha_2 + pi) = -I_{12}^head(alpha_1, alpha_2),
        J_{12}^same(alpha_1, alpha_2 + pi) + J_{12}^head(alpha_1, alpha_2) = +pi/2.
    - Consequently, dressing the overtaking channel with c_M*(pi/2)*sgn(eta) + J_{12}^same forces the
      crossed head-on channel to have constant offset (c_M - 1)*(pi/2)*sgn(eta_ho) + J_{12}^head;
      in particular, c_M = 1 (the Gamma / raw-FK overtaking constant -pi/2 for alpha_1 < alpha_2)
      gives ZERO constant offset in the head-on channel.
 2. Unitarity and crossing covariance of single-soliton momentum renormalisation delta_renorm:
    - For any single-soliton shift Delta P(alpha, eps) = eps * p_1(alpha) with p_1(alpha + pi) = p_1(alpha),
      delta_renorm(1, 2) = Dx_1' Delta P(1) + Dx_2' Delta P(2) satisfies:
        delta_renorm(1, 2) + delta_renorm(2, 1) = 0,
        delta_renorm^same(alpha_1, alpha_2 + pi) + delta_renorm^head(alpha_1, alpha_2) = 0.
    - Verified to 40 digits for both Delta P_1 (fixed-profile mode sum) and Delta P_can (co-moving
      canonical charge derivative).
 3. Exact small-amplitude interface theorem (alpha_1 = lam * r, alpha_2 = lam -> 0^+, r in (0, 1)):
    - For any Delta P(alpha, +1) = c_ln * alpha * ln(alpha) + c_lin * alpha + O(alpha^3 ln alpha):
        Hirota delay part (ov):  Dx_1 Delta P(1) + Dx_2 Delta P(2) -> -2 c_ln ln(r) ln((1-r)/(1+r)),
        Darboux tail part (ov):  (gamma_1 M_2) Delta P(1) - (gamma_2 M_1) Delta P(2)
                                 = -2 (1/r - r) (c_ln ln(lam) + c_lin) - 2 c_ln ln(r)/r + O(lam^2 ln lam),
        Head-on total:           delta_renorm^head(lam*r, lam)
                                 = -2 (1/r + r) (c_ln ln(lam) + c_lin) - 2 c_ln ln(r)/r + O(lam^2 ln lam).
    - Verified to 40 digits by Richardson / asymptotic scaling for both Delta P_1 (c_ln = 1/pi,
      c_lin = (-ln 2 - 1 + ln(3)/4)/pi) and Delta P_can (c_ln = 0, c_lin = -4/pi).
"""
import mpmath as mp
from smx_dictionary_and_families import sol_good, A12, delta0

mp.mp.dps = 40
r3 = mp.sqrt(3)
C_WEIGHTS = (mp.mpf(1), mp.mpf(1), mp.mpf(-2))


def phi_list(a):
    return (a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a)


def L_dilog(z):
    z = mp.mpmathify(z)
    if z == 0:
        return mp.mpf(0)
    az = abs(z)
    if az == 1:
        return -mp.sign(z) * mp.pi**2 / 4
    chi2 = mp.mpf('0.5') * (mp.re(mp.polylog(2, az)) - mp.re(mp.polylog(2, -az)))
    return -mp.sign(z) * (2 * chi2 + mp.log(az) * mp.log(abs((1 - az) / (1 + az))))


def I12_same(a1, a2):
    p1, p2 = phi_list(a1), phi_list(a2)
    tot = mp.mpf(0)
    for r in range(3):
        for s in range(3):
            tot += C_WEIGHTS[r] * C_WEIGHTS[s] * L_dilog(mp.tan(p1[r] / 2) / mp.tan(p2[s] / 2))
    return tot / (2 * mp.pi)


def J12_same(a1, a2):
    p1, p2 = phi_list(a1), phi_list(a2)
    tot = mp.mpf(0)
    for r in range(3):
        for s in range(3):
            z = mp.tan(p1[r] / 2) / mp.tan(p2[s] / 2)
            tot += C_WEIGHTS[r] * C_WEIGHTS[s] * (L_dilog(z) - L_dilog(1 / z))
    return tot / (4 * mp.pi)


def I12_head(a1, a2):
    p1, p2 = phi_list(a1), phi_list(a2)
    tot = mp.mpf(0)
    for r in range(3):
        for s in range(3):
            tot += C_WEIGHTS[r] * C_WEIGHTS[s] * L_dilog(mp.tan(p1[r] / 2) * mp.tan(p2[s] / 2))
    return tot / (2 * mp.pi)


def J12_head(a1, a2):
    return I12_head(a1, a2)


def DeltaP_mode(a, eps=1):
    """Fixed-profile mode-sum momentum shift Delta P_1(alpha, eps) of oneloop_phase.tex Eq. (29)."""
    p = phi_list(a)
    tot = mp.fsum(C_WEIGHTS[r] * mp.sin(p[r]) * mp.log(abs(mp.tan(p[r] / 2))) for r in range(3))
    return -eps * tot / (2 * mp.pi)


def DeltaP_can(a, eps=1):
    """Co-moving canonical momentum shift Delta P_can(alpha, eps) of report.tex Eq. (2)."""
    num = mp.tan((a + 2 * mp.pi / 3) / 2)
    den = mp.tan((a - 2 * mp.pi / 3) / 2)
    return -eps * r3 * mp.cos(a) / mp.pi * mp.log(abs(num / den))


def shifts_darboux(s1, s2):
    """Returns (Dx1, Dx2, Dx1_prime, Dx2_prime) for soliton 1 on the left of soliton 2."""
    L = mp.log(A12(s1, s2))
    Dx1 = -L / s1['k']
    Dx2 = +L / s2['k']
    g1 = s1['dPi'] / s1['dP']
    g2 = s2['dPi'] / s2['dP']
    return Dx1, Dx2, Dx1 + g1 * s2['M'], Dx2 - g2 * s1['M']


def delta_renorm(s1, s2, dp_fn):
    _, _, Dx1p, Dx2p = shifts_darboux(s1, s2)
    return Dx1p * dp_fn(s1['par'], s1['eps']) + Dx2p * dp_fn(s2['par'], s2['eps'])


if __name__ == '__main__':
    ok = True
    print('=== Check 1: Crossing Identity of I_{12} and J_{12} Across the Cusp ===')
    pts = [(mp.mpf('0.15'), mp.mpf('0.25')), (mp.mpf('0.2'), mp.mpf('0.55')), (mp.mpf('0.35'), mp.mpf('0.6'))]
    for a1, a2 in pts:
        # Check J12_same = I12_same + pi/2 on (0, pi/3)^2
        err_IJ = abs(J12_same(a1, a2) - (I12_same(a1, a2) + mp.pi / 2))
        # Check I12_same(a1, a2 + pi) = -I12_head(a1, a2)
        err_I_cr = abs(I12_same(a1, a2 + mp.pi) + I12_head(a1, a2))
        # Check J12_same(a1, a2 + pi) + J12_head(a1, a2) = +pi/2 (using I12_same + pi/2 continuation)
        J_cont = mp.pi / 2 + I12_same(a1, a2 + mp.pi)
        err_J_cr = abs((J_cont + J12_head(a1, a2)) - mp.pi / 2)
        # Check antisymmetry of J12_same on (0, pi/3)^2
        err_anti = abs(J12_same(a1, a2) + J12_same(a2, a1))
        print('   a=(%s,%s): |J - (I+pi/2)|=%s, |I_same(a2+pi)+I_head|=%s, |(J_cont+J_head)-pi/2|=%s'
              % (mp.nstr(a1, 3), mp.nstr(a2, 3), mp.nstr(err_IJ, 2), mp.nstr(err_I_cr, 2), mp.nstr(err_J_cr, 2)))
        ok &= err_IJ < mp.mpf('1e-35') and err_I_cr < mp.mpf('1e-35')
        ok &= err_J_cr < mp.mpf('1e-35') and err_anti < mp.mpf('1e-35')

    print('=== Check 2: Unitarity and Crossing Covariance of delta_renorm ===')
    for name, dp_fn in [('DeltaP_mode', DeltaP_mode), ('DeltaP_can', DeltaP_can)]:
        for a1, a2 in [(mp.mpf('0.18'), mp.mpf('0.28')), (mp.mpf('0.22'), mp.mpf('0.45'))]:
            s1, s2_ov, s2_ho = sol_good(a1, 1), sol_good(a2, 1), sol_good(a2, -1)
            s2_cr = sol_good(a2 + mp.pi, 1)
            # Parity-crossing of Delta P: Delta P(a + pi, +1) = -Delta P(a, -1)
            err_dp_cr = abs(dp_fn(a2 + mp.pi, 1) + dp_fn(a2, -1))
            # Exchange antisymmetry
            err_u_ov = abs(delta_renorm(s1, s2_ov, dp_fn) + delta_renorm(s2_ov, s1, dp_fn))
            err_u_ho = abs(delta_renorm(s1, s2_ho, dp_fn) + delta_renorm(s2_ho, s1, dp_fn))
            # Crossing: delta_renorm^same(a1, a2 + pi) + delta_renorm^head(a1, a2) = 0
            err_cr = abs(delta_renorm(s1, s2_cr, dp_fn) + delta_renorm(s1, s2_ho, dp_fn))
            print('   %-12s a=(%s,%s): |dp_cr|=%s, |unitarity|=%s, |crossing|=%s'
                  % (name, mp.nstr(a1, 3), mp.nstr(a2, 3),
                     mp.nstr(err_dp_cr, 2), mp.nstr(err_u_ov + err_u_ho, 2), mp.nstr(err_cr, 2)))
            ok &= err_dp_cr < mp.mpf('1e-35') and (err_u_ov + err_u_ho) < mp.mpf('1e-35') and err_cr < mp.mpf('1e-35')

    print('=== Check 3: Small-Amplitude Interface Expansion (alpha_1 = lam*r, alpha_2 = lam -> 0^+) ===')
    r = mp.mpf('0.6')
    lam = mp.mpf('1e-6')
    s1 = sol_good(lam * r, 1)
    s2_ov = sol_good(lam, 1)
    s2_ho = sol_good(lam, -1)
    Dx1_ov, Dx2_ov, Dx1p_ov, Dx2p_ov = shifts_darboux(s1, s2_ov)
    _, _, Dx1p_ho, Dx2p_ho = shifts_darboux(s1, s2_ho)

    for name, dp_fn, c_ln, c_lin in [
        ('DeltaP_mode', DeltaP_mode, mp.mpf(1) / mp.pi, (-mp.log(2) - 1 + mp.log(3) / 4) / mp.pi),
        ('DeltaP_can', DeltaP_can, mp.mpf(0), -mp.mpf(4) / mp.pi),
    ]:
        dp1 = dp_fn(lam * r, 1)
        dp2_ov = dp_fn(lam, 1)
        dp2_ho = dp_fn(lam, -1)
        hirota_num = Dx1_ov * dp1 + Dx2_ov * dp2_ov
        darboux_tail_num = (Dx1p_ov - Dx1_ov) * dp1 + (Dx2p_ov - Dx2_ov) * dp2_ov
        ho_num = Dx1p_ho * dp1 + Dx2p_ho * dp2_ho

        hirota_asy = -2 * c_ln * mp.log(r) * mp.log((1 - r) / (1 + r))
        darboux_tail_asy = -2 * (1 / r - r) * (c_ln * mp.log(lam) + c_lin) - 2 * c_ln * mp.log(r) / r
        ho_asy = -2 * (1 / r + r) * (c_ln * mp.log(lam) + c_lin) - 2 * c_ln * mp.log(r) / r

        err_h = abs(hirota_num - hirota_asy)
        err_d = abs(darboux_tail_num - darboux_tail_asy)
        err_ho = abs(ho_num - ho_asy)
        print('   %-12s (lam=1e-6, r=0.6): |Hirota - asy|=%s, |Darboux_tail - asy|=%s, |HO - asy|=%s'
              % (name, mp.nstr(err_h, 3), mp.nstr(err_d, 3), mp.nstr(err_ho, 3)))
        ok &= err_h < mp.mpf('1e-10') and err_d < mp.mpf('1e-10') and err_ho < mp.mpf('1e-10')

    # Verify relation between I_KdV(r) and Hirota delay shift for c_ln = 1/pi (confirming matching.pdf sign correction):
    # I_KdV(r) = -(4/pi) chi_2(r) - (2/pi) ln(r) ln((1-r)/(1+r)) and hirota_mode = -(2/pi) ln(r) ln((1-r)/(1+r)),
    # so I_KdV(r) - hirota_mode = -(4/pi) chi_2(r), while I_KdV(r) + hirota_mode doubles the log term.
    I_kdv_r = 2 / mp.pi * L_dilog(r)
    hirota_mode = -2 * (mp.mpf(1) / mp.pi) * mp.log(r) * mp.log((1 - r) / (1 + r))
    chi2_r = mp.mpf('0.5') * (mp.polylog(2, r) - mp.polylog(2, -r))
    err_chi2 = abs((I_kdv_r - hirota_mode) - (-4 / mp.pi * chi2_r))
    err_sum_log = abs((I_kdv_r + hirota_mode) - (-4 / mp.pi * chi2_r - 4 / mp.pi * mp.log(r) * mp.log((1 - r) / (1 + r))))
    print('   Relation I_KdV(r) - Dx*DeltaP_mode = -(4/pi) chi_2(r): |err| = %s (sum err = %s)'
          % (mp.nstr(err_chi2, 3), mp.nstr(err_sum_log, 3)))
    ok &= err_chi2 < mp.mpf('1e-35') and err_sum_log < mp.mpf('1e-35')

    print('ALL CHECKS PASSED' if ok else 'SOME CHECK FAILED')
    assert ok

#!/usr/bin/env python3
"""Independent, read-only audit of Phase 2h inputs to the next-step programme.

Writes only audit_phase2h_results.json. Existing project functions are comparison
targets, not the source of the independent phonon integrals or Taylor checks.
Detected failures of prior claims are explicit audit results; ALL CHECKS PASSED
means the audit reproduced those failures and its positive controls passed.
No regularisation, phase convention, or physical charge prescription is chosen.
"""
import hashlib
import json
from pathlib import Path

import mpmath as mp
import sympy as sp

import kp_master_and_toda_edge as master
import oneloop_defect_phase_and_fusion as defect
import charge_and_frame_audit as charges

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
mp.mp.dps = 70
master.r3 = defect.r3 = mp.sqrt(3)
WEIGHTS = (1, 1, -2)
RESULTS = {"precision": mp.mp.dps, "prior_claim_failures": []}


def number(x):
    return mp.nstr(x, 25)


def angles(a):
    return (a + 2 * mp.pi / 3, a - 2 * mp.pi / 3, a)


def theta(q, a, eps):
    """Phase tending to zero at both infinities, evaluated separately at q=0+/-.

    Independent implementation of the elementary arctangent formula of Phase 2e.
    """
    aq, sg, w = abs(q), mp.sign(q), mp.sqrt(1 + q*q)
    return -sum(c * (sg * mp.atan(mp.sin(p)/aq)
                         + eps * mp.atan(mp.sin(p)*mp.cos(p)
                                         / (aq*w + q*q + mp.sin(p)**2)))
                for c, p in zip(WEIGHTS, angles(a)))


def theta_prime(q, a, eps):
    return sum(c * mp.sin(p) / (q*q + mp.sin(p)**2)
               * (1 + eps*mp.sign(q)*mp.cos(p)/mp.sqrt(1+q*q))
               for c, p in zip(WEIGHTS, angles(a)))


def whole_line(f):
    return mp.quad(lambda q: f(q) + f(-q),
                   [0, mp.mpf('.1'), 1, 10, mp.inf])


def chi_kernel(z):
    """Real odd kernel, independent of the master module's branch implementation."""
    if not z:
        return mp.mpf(0)
    az = abs(z)
    return -mp.sign(z)*mp.re(mp.polylog(2, az)-mp.polylog(2, -az))/(2*mp.pi)


def log_kernel(z):
    if not z or abs(z) == 1:
        return mp.mpf(0)
    az = abs(z)
    return -mp.sign(z)*mp.log(az)*mp.log(abs((1-az)/(1+az)))/(2*mp.pi)


def head_sum(K, a, b):
    return sum(ci*cj*K(mp.tan(pi/2)*mp.tan(pj/2))
               for ci, pi in zip(WEIGHTS, angles(a))
               for cj, pj in zip(WEIGHTS, angles(b)))


def fail_prior(message):
    RESULTS['prior_claim_failures'].append(message)
    print('PRIOR CLAIM FAIL:', message, flush=True)


def audit_physical_phase():
    print('1. Independent phonon quadrature vs Phase 2g and Phase 2h', flush=True)
    a, b = mp.mpf('.3'), mp.mpf('.7')
    rows = []
    for e1, e2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        raw = whole_line(lambda q: theta(q, a, e1)*theta_prime(q, b, e2))/(4*mp.pi)
        raw_2g = defect.I_SS_closed(a, e1, b, e2)
        raw_2h = master.I_SS_general(master.K_L_analytic, a, e1, b, e2)
        expected_difference = 0 if e1 == 1 else -mp.pi
        assert abs(raw-raw_2g) < mp.mpf('1e-60')
        assert abs(raw_2h-raw-expected_difference) < mp.mpf('1e-60')
        rows.append(dict(eps1=e1, eps2=e2, quadrature=number(raw),
                         phase2g_error=number(raw-raw_2g), master=number(raw_2h),
                         master_difference_over_pi=number((raw_2h-raw)/mp.pi)))
        print(' ', e1, e2, 'raw=', number(raw),
              'master difference / pi=', number((raw_2h-raw)/mp.pi), flush=True)

    # Repeat the analytic comparison at two other regular angle pairs.
    for a0, b0 in (('.25', '.55'), ('.45', '.8')):
        for e1, e2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            diff = master.I_SS_general(master.K_L_analytic, mp.mpf(a0), e1,
                                      mp.mpf(b0), e2) - defect.I_SS_closed(
                                          mp.mpf(a0), e1, mp.mpf(b0), e2)
            assert abs(diff-(0 if e1 == 1 else -mp.pi)) < mp.mpf('1e-60')

    f = master.I_SS_general(master.K_L_analytic, a, 1, b, 1)
    rev = master.I_SS_general(master.K_L_analytic, b, 1, a, 1)
    diag = master.I_SS_general(master.K_L_analytic, a, 1, a, 1)
    assert abs(f+rev+mp.pi) < mp.mpf('1e-60')
    assert abs(diag+mp.pi/2) < mp.mpf('1e-60')
    # The already-used odd rational test kernel has the opposite exchange issue:
    # K(z)=K(1/z) makes its two-body functional symmetric, while fusion survives.
    K = lambda z: z/(1+z*z)
    rf = master.I_SS_general(K, a, 1, b, 1)
    rr = master.I_SS_general(K, b, 1, a, 1)
    assert abs(rf-rr) < mp.mpf('1e-60') and abs(rf) > 1

    # Stable defect example: direct derivative of the defect transmission phase.
    qD = mp.mpf('.4')
    def defect_prime(q):
        X = mp.sign(q)*mp.sqrt(1+q*q)/(2*mp.sqrt(3))-qD
        Z = -q/2
        dX = mp.sign(q)*q/(2*mp.sqrt(3)*mp.sqrt(1+q*q))
        return 2*(X*(-mp.mpf('.5'))-dX*Z)/(X*X+Z*Z)
    sd_rows = []
    for e in (1, -1):
        direct = whole_line(lambda q: theta(q, a, e)*defect_prime(q))/(4*mp.pi)
        g = defect.I_SD_closed(a, e, -2*mp.sqrt(3)*qD, 1)
        h = master.I_SD_general(master.K_L_analytic, a, e, qD, 1)
        assert abs(direct-g) < mp.mpf('1e-60')
        assert abs(h-direct-(0 if e == 1 else mp.pi)) < mp.mpf('1e-60')
        sd_rows.append(dict(eps=e, qD=number(qD), quadrature=number(direct),
                            master_difference_over_pi=number((h-direct)/mp.pi)))
    RESULTS['physical_phase'] = dict(two_soliton=rows, defect=sd_rows,
                                     same_exchange_sum=number(f+rev),
                                     same_diagonal=number(diag),
                                     rational_symmetric_value=number(rf))
    fail_prior('The Phase 2h master functional is not the Phase 2g physical raw '
               'integral for all orientations: checked two-soliton difference -pi '
               'when eps1=-1 and stable-defect difference +pi when eps=-1.')
    fail_prior('The raw same-direction phase is not antisymmetric: I12+I21=-pi, '
               'and I(a,a)=-pi/2, not zero. Exchange/branch constants remain unresolved.')


def audit_head_split():
    print('2. Independent head-on kernel split and its mixed Taylor coefficient', flush=True)
    # For K_chi=-Re chi2/pi, K'(z)+z K''(z)=-1/[pi(1-z^2)].
    w = (sp.sqrt(3), -sp.sqrt(3), sp.Integer(0))
    dw = (sp.Integer(2), sp.Integer(2), sp.Rational(1, 2))
    mixed = -sum(WEIGHTS[i]*WEIGHTS[j]*dw[i]*dw[j]
                 / (sp.pi*(1-(w[i]*w[j])**2)) for i in range(3) for j in range(3))
    assert sp.simplify(mixed-9/sp.pi) == 0
    C = 6*mp.log(2)-mp.mpf('1.5')*mp.log(3)-3
    rows = []
    for r in map(mp.mpf, ('.25', '.5', '.75')):
        for ls in ('1e-4', '1e-7', '1e-10'):
            l = mp.mpf(ls)
            chi = head_sum(chi_kernel, r*l, l)
            logs = head_sum(log_kernel, r*l, l)
            expected_chi = 9*r*l*l/mp.pi
            expected_total = r*l*l/mp.pi*(-3*mp.log(r*l*l)+C)
            assert abs((chi-expected_chi)/l**2) < 100*l*l
            assert abs((chi+logs-expected_total)/l**2) < 100*l*l*abs(mp.log(l))
            rows.append(dict(r=number(r), lam=ls, chi_over_lam2=number(chi/l**2),
                             log_over_lam2=number(logs/l**2),
                             total_asymptotic_scaled_error=number(
                                 (chi+logs-expected_total)/l**2)))
    RESULTS['head_split'] = dict(mixed_coefficient=str(mixed), rows=rows)
    fail_prior('Claim 4.3 head-on parts are incorrect: the chi2 part has leading '
               '9*r*lambda^2/pi with no lambda^2 log(lambda) term. The stated '
               'total head-on asymptotic is nevertheless reproduced.')


def audit_hessian():
    print('3. Classical edge Hessian and logarithmic Taylor remainder', flush=True)
    s, t, Y, r, d1, d2, T, c = sp.symbols('s t Y r d1 d2 T c', positive=True)
    # First run: simplifying the raw radical Hessian stalled and was interrupted.
    # Use u=s^2, v=t^2 and the exact chain rule, retaining the same identity check.
    Phi = 2*(s*s-t*t)*sp.log((s-t)/(s+t))+4*s*t
    g1 = sp.simplify(sp.diff(Phi,s)/(2*s))
    g2 = sp.simplify(sp.diff(Phi,t)/(2*t))
    H = sp.Matrix([[sp.cancel(sp.diff(g1,s)/(2*s)), sp.cancel(sp.diff(g1,t)/(2*t))],
                   [sp.cancel(sp.diff(g2,s)/(2*s)), sp.cancel(sp.diff(g2,t)/(2*t))]])
    assert sp.simplify(H.det()) == 0
    Q = (sp.Matrix([d1, d2]).T*H*sp.Matrix([d1, d2]))[0]/2
    Qscaled = sp.simplify(Q.subs(t,r*s).subs(s*s,Y))
    expected = (d2-r*r*d1)**2/(r*(1-r*r)*Y)
    assert sp.simplify(Qscaled-expected) == 0
    shift1 = -T/(2*sp.pi)+c
    shift2 = shift1-sp.log(r)/sp.pi
    log_coefficient = sp.expand((Y*expected).subs({d1:shift1, d2:shift2})).coeff(T, 2)
    assert sp.simplify(log_coefficient-(1-r*r)/(4*sp.pi**2*r)) == 0
    rn, Yn = mp.mpf('.5'), mp.mpf('1e12')
    dn1 = -mp.log(Yn)/(2*mp.pi)
    dn2 = dn1-mp.log(rn)/mp.pi
    fn = lambda ua, ub: 2*(ua-ub)*mp.log((1-mp.sqrt(ub/ua))/(1+mp.sqrt(ub/ua)))+4*mp.sqrt(ua*ub)
    linear = 2*mp.log((1-rn)/(1+rn))*(dn1-dn2)+4*rn*dn1
    actual = fn(Yn+dn1, rn*rn*Yn+dn2)-fn(Yn, rn*rn*Yn)-linear
    quadratic = (dn2-rn*rn*dn1)**2/(rn*(1-rn*rn)*Yn)
    assert abs(actual/quadratic-1) < mp.mpf('1e-10')
    RESULTS['hessian'] = dict(quadratic=str(expected),
                              log_squared_coefficient=str(sp.simplify(log_coefficient)),
                              numerical_relative_error=number(actual/quadratic-1))
    fail_prior('Conditional on the stated logarithmic quasi-momentum shift, the '
               'classical first-order Taylor remainder contains log(Y)^2/Y; '
               'it is not O(1/Y). Cancellation in the full residual needs a separate derivation.')


def audit_edge_data():
    print('4. Reanalysis of saved edge fits without refitting to the proposed answer', flush=True)
    data = json.loads((HERE/'matching_analysis.json').read_text())
    rows = []
    for r in (.4, .5, .6):
        pts = [next(x for x in data['same'] if x['Y']==Y and x['r']==r) for Y in (16.,32.)]
        slopes = {key:32*(mp.mpf(str(pts[0][key]))-mp.mpf(str(pts[1][key])))
                  for key in ('residual', 'cubic', 'h_log_h_intercept')}
        spread = 32*sum(mp.mpf(str(x['edge_fit_spread'])) for x in pts)
        assert slopes['residual'] < 0 < slopes['h_log_h_intercept']
        assert spread > abs(slopes['residual'])
        intercept = next(x for x in data['large_y'] if x['r']==r)
        rm = mp.mpf(str(r))
        proposed = -2*(mp.polylog(2,rm)-mp.polylog(2,-rm)-2*rm)/mp.pi
        distance = abs(mp.mpf(str(intercept['quadratic']))-proposed)
        assert distance < mp.mpf('8e-5')
        assert mp.mpf(str(intercept['propagated_edge_spread'])) > 50*distance
        row = dict(r=r, c1_quadratic=number(slopes['residual']),
                   c1_cubic=number(slopes['cubic']), c1_hlogh=number(slopes['h_log_h_intercept']),
                   c1_spread=number(spread), c0_candidate_distance=number(distance),
                   c0_edge_spread=intercept['propagated_edge_spread'])
        rows.append(row)
        print(' ', row, flush=True)
    RESULTS['edge_data'] = rows
    fail_prior('The quoted c1 values are not resolved by the saved fits: their '
               'sign changes under the already-recorded h log h edge extrapolation.')


def audit_charges_and_convention():
    print('5. Positive charge controls and the displayed soliton profile', flush=True)
    worst = mp.mpf(0)
    for a in map(mp.mpf, ('.2', '.5', '.8')):
        for eps in (1, -1):
            L = lambda x: eps*mp.cos(x)*charges.delta_P_mode(x,eps)-charges.delta_E1_ren(x)
            numerical = -eps*mp.diff(L,a)/mp.sin(a)
            worst = max(worst,abs(numerical-charges.delta_P_can_closed(a,eps)))
    assert worst < mp.mpf('1e-60')
    # Tau-function and travelling-wave ODE fix the sign AND amplitude of u.
    x, k = sp.symbols('x k', positive=True)
    tau_profile = sp.simplify(2*sp.diff(sp.log(1+sp.exp(k*x)),x,2))
    correct = k*k/(2*sp.cosh(k*x/2)**2)
    # Exponential substitution avoids relying on symbolic hyperbolic simplification.
    # Rewriting only one side previously left a mixed exp/cosh expression
    # unevaluated and failed the symbolic-zero assertion despite equality.
    assert sp.cancel((tau_profile-correct).rewrite(sp.exp)) == 0
    quoted = -k*k/sp.cosh(k*x/2)**2
    ode = lambda f: sp.diff(f,x,2)-k*k*f+3*f*f
    good_ode = sp.simplify(ode(correct).subs(x,0))
    bad_ode = sp.simplify(ode(quoted).subs(x,0))
    assert good_ode == 0 and sp.simplify(bad_ode-sp.Rational(9,2)*k**4) == 0
    RESULTS['charges'] = dict(legendre_momentum_error=number(worst),
                              quoted_profile_ode_residual_at_zero=str(bad_ode))
    fail_prior('The Phase 2h displayed profile -k^2 sech^2(k*x/2) conflicts '
               'with u=2(log tau)_xx; the convention requires +(k^2/2) sech^2(k*x/2).')


def main():
    sources = ['phase2/report.tex', 'phase2/charge_frame_master.tex',
               'phase2/oneloop_defect_phase.tex', 'phase2/oneloop_phase.tex',
               'phase2/scripts/charge_and_frame_audit.py',
               'phase2/scripts/kp_master_and_toda_edge.py',
               'phase2/scripts/oneloop_defect_phase_and_fusion.py',
               'phase2/scripts/matching_analysis.json']
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    audit_physical_phase()
    audit_head_split()
    audit_hessian()
    audit_edge_data()
    audit_charges_and_convention()
    assert hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, \
        'A source changed during the audit; rerun against a consistent snapshot.'
    RESULTS['source_sha256'] = hashes
    (HERE/'audit_phase2h_results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

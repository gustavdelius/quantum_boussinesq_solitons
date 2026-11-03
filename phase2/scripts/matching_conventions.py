"""High-precision checks of the proposed Toda/good-Boussinesq dictionary.

All new mathematical statements in matching.tex are checked here, except the
Toda numerical extrapolations (matching_analysis.py). No quantum equivalence
between the two continuum equations is assumed.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp
import oneloop_good_phonon_scattering as phonon
from oneloop_good_twosoliton_phase import (
    sol_good, A12_good, delta0_good, Delta_P1_closed, delta_renorm_good,
    I_same_closed, I_head_closed, L_chi2,
)
import smatrix_ansatz as bad


def asymptotic(r, lam, head=False):
    c = -mp.log(2)-1+mp.log(3)/4
    logA = 2*mp.log((1-r)/(1+r))
    if head:
        return -2/mp.pi*((r+1/r)*(mp.log(lam)+c)+mp.log(r)/r)
    phonon_part = 2/mp.pi*L_chi2(r)
    hirota = -logA*mp.log(r)/mp.pi
    topological = 2/mp.pi*((r-1/r)*(mp.log(lam)+c)-mp.log(r)/r)
    return phonon_part+hirota+topological


def full(r, lam, head=False):
    a1, a2 = r*lam, lam
    return ((I_head_closed(a1, a2) if head else I_same_closed(a1, a2))
            + delta_renorm_good(a1, 1, a2, -1 if head else 1))


def main():
    mp.mp.dps = 60
    # The imported soliton module caches sqrt(3) at import precision.
    # Recompute that constant explicitly for a genuine 60-digit check.
    phonon.r3 = mp.sqrt(3)
    X, T, lam = sp.symbols('X T lam', real=True)
    U = sp.Function('U')(X, T)
    dx = lambda f: lam*sp.diff(f, X)
    dt = lambda f: -lam*sp.diff(f, X)+lam**3*sp.diff(f, T)
    u = lam**2*U
    for sign in (1, -1):  # +1 good, -1 bad
        residual = dt(dt(u))-dx(dx(u))+sign*(dx(dx(dx(dx(u))))+3*dx(dx(u*u)))
        leading = -2*sp.diff(U, X, T)+sign*(sp.diff(U, X, 4)+3*sp.diff(U*U, X, 2))
        assert sp.expand(residual-lam**6*leading-lam**8*sp.diff(U, T, 2)) == 0
    print('KdV reductions: good and bad have opposite slow-time evolution.')

    # Leading canonical coordinate/charge dictionary for Toda and bad Boussinesq.
    b = mp.exp(-mp.mpf(1)/2)
    be = mp.mpf('1e-8'); kap = mp.sqrt(3)*be
    pn = 4*b*(kap*mp.cosh(kap)-mp.sinh(kap))
    assert abs(pn/(6*mp.sqrt(3)*b*bad.sol(be)['P'])-1) < 2*be*be
    assert abs(mp.sinh(kap)/kap-mp.cosh(be)) < be**4

    rows = []
    worst_derivative = mp.mpf(0)
    for rstr in ('0.25', '0.5', '0.75'):
        r = mp.mpf(rstr)
        # Verify the classical dictionary with COLLIDING order in each theory.
        l = mp.mpf('1e-8')
        logA = 2*mp.log((1-r)/(1+r))
        dg = delta0_good(r*l, 1, l, 1)
        db = bad.delta(bad.sol(l), bad.sol(r*l))
        assert abs(dg/l**2-(r*r-1)*logA) < mp.mpf('1e-12')
        assert abs(db/l**2-(1-r*r)*logA) < mp.mpf('1e-12')
        assert abs((dg+db)/l**2) < mp.mpf('1e-12')
        for head in (False, True):
            errors = []
            for ls in ('1e-3', '1e-6', '1e-9'):
                l = mp.mpf(ls)
                val = full(r, l, head)
                prediction = asymptotic(r, l, head)
                error = abs(val-prediction)
                errors.append(error)
                rows.append(dict(r=rstr, lam=ls, channel='head' if head else 'same',
                                 phase=float(val), asymptotic=float(prediction), error=float(error)))
                assert error < 100*l*l*abs(mp.log(l))
            assert errors[2] < errors[1]/100000
            # Logarithmic slope and its ratio derivative exclude a constant-phase explanation.
            l = mp.mpf('1e-9')
            slope = mp.diff(lambda z: full(r, mp.exp(z), head), mp.log(l))
            expected = -2/mp.pi*(r+1/r) if head else 2/mp.pi*(r-1/r)
            assert abs(slope-expected) < mp.mpf('1e-13')
            derivative = mp.diff(lambda rr: full(rr, l, head), r)
            predicted_derivative = mp.diff(lambda rr: asymptotic(rr, l, head), r)
            assert abs(derivative-predicted_derivative) < mp.mpf('1e-12')
            # Independent numerical derivative of the classical phase, rather than gamma formula.
            a1, a2, e2 = r/3, mp.mpf(1)/3, -1 if head else 1
            d1 = mp.diff(lambda a: delta0_good(a, 1, a2, e2), a1)/sol_good(a1, 1)['dP_da']
            d2 = mp.diff(lambda a: delta0_good(a1, 1, a, e2), a2)/sol_good(a2, e2)['dP_da']
            direct = -d1*Delta_P1_closed(a1, 1)-d2*Delta_P1_closed(a2, e2)
            err = abs(direct-delta_renorm_good(a1, 1, a2, e2))
            worst_derivative = max(worst_derivative, err)
            assert err < mp.mpf('1e-50')
        # Head-on classical phases agree at order lambda^2, despite overtaking reversal.
        l = mp.mpf('1e-8')
        for d in (delta0_good(r*l, 1, l, -1), bad.delta(bad.sol(r*l), bad.sol(l, -1))):
            assert abs(d/l**2-4*r) < mp.mpf('1e-12')

    # Explicitly test the cancellation claimed in oneloop_phase.tex, eq. following Delta P.
    r = mp.mpf('.5'); l = mp.mpf('1e-9')
    s1, s2 = sol_good(r*l, 1), sol_good(l, 1)
    hirota = -mp.log(A12_good(s1, s2))*(Delta_P1_closed(r*l)/s1['k']-Delta_P1_closed(l)/s2['k'])
    chi = (mp.polylog(2, r)-mp.polylog(2, -r))/2
    actual = I_same_closed(r*l, l)+hirota
    claimed = -4*chi/mp.pi
    corrected = claimed-4/mp.pi*mp.log(r)*mp.log((1-r)/(1+r))
    assert abs(actual-corrected) < mp.mpf('1e-14')
    assert abs(actual-claimed) > mp.mpf('.9')
    print('Recorded failed prior cancellation: I + Hirota =', mp.nstr(actual, 14),
          '; claimed pure chi =', mp.nstr(claimed, 14))

    # Nonuniformity of the physical-momentum expansion at fixed edge variable y.
    c = -mp.log(2)-1+mp.log(3)/4
    a = mp.mpf('1e-9'); y = mp.mpf(16); h = a*a/y
    relative = h*Delta_P1_closed(a)/sol_good(a, 1)['P']
    predicted = 3/(2*mp.pi*y)*(mp.log(a)+c)
    assert abs(relative-predicted) < mp.mpf('1e-15')
    result = dict(precision=60, rows=rows, derivative_error=float(worst_derivative),
                  cancellation_actual=float(actual), cancellation_claimed=float(claimed),
                  relative_momentum_shift=float(relative))
    Path(__file__).with_name('matching_conventions.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Classical dictionary, full small-amplitude limits, derivatives and nonuniformity verified.')
    print('Worst independent phase derivative error:', mp.nstr(worst_derivative, 4))
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

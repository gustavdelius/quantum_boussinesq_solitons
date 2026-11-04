"""Overtaking in the double-scaling corner with the two-term normalisation justified by the classical relation
Phi(1|2) = delta_T + M1 (Pn2/a - Pi2) + f(kappa1)  (verified for kappa1 > kappa2, toda_ba_vs_label.py):
    Psi2_T = Phi(1|2) - Phi(1|e2) - M1 (Pn2/a - Pi2)       (removes f(kappa1) only)
and the candidate normalised the same way, arg S(1,2) - arg S(1,e2).
Test: D(y1,y2) = Psi2_T/hbar_T - candidate is a function of y2 alone (a single-particle convention for
soliton 2), i.e. the two-body structure of the candidate is exact in the corner.  Richardson in hbar_T."""
import json, numpy as np, mpmath as mp
from toda_edge_ba import EdgeBA, Pn, Pi
import smatrix_ansatz as SA

def toda2(m, y1, s1, y2, s2):
    a, b = m.a, m.b; hB = m.h/(3*b)
    be1, be2 = np.sqrt(y1*hB), np.sqrt(y2*hB); k1, k2 = np.sqrt(3)*be1, np.sqrt(3)*be2
    p1, p2 = m.p_of_P(Pn(k1, s1, b)/a, s1), m.p_of_P(Pn(k2, s2, b)/a, s2)
    e2 = m.edge(s2)
    val = m.Phi(p1, p2) - m.Phi(p1, e2) - (-2*k1)*(Pn(k2, s2, b)/a - Pi(k2, s2, b))
    return val/m.h, be1, be2, hB

def cand2(be1, s1, be2, s2, hB):
    S = lambda x1, x2: mp.im(SA.logS(SA.sol(x1, s1), SA.sol(x2, s2), hB))
    return float(S(be1, be2) - S(be1, mp.mpf('1e-12')))

def semi(be1, s1, be2, s2, hB):
    return float(SA.delta(SA.sol(be1, s1), SA.sol(be2, s2))/hB)

if __name__ == '__main__':
    pts = [(y1, y2) for y2 in (0.5, 1.0) for y1 in (1.5, 3.0, 6.0)] + [(0.3, 0.1), (3.0, 0.1)]
    hs = [0.01, 0.005, 0.0025]
    out = {}
    for h in hs:
        m = EdgeBA(1.0, h)
        for (y1, y2) in pts:
            t, b1, b2, hB = toda2(m, y1, 1, y2, 1)
            c, s = cand2(b1, 1, b2, 1, hB), semi(b1, 1, b2, 1, hB)
            out.setdefault('%g,%g' % (y1, y2), []).append((h, t, c, s))
            print('h=%.4f N=%d y=(%g,%g): Toda %+.6f cand %+.6f semi %+.6f  D=T-c %+.6f' % (h, m.N, y1, y2, t, c, s, t - c), flush=True)
        json.dump(out, open('toda_edge_twoterm.json', 'w'), indent=1)
    print('\nRichardson (quadratic fit in hbar through 3 points) of D = Toda - cand, and of Toda - semi:')
    for key, rows in out.items():
        H = np.array([r[0] for r in rows]); Dv = np.array([r[1] - r[2] for r in rows]); Sv = np.array([r[1] - r[3] for r in rows])
        D0 = np.polyfit(H, Dv, 2)[-1]; S0 = np.polyfit(H, Sv, 2)[-1]
        print('  y=(%s): D(0) = %+.5f   (Toda-semi)(0) = %+.5f' % (key, D0, S0))

    # (V), (c): toda_edge.tex, results table (four decimal places). A 1e-4
    # absolute bound allows its rounding and the printed residual subtraction.
    _check_table = {
        '1.5,0.5': (-2.7009, -2.6949, -2.6339, -0.0060),
        '3,0.5': (-4.3477, -4.3463, -4.3351, -0.0014),
        '6,0.5': (-6.5400, -6.5392, -6.5366, -0.0008),
        '1.5,1': (-2.6215, -2.5451, -2.2924, -0.0764),
        '3,1': (-5.3134, -5.2962, -5.2678, -0.0172),
        '6,1': (-8.6817, -8.6757, -8.6701, -0.0059),
        '0.3,0.1': (-0.6129, -0.7044, -0.5268, 0.0914),
        '3,0.1': (-2.1425, -2.1438, -2.1419, 0.0013),
    }
    _check_residuals = {}
    for _check_key, _check_expected in _check_table.items():
        _check_rows = np.array(out[_check_key])
        _check_ext = np.polyfit(_check_rows[:, 0], _check_rows[:, 1:], 2)[-1]
        _check_D = _check_ext[0] - _check_ext[1]
        assert np.all(np.isfinite(_check_ext)), (_check_key, _check_ext)
        assert np.max(np.abs(np.append(_check_ext, _check_D) - _check_expected)) < 1e-4, (_check_key, _check_ext, _check_D)
        _check_residuals[_check_key] = _check_D
    # At fixed y2 the residual depends on y1 by far more than table rounding.
    assert abs(_check_residuals['1.5,1'] - _check_residuals['3,1']) > 0.01
    assert _check_residuals['1.5,1'] < -0.01 < 0 < _check_residuals['0.3,0.1']
    # Preserve the write-up's explicit bound, even if it fails. Report every
    # offending Toda step, after computing and writing ALL original outputs.
    _check_steps = [(key, rows[i-1][0], rows[i][0], abs(rows[i][1] - rows[i-1][1]))
                    for key, rows in out.items() for i in range(1, len(rows))]
    assert all(step[3] < 4e-3 for step in _check_steps), (
        "toda_edge.tex: successive Toda changes < 0.004 FAILED",
        [step for step in _check_steps if not step[3] < 4e-3])
    print("ALL CHECKS PASSED")

"""Double-scaling scan: exact (quantum Toda Bethe ansatz) normalised soliton phase vs the Phase-2c candidate
and vs the semiclassical phase, at fixed y = beta^2/hbar_B as hbar_T -> 0.  Results written to
toda_edge_scan.json (committed early for the parallel one-loop session)."""
import json, time
import numpy as np
import mpmath as mp
from toda_edge_ba import EdgeBA, Psi_T
import smatrix_ansatz as SA

def cand_norm(be1, s1, be2, s2, hB):
    t1, t2 = mp.mpf('1e-12'), mp.mpf('0.5e-12')     # distinct, so that l1 - l2 != 0 at the double edge
    S = lambda x1, x2: mp.im(SA.logS(SA.sol(x1, s1), SA.sol(x2, s2), hB))
    return float(S(be1, be2) - S(be1, t2) - S(t1, be2) + S(t1, t2))

def semi_norm(be1, s1, be2, s2, hB):
    return float(SA.delta(SA.sol(be1, s1), SA.sol(be2, s2))/hB)

if __name__ == '__main__':
    pairs = [('ov', 0.8, 0.5), ('ov', 2.0, 0.5), ('ov', 4.0, 0.5), ('ov', 8.0, 1.0),
             ('ho', 0.5, 0.5), ('ho', 1.0, 1.0), ('ho', 2.0, 0.5)]
    hs = [0.04, 0.02, 0.01, 0.005]
    out = {}
    for h in hs:
        t = time.time(); m = EdgeBA(1.0, h)
        for ch, y1, y2 in pairs:
            s2 = 1 if ch == 'ov' else -1
            psi, (be1, be2, hB) = Psi_T(m, y1, 1, y2, s2)
            c = cand_norm(be1, 1, be2, s2, hB); sc = semi_norm(be1, 1, be2, s2, hB)
            out.setdefault('%s %.1f %.1f' % (ch, y1, y2), []).append(dict(hT=h, beta1=be1, beta2=be2, toda=psi, cand=c, semi=sc))
            print('hT=%.3f N=%d %s y=(%.1f,%.1f) beta=(%.4f,%.4f): Toda %+.6f  cand %+.6f  semi %+.6f'
                  % (h, m.N, ch, y1, y2, be1, be2, psi, c, sc), flush=True)
        print('   (%.0fs)' % (time.time() - t), flush=True)
        json.dump(out, open('toda_edge_scan.json', 'w'), indent=1)
    # Richardson (linear in hbar) from the two smallest hbar
    print('\nextrapolated hbar_T -> 0 (Richardson, last two points):')
    for key, rows in out.items():
        r = rows[-2:]
        ext = lambda f: 2*r[1][f] - r[0][f]
        print('  %-12s Toda %+.5f   cand %+.5f   semi %+.5f   Toda-cand %+.5f   Toda-semi %+.5f'
              % (key, ext('toda'), ext('cand'), ext('semi'), ext('toda') - ext('cand'), ext('toda') - ext('semi')))

    # (V), (c): toda_edge.tex, head-on result, explicitly within 1e-4.
    # The overtaking four-term comparison is superseded in that write-up;
    # do not assert agreement for its exploratory overtaking output.
    for _check_y1, _check_y2 in ((0.5, 0.5), (1.0, 1.0), (2.0, 0.5)):
        _check_rows = out['ho %.1f %.1f' % (_check_y1, _check_y2)]
        _check_expected = 4*np.sqrt(_check_y1*_check_y2)
        for _check_field in ('toda', 'cand', 'semi'):
            _check_ext = 2*_check_rows[-1][_check_field] - _check_rows[-2][_check_field]
            assert np.isfinite(_check_ext)
            assert abs(_check_ext - _check_expected) < 1e-4, (
                _check_y1, _check_y2, _check_field, _check_ext, _check_expected)
    print("ALL CHECKS PASSED")

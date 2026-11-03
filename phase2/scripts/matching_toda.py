"""Reproducible Toda overlap scan; writes only matching_* artifacts.

Uses the existing edge solver and its classical label-frame conversion, without
claiming that conversion is quantum exact. Run with OPENBLAS_NUM_THREADS=1.
The h extrapolation is performed before the large-y comparison.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre
from toda_edge_ba import EdgeBA, Psi_T
from toda_edge_twoterm import toda2

HERE = Path(__file__).resolve().parent


class FastNodesEdgeBA(EdgeBA):
    """Same solver, with Gauss nodes obtained without a dense companion matrix."""
    def __init__(self, a, hbar, npts):
        self.a, self.h, self.b = a, hbar, np.exp(-a/2)
        self.w = 2*self.b
        self.N = npts
        self.x, self.wx = roots_legendre(self.N)
        self._ground()


def classical_edge(y1, y2):
    r = np.sqrt(y2 / y1)
    return (y1 - y2) * 2 * np.log((1 - r) / (1 + r))


def points():
    # Include nearby ratios for a derivative test independent of phase constants.
    return [(Y, r) for Y in (4., 8., 16., 32.)
            for r in (.4, .5, .6)] + [(16., .49), (16., .51)]


def run(hs, factor, filename, fast_nodes=False):
    data = []
    for h in hs:
        start = time.monotonic()
        n = int(np.ceil(factor * (int(4 * np.pi * 2 * np.exp(-.5) / h) + 300)))
        print(f'Building ground state: h={h:g}, N={n}; dense solve may take several minutes.', flush=True)
        model = (FastNodesEdgeBA if fast_nodes else EdgeBA)(1., h, npts=n)
        print(f'Ground state ready after {time.monotonic()-start:.1f}s; evaluating scattering phases.', flush=True)
        count_error = abs(np.dot(model.wq, model.rho) - 1)
        assert count_error < 1e-9
        row = dict(h=h, n=n, q=float(model.q), count_error=float(count_error),
                   node_method='scipy' if fast_nodes else 'numpy', same=[], head=[])
        for Y, r in points():
            y2 = Y * r*r
            val, _, _, _ = toda2(model, Y, 1, y2, 1)
            row['same'].append(dict(Y=Y, r=r, phase=float(val),
                                    residual=float(val-classical_edge(Y, y2))))
        for y1, y2 in ((1., 1.), (4., 1.), (16., 4.)):
            val, _ = Psi_T(model, y1, 1, y2, -1)
            row['head'].append(dict(y1=y1, y2=y2, phase=float(val),
                                    residual=float(val-4*np.sqrt(y1*y2))))
        row['seconds'] = time.monotonic()-start
        assert all(np.isfinite(v['residual']) for channel in ('same', 'head') for v in row[channel])
        data.append(row)
        (HERE/filename).write_text(json.dumps(data, indent=2)+'\n')
        print(f'h={h:g} N={n} seconds={row["seconds"]:.1f}', flush=True)
        print('  same residuals:', [round(p['residual'], 7) for p in row['same']], flush=True)
        print('  head residuals:', [round(p['residual'], 7) for p in row['head']], flush=True)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hs', nargs='+', type=float, default=[.04, .02, .01, .005])
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--output', default='matching_toda_data.json')
    parser.add_argument('--fast-nodes', action='store_true')
    args = parser.parse_args()
    assert args.output.startswith('matching_') and '/' not in args.output
    run(args.hs, args.factor, args.output, args.fast_nodes)

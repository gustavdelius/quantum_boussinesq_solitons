"""Analyze independent mesh/edge/large-y limits and generate matching_tables.tex.

Reproduce inputs with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:
 python3 matching_toda.py
 python3 matching_toda.py --hs .0025 --output matching_toda_fine.json
 python3 matching_toda.py --hs .01 --fast-nodes --output matching_toda_fast.json
 python3 matching_toda.py --hs .01 --factor 1.4 --output matching_toda_nodes.json
Then run this script. Fit spreads are diagnostics, not rigorous error bounds.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def intercept(x, v, degree):
    # Scale x so conditioning does not depend on the small hbar units.
    return float(np.polynomial.polynomial.polyfit(np.asarray(x)/max(x), v, degree)[0])


def main():
    rows = json.loads((HERE/'matching_toda_data.json').read_text())
    fast = json.loads((HERE/'matching_toda_fast.json').read_text())
    assert fast[0]['h'] == .01
    rows += json.loads((HERE/'matching_toda_fine.json').read_text())
    rows.sort(key=lambda d: d['h'], reverse=True)
    assert [v['h'] for v in rows] == [.04, .02, .01, .005, .0025]
    node = json.loads((HERE/'matching_toda_nodes.json').read_text())[0]
    base = next(v for v in rows if v['h'] == node['h'])
    assert node['n'] > base['n']
    mesh = max(abs(x['phase']-y['phase']) for channel in ('same', 'head')
               for x, y in zip(node[channel], base[channel]))
    assert mesh < 1e-7
    method_error = max(abs(x['phase']-y['phase']) for channel in ('same', 'head')
                       for x, y in zip(fast[0][channel], base[channel]))
    assert method_error < 1e-7
    h = np.array([v['h'] for v in rows])
    result = dict(mesh_difference=mesh, mesh_h=node['h'], mesh_n=[base['n'], node['n']],
                  node_method_difference=method_error,
                  h_sequence=list(h), node_sequence=[v['n'] for v in rows], same=[], head=[])
    for channel in ('same', 'head'):
        for j, item in enumerate(rows[0][channel]):
            v = np.array([row[channel][j]['residual'] for row in rows])
            quadratic = intercept(h[-3:], v[-3:], 2)
            old_quadratic = intercept(h[-4:-1], v[-4:-1], 2)
            cubic = intercept(h[-4:], v[-4:], 3)
            linear = intercept(h[-2:], v[-2:], 1)
            z = h[-3:]/h[-3]
            hlogh = float(np.linalg.solve(np.column_stack([np.ones(3), z, z*np.log(z)]), v[-3:])[0])
            spread = max(abs(quadratic-old_quadratic), abs(quadratic-cubic), abs(quadratic-hlogh))
            out = {k: item[k] for k in (('Y','r') if channel == 'same' else ('y1','y2'))}
            out.update(residual=quadratic, edge_fit_spread=spread,
                       linear_quadratic_difference=abs(linear-quadratic), cubic=cubic,
                       h_log_h_intercept=hlogh)
            result[channel].append(out)
    # These bounds check resolution of the reported residual, not an assumed match.
    assert max(v['edge_fit_spread'] for v in result['same']) < .003
    assert max(abs(v['residual']) for v in result['head']) < .002
    assert all(abs(v['residual']) > 5*(v['edge_fit_spread']+mesh) for v in result['same'])

    def point(Y, r):
        return next(v for v in result['same'] if v['Y'] == Y and v['r'] == r)

    result['large_y'] = []
    for r in (.4, .5, .6):
        inv = 1/np.array([4., 8., 16., 32.])
        vals = [point(Y, r)['residual'] for Y in (4., 8., 16., 32.)]
        linear = intercept(inv[-2:], vals[-2:], 1)
        quadratic = intercept(inv[-3:], vals[-3:], 2)
        cubic = intercept(inv, vals, 3)
        # The last-three-point intercept has weights 1/3, -2, 8/3.
        edge_spread = sum(abs(w)*point(Y, r)['edge_fit_spread']
                          for w, Y in zip((1/3, -2, 8/3), (8., 16., 32.)))
        result['large_y'].append(dict(r=r, linear=linear, quadratic=quadratic, cubic=cubic,
                                     model_spread=max(abs(quadratic-linear), abs(quadratic-cubic)),
                                     propagated_edge_spread=edge_spread))
    derivative = (point(16., .51)['residual']-point(16., .49)['residual'])/.02
    derivative_coarse = (point(16., .6)['residual']-point(16., .4)['residual'])/.2
    derivative_edge_error = (point(16., .51)['edge_fit_spread']+point(16., .49)['edge_fit_spread'])/.02
    result['ratio_derivative'] = dict(Y=16, r=.5, step=.01, value=derivative,
                                      coarse_step=.1, coarse_value=derivative_coarse,
                                      edge_fit_spread=derivative_edge_error)
    assert abs(derivative) > 5*derivative_edge_error
    (HERE/'matching_analysis.json').write_text(json.dumps(result, indent=2)+'\n')

    text = [r'\paragraph{Resolution and order of limits.}',
            r'The edge sequence is $\hbar_T=0.04,0.02,0.01,0.005,0.0025$,',
            'with '+', '.join(str(v['n']) for v in rows)+r' quadrature nodes, respectively.',
            r'At $\hbar_T=0.01$, increasing the node count from '
            +f'{base["n"]} to {node["n"]} changes all sampled phases by at most '
            +fr'${mesh:.2g}$ radians. This mesh test is at one fixed $\hbar_T$;',
            r'it is not a convergence guarantee at every smaller $\hbar_T$.',
            r'The edge sequence uses the original NumPy Gauss--Legendre nodes.',
            r'At $\hbar_T=0.01$ and 1824 nodes, the NumPy/SciPy phase difference is '
            +f'${method_error:.2g}$ radians.',
            r'The central edge estimate is the quadratic intercept from the three',
            r'smallest $\hbar_T$. The stated fit spread is the largest change on using',
            r'the preceding quadratic window, the four-smallest-point cubic fit, or',
            r'a fit to $1,\hbar_T,\hbar_T\log\hbar_T$ at the three smallest points.',
            r'The last-two-point linear fit is stored separately in the JSON diagnostics.',
            r'\begin{center}\small',
            r'\begin{tabular}{@{}ccrr@{}}\toprule',
            r'$Y$ & $r$ & Edge residual $R_0$ & Edge fit spread\\\midrule']
    for Y in (4., 8., 16., 32.):
        for r in (.4, .5, .6):
            p = point(Y,r)
            text.append(f'{Y:g} & {r:g} & {p["residual"]:+.6f} & ${p["edge_fit_spread"]:.1e}$'+r'\\')
    text += [r'\bottomrule\end{tabular}\end{center}',
             r'\paragraph{Large-$Y$ diagnostic, not an exact limit.}',
             r'A quadratic fit in $1/Y$ to $Y=8,16,32$ gives the following exploratory',
             r'intercepts. The model spread compares a linear fit to the largest two',
             r'$Y$ values and a cubic fit to all four; edge-extrapolation uncertainty',
             r'is propagated with the absolute interpolation weights and shown separately.',
             r'\begin{center}\small\begin{tabular}{@{}crrr@{}}\toprule',
             r'$r$ & Exploratory $Y\to\infty$ intercept & Model spread & Edge spread\\\midrule']
    for p in result['large_y']:
        text.append(f'{p["r"]:g} & {p["quadratic"]:+.5f} & ${p["model_spread"]:.1e}$ & ${p["propagated_edge_spread"]:.1e}$'+r'\\')
    text += [r'\bottomrule\end{tabular}\end{center}',
             r'\paragraph{Derivative and head-on controls.}',
             f'At $Y=16,r=0.5$, the central ratio derivative is ${derivative:+.5f}$ '
             +r'with step $0.01$, versus '+f'${derivative_coarse:+.5f}$'+r' with step $0.1$.',
             f'The propagated edge-fit spread for the finer derivative is ${derivative_edge_error:.1e}$.',
             r'This supports nonconstant ratio dependence independently of an additive phase.',
             r'The head-on residuals after subtracting $4\sqrt{y_1y_2}$ are:',
             r'\begin{center}\small\begin{tabular}{@{}crr@{}}\toprule',
             r'$(y_1,y_2)$ & Edge residual & Edge fit spread\\\midrule']
    for p in result['head']:
        text.append(f'$({p["y1"]:g},{p["y2"]:g})$ & ${p["residual"]:+.2e}$ & ${p["edge_fit_spread"]:.1e}$'+r'\\')
    text += [r'\bottomrule\end{tabular}\end{center}',
             r'These controls support the previous numerical head-on result in the',
             r'Toda extraction convention. They do not establish the full good-equation match.']
    # Format scientific notation as LaTeX, not floating-point programming syntax.
    import re
    formatted = '\n'.join(text)+'\n'
    formatted = re.sub(r'([+-]?\d+(?:\.\d+)?)e([+-]?\d+)',
                       lambda m: m.group(1)+r'\times10^{'+str(int(m.group(2)))+'}', formatted)
    (HERE.parent/'matching_tables.tex').write_text(formatted)
    print(json.dumps(result, indent=2))
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

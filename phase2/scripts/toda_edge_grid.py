"""Overtaking grid at hbar_T = 0.01 and 0.005: exact Toda normalised phase vs candidate and semiclassics."""
import json, numpy as np
from toda_edge_ba import EdgeBA, Psi_T
from toda_edge_scan import cand_norm, semi_norm
res = {}
for h in [0.01]:
    m = EdgeBA(1.0, h)
    for y2 in [0.25, 0.5, 1.0, 2.0]:
        for y1 in [0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 8.0]:
            if abs(y1 - y2) < 1e-9: continue
            t, (b1, b2, hB) = Psi_T(m, y1, 1, y2, 1)
            c, s = cand_norm(b1, 1, b2, 1, hB), semi_norm(b1, 1, b2, 1, hB)
            res.setdefault(h, []).append((y1, y2, t, c, s))
            print('h=%.3f y=(%.2f,%.2f) Toda %+.5f cand %+.5f semi %+.5f  T-c %+.5f  T-s %+.5f' % (h, y1, y2, t, c, s, t - c, t - s), flush=True)
json.dump({str(k): v for k, v in res.items()}, open('toda_edge_grid.json', 'w'))

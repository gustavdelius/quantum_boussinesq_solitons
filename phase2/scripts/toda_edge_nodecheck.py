"""Node-count convergence of the normalised edge phase (toda_edge_ba.py) at hbar_T = 0.04."""
import numpy as np, time
from toda_edge_ba import EdgeBA, Psi_T
for N in [700, 1000, 1400]:
    t = time.time(); m = EdgeBA(1.0, 0.04, npts=N)
    ov, _ = Psi_T(m, 2.0, 1, 0.5, 1); ho, _ = Psi_T(m, 1.0, 1, 1.0, -1)
    print('N=%d q=%.10f  Psi_ov=%.8f  Psi_ho=%.8f  (%.1fs)' % (N, m.q, ov, ho, time.time() - t))

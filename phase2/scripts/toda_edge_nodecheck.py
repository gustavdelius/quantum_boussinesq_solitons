"""Node-count convergence of the normalised edge phase (toda_edge_ba.py) at hbar_T = 0.04."""
import numpy as np, time
from toda_edge_ba import EdgeBA, Psi_T
# (V), (c): retain each result to check the displayed mesh convergence.
_check_mesh = []
for N in [700, 1000, 1400]:
    t = time.time(); m = EdgeBA(1.0, 0.04, npts=N)
    ov, _ = Psi_T(m, 2.0, 1, 0.5, 1); ho, _ = Psi_T(m, 1.0, 1, 1.0, -1)
    print('N=%d q=%.10f  Psi_ov=%.8f  Psi_ho=%.8f  (%.1fs)' % (N, m.q, ov, ho, time.time() - t))
    _check_mesh.append((m.q, ov, ho))
# The output displays q to 10 decimals and phases to 8 decimals. Require
# agreement within one unit of each last displayed digit across ALL meshes.
assert np.all(np.isfinite(_check_mesh)), _check_mesh
assert np.ptp(np.array(_check_mesh)[:, 0]) < 1e-10, _check_mesh
assert np.all(np.ptp(np.array(_check_mesh)[:, 1:], axis=0) < 1e-8), _check_mesh
print("ALL CHECKS PASSED")

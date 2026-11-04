#!/usr/bin/env python3
"""Independent audit of the frame_dictionary package:
  - phase2/frame_dictionary.tex
  - phase2/scripts/frame_identities.py (+ frame_identities.json)
  - phase2/scripts/frame_edge.py (+ frame_edge_data.json)
  - phase2/scripts/frame_analysis.py (+ frame_analysis.json)

Does not import or reuse the package's solver classes or extrapolation code.
Uses independent Fejer-I (Chebyshev-root) Nystrom discretization, 70-digit
mpmath arithmetic, exact SymPy derivations, and independent h-grids and
Padé / Neville-Richardson extrapolations.

Writes only phase2/scripts/audit_frame_dictionary_results.json and ends by
printing ALL CHECKS PASSED.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.special import digamma, loggamma
import sympy as sp

mp.mp.dps = 70
START_TIME = time.monotonic()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

AUDITED_FILES = [
    "phase2/frame_dictionary.tex",
    "phase2/scripts/frame_identities.py",
    "phase2/scripts/frame_identities.json",
    "phase2/scripts/frame_edge.py",
    "phase2/scripts/frame_edge_data.json",
    "phase2/scripts/frame_analysis.py",
    "phase2/scripts/frame_analysis.json",
]
INITIAL_HASHES = {
    rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    for rel in AUDITED_FILES
}


# ==============================================================================
# 1. Exact Symbolic Derivations & Circularity Audit (Points 1 & 5)
# ==============================================================================
def check_1_symbolic_and_high_prec_identities() -> dict:
    print("=== Check 1: Symbolic Derivations, Missing Jacobian Tests & Fejer-I 70-dps Solver ===", flush=True)
    t0 = time.monotonic()

    # 1a. Classical Toda-to-Boussinesq dictionary (Claim 1.1 & 3.1)
    kap, b, a, h, y, be, al = sp.symbols("kap b a h y be al", positive=True)
    P0 = 4 * b * (kap * sp.cosh(kap) - sp.sinh(kap)) / a
    E0 = 2 * b**2 * (sp.sinh(kap) * sp.cosh(kap) - kap)
    PB0 = sp.Rational(2, 3) * sp.sinh(be)**3 * sp.cosh(be)
    EB0 = sp.Rational(2, 15) * sp.sinh(be)**3 * (5 + 4 * sp.sinh(be)**2)

    assert sp.simplify(sp.diff(E0, kap) / sp.diff(P0, kap) - a * b * sp.sinh(kap) / kap) == 0
    assert sp.limit(a * P0.subs(kap, sp.sqrt(3) * be) / (6 * sp.sqrt(3) * b * PB0), be, 0) == 1
    assert sp.limit(E0.subs(kap, sp.sqrt(3) * be) / (6 * sp.sqrt(3) * b**2 * EB0), be, 0) == 1
    assert sp.series(E0 - a * b * P0, kap, 0, 7).removeO() == sp.Rational(2, 15) * b**2 * kap**5
    assert sp.series(EB0 - PB0, be, 0, 7).removeO() == sp.Rational(1, 5) * be**5
    assert sp.diff(sp.cosh(be), be) == sp.sinh(be)
    assert sp.diff(sp.cos(al), al) == -sp.sin(al)

    # 1b. Non-tautological series proof of Eq. (9) (contrasting frame_identities.py:30-32)
    p0, dp1, dp2 = sp.symbols("p0 dp1 dp2", real=True)
    P0_fn = sp.Function("P0_fn")
    P1_fn = sp.Function("P1_fn")
    E0_fn = sp.Function("E0_fn")
    E1_fn = sp.Function("E1_fn")
    p_exp = p0 + h * dp1 + h**2 * dp2
    P_series = sp.series(P0_fn(p_exp) + h * P1_fn(p_exp), h, 0, 2).removeO()
    dp1_sol = sp.solve(sp.expand(P_series - P0_fn(p0)).coeff(h, 1), dp1)[0]
    assert sp.simplify(dp1_sol + P1_fn(p0) / sp.diff(P0_fn(p0), p0)) == 0
    E_series = sp.series(E0_fn(p_exp) + h * E1_fn(p_exp), h, 0, 2).removeO().subs(dp1, dp1_sol)
    E1_fixed_P = sp.simplify(sp.expand(E_series - E0_fn(p0)).coeff(h, 1))
    assert sp.simplify(E1_fixed_P - (E1_fn(p0) - sp.diff(E0_fn(p0), p0) * P1_fn(p0) / sp.diff(P0_fn(p0), p0))) == 0

    # 1c. Exact algebraic derivation of the new opposite-edge dispersion identity:
    #     E'(p) - v_s P'(p) = \overline{P}(p) - 2 V(q) [F_p(-q) - F_q(-q)]
    t_s, rhoq, etaq, Dp, Vp, Pbar, fp_plus, fp_minus = sp.symbols(
        "t_s rhoq etaq Dp Vp Pbar fp_plus fp_minus", nonzero=True
    )
    Dq_s = t_s * rhoq
    Vq_s = t_s * etaq
    vs_s = etaq / rhoq
    # From Eq. (5): t*rho(q)*(fp+ - fp-) = D(q) - D(p), t*eta(q)*(fp+ + fp-) = Pbar - (V(p) - V(q))
    Dp_from_end = Dq_s - t_s * rhoq * (fp_plus - fp_minus)
    Vp_from_end = Vq_s + Pbar - t_s * etaq * (fp_plus + fp_minus)
    rel_disp_deriv = sp.simplify(Vp_from_end - vs_s * Dp_from_end)
    assert sp.simplify(rel_disp_deriv - (Pbar - 2 * Vq_s * fp_minus)) == 0

    # 1d. Independent 70-digit mpmath solver using Fejer's first rule (Chebyshev roots)
    #     instead of Gauss-Legendre quadrature.
    class FejerSeaMP:
        def __init__(self, N: int, h_val: mp.mpf, q_val: mp.mpf):
            self.N = N
            self.h = h_val
            self.q = q_val
            self.t = 2 * mp.pi * h_val
            # Fejer's first rule on [-1, 1]: nodes cos((2k+1)pi/(2N))
            thetas = [mp.pi * (2 * k + 1) / (2 * N) for k in range(N)]
            z = [mp.cos(th) for th in thetas]
            w = []
            for th in thetas:
                s_val = mp.mpf(0)
                for m in range(1, N // 2 + 1):
                    s_val += mp.cos(2 * m * th) / (4 * m**2 - 1)
                w.append((2 / mp.mpf(N)) * (1 - 2 * s_val))
            self.x = [q_val * zk for zk in z]
            self.w = [q_val * wk for wk in w]
            A = mp.matrix([
                [self.t * (1 if i == j else 0) - self.K(self.x[i] - self.x[j]) * self.w[j]
                 for j in range(N)]
                for i in range(N)
            ])
            self.inv = A**-1
            self.rho = self.inv * mp.ones(N, 1)
            self.eta = self.inv * mp.matrix(self.x)
            self.n = sum(self.w[i] * self.rho[i] for i in range(N))
            self.a = 1 / self.n
            self.b = mp.exp(-self.a / 2)
            self.Dq = self.D(q_val)
            self.Vq = self.V(q_val)
            self.rhoq = self.Dq / self.t
            self.etaq = self.Vq / self.t
            self.vs = self.Vq / self.Dq

        def K(self, d: mp.mpf) -> mp.mpf:
            return 2 * mp.re(mp.digamma(1 + mp.j * d / self.h)) + 2 * mp.log(self.h)

        def Th(self, d: mp.mpf) -> mp.mpf:
            return 2 * self.h * mp.im(mp.loggamma(1 + mp.j * d / self.h)) + 2 * d * mp.log(self.h)

        def dot(self, f, g) -> mp.mpf:
            return sum(self.w[i] * f[i] * g[i] for i in range(self.N))

        def tv(self, p: mp.mpf):
            return mp.matrix([self.Th(p - xk) for xk in self.x])

        def kv(self, p: mp.mpf):
            return mp.matrix([self.K(p - xk) for xk in self.x])

        def F_vec(self, p: mp.mpf):
            return self.inv * self.tv(p)

        def P(self, p: mp.mpf) -> mp.mpf:
            return p + self.dot(self.tv(p), self.rho)

        def E(self, p: mp.mpf) -> mp.mpf:
            return p**2 / 2 + self.dot(self.tv(p), self.eta)

        def D(self, p: mp.mpf) -> mp.mpf:
            return 1 + self.dot(self.kv(p), self.rho)

        def V(self, p: mp.mpf) -> mp.mpf:
            return p + self.dot(self.kv(p), self.eta)

        def ends(self, p: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
            F = self.F_vec(p)
            return tuple(
                (self.Th(p - e) + self.dot(self.kv(e), F)) / self.t
                for e in (self.q, -self.q)
            )

        def Phi(self, p: mp.mpf, z: mp.mpf) -> mp.mpf:
            return self.Th(p - z) - self.dot(self.kv(p), self.F_vec(z))

        def invert_y(self, y_val: mp.mpf) -> mp.mpf:
            kap_v = mp.sqrt(y_val * self.h / self.b)
            target = 4 * self.b * (kap_v * mp.cosh(kap_v) - mp.sinh(kap_v)) / self.a
            P_edge = self.P(self.q)
            return mp.findroot(
                lambda p: self.P(p) - P_edge - target,
                self.q + target / self.Dq,
                df=self.D,
                solver="newton",
                tol=mp.mpf("1e-55"),
            )

    # Test on independent parameter point (h=0.35, q=1.4) with N=68 Fejer-I nodes
    # (N=56 Fejer-I nodes gave max_id_err = 2.05994e-12 > 1e-12 because Fejer-I has
    # polynomial exactness N-1 rather than 2N-1; refining to N=68 passes < 1e-12
    # without weakening the tolerance).
    sea = FejerSeaMP(68, mp.mpf("0.35"), mp.mpf("1.4"))
    fq_p, fq_m = sea.ends(sea.q)
    id_errors = {
        "edge_momentum_Pq": abs(sea.P(sea.q) - mp.pi * sea.h / sea.a),
        "parity_P": abs(sea.P(mp.mpf("2.1")) + sea.P(mp.mpf("-2.1"))),
        "parity_E": abs(sea.E(mp.mpf("2.1")) - sea.E(mp.mpf("-2.1"))),
    }
    for p_test in (sea.q, mp.mpf("2.1"), mp.mpf("-2.1")):
        fp, fm = sea.ends(p_test)
        F_vec = sea.F_vec(p_test)
        id_errors[f"end_D_{p_test}"] = abs(
            sea.t * sea.rhoq * (fp - fm) - (1 - sea.D(p_test))
        )
        id_errors[f"end_V_{p_test}"] = abs(
            sea.t * sea.etaq * (fp + fm) - (sea.P(p_test) - sea.V(p_test))
        )
        id_errors[f"self_adj_P_{p_test}"] = abs(
            sea.P(p_test) - (p_test + sum(sea.w[i] * F_vec[i] for i in range(sea.N)))
        )
        id_errors[f"self_adj_E_{p_test}"] = abs(
            sea.E(p_test) - (p_test**2 / 2 + sum(sea.w[i] * sea.x[i] * F_vec[i] for i in range(sea.N)))
        )
        # New exact opposite-edge dispersion identity
        Pbar_t = sea.P(p_test) - sea.P(sea.q)
        id_errors[f"opposite_edge_disp_{p_test}"] = abs(
            (sea.V(p_test) - sea.vs * sea.D(p_test))
            - (Pbar_t - 2 * sea.Vq * (fm - fq_m))
        )

    # Check both boxed quantum Jacobian equations in Eq. (7) via direct numerical y-differentiation
    # (note: d\overline{E}/dy was never checked in frame_identities.py or frame_edge.py!)
    jac_errors = {}
    dy_step = mp.mpf("1e-12")
    for y_t in (mp.mpf("0.5"), mp.mpf("2.0")):
        p_c = sea.invert_y(y_t)
        p_p = sea.invert_y(y_t + dy_step)
        p_m = sea.invert_y(y_t - dy_step)
        du_dy_num = ((p_p - p_m) / (2 * dy_step)) / sea.h
        dE_dy_num = (sea.E(p_p) - sea.E(p_m)) / (2 * dy_step)

        kap_c = mp.sqrt(y_t * sea.h / sea.b)
        du_dy_exact = 2 * mp.sinh(kap_c) / (sea.a * sea.D(p_c))
        dE_dy_exact = (2 * sea.h * mp.sinh(kap_c) / sea.a) * (sea.V(p_c) / sea.D(p_c))
        jac_errors[f"du_dy_y{y_t}"] = abs(du_dy_num - du_dy_exact)
        jac_errors[f"dE_dy_y{y_t}"] = abs(dE_dy_num - dE_dy_exact)

    # Check pressure response Eq. (13) with central difference in q at dq = 1e-12 (O(dq^2) = 1e-24 at 70 dps)
    dq = mp.mpf("1e-12")
    s_p1 = FejerSeaMP(68, mp.mpf("0.35"), sea.q + dq)
    s_m1 = FejerSeaMP(68, mp.mpf("0.35"), sea.q - dq)
    p_pr = mp.mpf("2.1")
    fp_pr, fm_pr = sea.ends(p_pr)
    dn_dq = (s_p1.n - s_m1.n) / (2 * dq)
    dP_dq = (s_p1.P(p_pr) - s_m1.P(p_pr)) / (2 * dq)
    da_dq = (s_p1.a - s_m1.a) / (2 * dq)
    dP_da = dP_dq / da_dq

    id_errors["pressure_dn_dq"] = abs(dn_dq - 2 * sea.t * sea.rhoq**2)
    id_errors["pressure_dP_dq"] = abs(dP_dq - sea.t * sea.rhoq * (fp_pr + fm_pr))
    id_errors["pressure_dP_da"] = abs(-sea.a * dP_da - (fp_pr + fm_pr) / (2 * sea.a * sea.rhoq))

    max_id_err = max(id_errors.values())
    max_jac_err = max(jac_errors.values())
    print(f"  [PASS] Fejer-I N=68 max endpoint/pressure error = {mp.nstr(max_id_err, 6)}", flush=True)
    print(f"  [PASS] Direct y-derivative check of Eq. (7) (du/dy and dE/dy) max error = {mp.nstr(max_jac_err, 6)}", flush=True)
    assert max_id_err < mp.mpf("1e-12")
    assert max_jac_err < mp.mpf("1e-18")

    return {
        "fejer_N": 68,
        "exploratory_N56_max_error": "2.05994e-12",
        "h": "0.35",
        "q": "1.4",
        "a": str(sea.a),
        "max_identity_error": str(max_id_err),
        "max_jacobian_derivative_error": str(max_jac_err),
        "identity_errors": {k: str(v) for k, v in id_errors.items()},
        "jacobian_errors": {k: str(v) for k, v in jac_errors.items()},
        "seconds": time.monotonic() - t0,
    }


# ==============================================================================
# 2. Independent Dispersion Solver, Error-Budget Audit & Exact Limits (Point 2)
# ==============================================================================
class FejerSeaSciPy:
    """Independent finite-sea solver using Fejer's first rule (Chebyshev nodes)
    rather than Gauss-Legendre quadrature."""
    def __init__(self, h: float, N: int):
        self.h = float(h)
        self.a = 1.0
        self.b = float(np.exp(-0.5))
        self.t = 2.0 * np.pi * self.h
        self.N = N
        k = np.arange(N, dtype=float)
        th = np.pi * (2.0 * k + 1.0) / (2.0 * N)
        self.z = np.cos(th)
        m = np.arange(1, N // 2 + 1, dtype=float)[:, None]
        self.w = (2.0 / N) * (1.0 - 2.0 * np.sum(np.cos(2.0 * m * th[None, :]) / (4.0 * m**2 - 1.0), axis=0))

        def count_res(q_val: float, store: bool = False) -> float:
            x = q_val * self.z
            wq = q_val * self.w
            A = self.t * np.eye(N) - self.K(x[:, None] - x[None, :]) * wq[None, :]
            lu = lu_factor(A)
            rho = lu_solve(lu, np.ones(N))
            if store:
                self.x, self.wq, self.lu, self.rho = x, wq, lu, rho
            return float(np.dot(wq, rho) - 1.0)

        self.q = float(brentq(count_res, 2.0 * self.b + 0.5 * self.h, 2.0 * self.b + 3.0 * self.h, xtol=1e-14))
        count_res(self.q, store=True)
        self.eta = lu_solve(self.lu, self.x)
        self.Dq = self.D(self.q)
        self.Vq = self.V(self.q)
        self.rhoq = self.Dq / self.t
        self.etaq = self.Vq / self.t
        self.vs = self.Vq / self.Dq

    def K(self, d):
        return 2.0 * np.real(digamma(1.0 + 1j * d / self.h)) + 2.0 * np.log(self.h)

    def Th(self, d):
        return 2.0 * self.h * np.imag(loggamma(1.0 + 1j * d / self.h)) + 2.0 * d * np.log(self.h)

    def D(self, p: float) -> float:
        return float(1.0 + np.dot(self.wq * self.K(p - self.x), self.rho))

    def V(self, p: float) -> float:
        return float(p + np.dot(self.wq * self.K(p - self.x), self.eta))

    def relP(self, p: float) -> float:
        return float(p - self.q + np.dot(self.wq * (self.Th(p - self.x) - self.Th(self.q - self.x)), self.rho))

    def relE(self, p: float) -> float:
        return float(0.5 * (p - self.q) * (p + self.q) + np.dot(self.wq * (self.Th(p - self.x) - self.Th(self.q - self.x)), self.eta))

    def eval_y(self, y: float) -> dict:
        kap = float(np.sqrt(y * self.h / self.b))
        target = float(4.0 * self.b * (kap * np.cosh(kap) - np.sinh(kap)))
        p = float(brentq(lambda p_v: self.relP(p_v) - target, self.q, self.q + max(2.0, 4.0 * self.h * y), xtol=1e-15))
        d_val = self.D(p)
        v_val = self.V(p)
        df = lu_solve(self.lu, self.Th(p - self.x) - self.Th(self.q - self.x))
        ends = [
            float((self.Th(p - e) - self.Th(self.q - e) + np.dot(self.wq * self.K(e - self.x), df)) / self.t)
            for e in (self.q, -self.q)
        ]
        M_val = self.t * self.rhoq * (ends[0] - ends[1])
        Q_val = (ends[0] + ends[1]) / (2.0 * self.rhoq)
        E_val = self.relE(p)
        E_cl = float(2.0 * self.b**2 * (np.sinh(kap) * np.cosh(kap) - kap))
        u_cl = float(2.0 * self.b * (np.cosh(kap) - 1.0) / self.h)
        du_cl = float(np.sinh(kap) / kap)
        return {
            "y": y,
            "k": kap,
            "p": p,
            "u": (p - self.q) / self.h,
            "u_minus_ucl": (p - self.q) / self.h - u_cl,
            "du_dy": 2.0 * np.sinh(kap) / d_val,
            "du_minus_ducl": 2.0 * np.sinh(kap) / d_val - du_cl,
            "P": target,
            "E": E_val,
            "Ecl": E_cl,
            "cE": (E_val - E_cl) / (self.h * target),
            "D": d_val,
            "V": v_val,
            "M": M_val,
            "Q": Q_val,
            "ends": ends,
            "df": df,
        }

    def four_phase(self, pt1: dict, pt2: dict) -> float:
        p1, p2 = pt1["p"], pt2["p"]
        return float(
            self.Th(p1 - p2) - self.Th(p1 - self.q) - self.Th(self.q - p2) + self.Th(0.0)
            - np.dot(self.wq * (self.K(p1 - self.x) - self.K(self.q - self.x)), pt2["df"])
        )


def neville_zero(hs: list[float], vals: list[float]) -> float:
    """Neville polynomial extrapolation to h = 0."""
    n = len(hs)
    T = [float(v) for v in vals]
    for m in range(1, n):
        for i in range(n - m):
            T[i] = (hs[i + m] * T[i] - hs[i] * T[i + 1]) / (hs[i + m] - hs[i])
    return T[0]


def pade_2_1_zero(hs: list[float], vals: list[float]) -> float:
    """Rational Pade [2/1] extrapolation f(h) = (a0 + a1 h + a2 h^2)/(1 + b1 h) at h = 0
    using 4 points."""
    h = np.asarray(hs[-4:], dtype=float)
    v = np.asarray(vals[-4:], dtype=float)
    # a0 + a1 h + a2 h^2 - b1 h v = v
    M = np.column_stack([np.ones(4), h, h**2, -h * v])
    sol = np.linalg.solve(M, v)
    return float(sol[0])


def pade_1_1_zero(hs: list[float], vals: list[float]) -> float:
    """Rational Pade [1/1] extrapolation f(h) = (a0 + a1 h)/(1 + b1 h) at h = 0
    using last 3 points."""
    h = np.asarray(hs[-3:], dtype=float)
    v = np.asarray(vals[-3:], dtype=float)
    M = np.column_stack([np.ones(3), h, -h * v])
    sol = np.linalg.solve(M, v)
    return float(sol[0])


def check_2_dispersion_and_error_budgets() -> dict:
    print("\n=== Check 2: Independent Fejer-I Dispersion Scan & Error-Budget Audit ===", flush=True)
    t0 = time.monotonic()

    # Use an independent h-grid h in {0.06, 0.03, 0.015, 0.0075} with Fejer-I quadrature
    h_ind = [0.06, 0.03, 0.015, 0.0075]
    N_ind = [420, 640, 1050, 1800]
    ys_sample = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    ys_extra = [0.25, 1.44, 2.56]
    all_ys = sorted(set(ys_sample + ys_extra))
    Ps_fixed = [0.15, 0.4, 0.6]

    pkg_analysis = json.loads((HERE / "frame_analysis.json").read_text())
    pkg_one = {row["y"]: row for row in pkg_analysis["one_body"]}
    pkg_fixed = {row["P"]: row for row in pkg_analysis["fixed_P_energy"]}

    ind_solves = []
    for h_v, N_v in zip(h_ind, N_ind):
        sea = FejerSeaSciPy(h_v, N_v)
        pts = {y: sea.eval_y(y) for y in all_ys}
        fixed_p = {}
        for P_v in Ps_fixed:
            kap_p = float(brentq(lambda k: 4.0 * sea.b * (k * np.cosh(k) - np.sinh(k)) - P_v, 1e-5, 3.0))
            pt_p = sea.eval_y(sea.b * kap_p**2 / h_v)
            fixed_p[P_v] = (pt_p["E"] - pt_p["Ecl"]) / h_v
        ind_solves.append({"h": h_v, "N": N_v, "sea": sea, "pts": pts, "fixed_p": fixed_p, "vs_corr": (sea.vs - sea.b) / h_v})

    # Compare independent Fejer-I + Neville/Pade extrapolations against Table 1 & Table 2
    reproduced_table1 = []
    for y_v in ys_sample:
        u_vals = [s["pts"][y_v]["u_minus_ucl"] for s in ind_solves]
        du_vals = [1.0 + s["pts"][y_v]["du_minus_ducl"] for s in ind_solves]
        cE_vals = [s["pts"][y_v]["cE"] for s in ind_solves]

        u_nev3 = neville_zero(h_ind[-3:], u_vals[-3:])
        u_nev4 = neville_zero(h_ind, u_vals)
        u_pade = pade_2_1_zero(h_ind, u_vals)

        du_nev3 = neville_zero(h_ind[-3:], du_vals[-3:])
        du_nev4 = neville_zero(h_ind, du_vals)
        du_pade = pade_2_1_zero(h_ind, du_vals)

        cE_nev3 = neville_zero(h_ind[-3:], cE_vals[-3:])
        cE_nev4 = neville_zero(h_ind, cE_vals)
        cE_pade = pade_2_1_zero(h_ind, cE_vals)

        pkg_u = pkg_one[y_v]["u_minus_y"]["value"]
        pkg_du = 1.0 + pkg_one[y_v]["jacobian_minus_one"]["value"]
        pkg_cE = pkg_one[y_v]["energy_coefficient"]["value"]

        reproduced_table1.append({
            "y": y_v,
            "ind_u_nev4": u_nev4,
            "ind_u_pade21": u_pade,
            "pkg_u": pkg_u,
            "diff_u": abs(u_nev4 - pkg_u),
            "ind_du_nev4": du_nev4,
            "ind_du_pade21": du_pade,
            "pkg_du": pkg_du,
            "diff_du": abs(du_nev4 - pkg_du),
            "ind_cE_nev4": cE_nev4,
            "ind_cE_pade21": cE_pade,
            "pkg_cE": pkg_cE,
            "diff_cE": abs(cE_nev4 - pkg_cE),
            "kap_at_h004": float(np.sqrt(y_v * 0.04 / np.exp(-0.5))),
        })
        # Independent grid extrapolations agree with package values within combined extrapolation spreads
        tol_y = max(5e-7, 3.0 * pkg_one[y_v]["u_minus_y"]["edge_spread"])
        assert abs(u_nev4 - pkg_u) < tol_y
        assert abs(cE_nev4 - pkg_cE) < max(5e-7, 3.0 * pkg_one[y_v]["energy_coefficient"]["edge_spread"])

    reproduced_table2 = []
    for P_v in Ps_fixed:
        e1_vals = [s["fixed_p"][P_v] for s in ind_solves]
        e1_nev4 = neville_zero(h_ind, e1_vals)
        e1_pade = pade_2_1_zero(h_ind, e1_vals)
        pkg_e1 = pkg_fixed[P_v]["coefficient"]["value"]
        reproduced_table2.append({
            "P": P_v,
            "ind_E1_nev4": e1_nev4,
            "ind_E1_pade21": e1_pade,
            "pkg_E1": pkg_e1,
            "diff": abs(e1_nev4 - pkg_e1),
        })
        assert abs(e1_nev4 - pkg_e1) < 2e-7

    # Qualification 2(a): Sound speed correction (v_s - b)/h equals 1/(4*pi)
    vs_vals = [s["vs_corr"] for s in ind_solves]
    vs_nev4 = neville_zero(h_ind, vs_vals)
    vs_pade = pade_2_1_zero(h_ind, vs_vals)
    exact_quarter_pi = float(1.0 / (4.0 * np.pi))
    pkg_vs_quad = pkg_analysis["sound"]["value"]
    pkg_vs_cubic = pkg_analysis["sound"]["cubic"]
    assert abs(vs_nev4 - exact_quarter_pi) < 2e-10
    assert abs(pkg_vs_cubic - exact_quarter_pi) < 5e-11

    # Qualification 2(b): Large-y limit of c_E(y) is 1/(6*pi) with -d_0/(2y) subleading tail
    exact_sixth_pi = float(1.0 / (6.0 * np.pi))
    d0_wh = float(1.0 / 48.0 - 1.0 / (4.0 * np.pi**2))
    cE_asymptotic_checks = []
    for y_v in (4.0, 8.0, 16.0, 32.0, 64.0):
        cE_num = pkg_one[y_v]["energy_coefficient"]["cubic"]
        cE_asy = exact_sixth_pi - 0.5 * d0_wh / y_v
        cE_asymptotic_checks.append({
            "y": y_v,
            "cE_cubic": cE_num,
            "cE_asy_1loop_2loop": cE_asy,
            "diff": abs(cE_num - cE_asy),
        })
    assert abs(cE_asymptotic_checks[2]["diff"]) < 2e-6  # at y=16, matches to 1.2e-6

    # Qualification 2(c): Conditioning amplification vs true quadrature error at y=0.1
    # At y=0.1, h=0.005, h*P_0 ~ 9.57e-8, so float64 roundoff (~4.4e-17) / (h*P_0) ~ 4.6e-10.
    b_val = float(np.exp(-0.5))
    k_small = float(np.sqrt(0.1 * 0.005 / b_val))
    P0_small = float(4.0 * b_val * (k_small * np.cosh(k_small) - np.sinh(k_small)))
    conditioning_factor = 1.0 / (0.005 * P0_small)
    pkg_mesh_y01 = pkg_one[0.1]["energy_coefficient"]["mesh"]
    pkg_mesh_prop_y01 = pkg_one[0.1]["energy_coefficient"]["mesh_propagated"]
    pkg_spread_y01 = pkg_one[0.1]["energy_coefficient"]["edge_spread"]
    # Verify that propagated mesh diagnostic (6.94e-9) actually exceeds edge_spread (5.86e-9) at y=0.1
    assert pkg_mesh_prop_y01 > pkg_spread_y01
    assert conditioning_factor > 1e7

    print(f"  [PASS] Independent Fejer-I h-grid reproduces Table 1 & Table 2 within extrapolation spreads.", flush=True)
    print(f"  [PASS] Sound shift (v_s-b)/h: ind={vs_nev4:.12f}, 1/(4*pi)={exact_quarter_pi:.12f} (diff={abs(vs_nev4-exact_quarter_pi):.2e})", flush=True)
    print(f"  [PASS] Large-y c_E(16)={pkg_one[16.0]['energy_coefficient']['cubic']:.10f} vs 1/(6*pi)-d0/(2y)={exact_sixth_pi - 0.5*d0_wh/16:.10f}", flush=True)
    print(f"  [PASS] Conditioning check at y=0.1: 1/(h*P0)={conditioning_factor:.2e}, mesh_prop={pkg_mesh_prop_y01:.2e} > edge_spread={pkg_spread_y01:.2e}", flush=True)

    return {
        "independent_h_grid": h_ind,
        "independent_N_grid": N_ind,
        "table1_comparison": reproduced_table1,
        "table2_comparison": reproduced_table2,
        "sound_shift": {
            "ind_neville4": vs_nev4,
            "ind_pade21": vs_pade,
            "pkg_quad": pkg_vs_quad,
            "pkg_cubic": pkg_vs_cubic,
            "exact_1_over_4pi": exact_quarter_pi,
            "cubic_minus_exact": abs(pkg_vs_cubic - exact_quarter_pi),
        },
        "cE_large_y_asymptotics": cE_asymptotic_checks,
        "y01_conditioning_diagnostic": {
            "h_times_P0_at_h0005": 0.005 * P0_small,
            "amplification_1_over_hP0": conditioning_factor,
            "pkg_mesh_diff": pkg_mesh_y01,
            "pkg_mesh_propagated": pkg_mesh_prop_y01,
            "pkg_edge_spread": pkg_spread_y01,
            "propagated_mesh_exceeds_edge_spread": bool(pkg_mesh_prop_y01 > pkg_spread_y01),
        },
        "ind_solves_cache": ind_solves,
        "seconds": time.monotonic() - t0,
    }


# ==============================================================================
# 3. Mixed Classical-Frame Remainder: Exact Reduction, New Bases & Point Sets (Point 3)
# ==============================================================================
def check_3_mixed_remainder_and_exact_reduction(ind_solves: list[dict]) -> dict:
    print("\n=== Check 3: Mixed Classical-Frame Remainder, Alternative Bases, New Point Sets & Exact Reduction ===", flush=True)
    t0 = time.monotonic()

    pkg_analysis = json.loads((HERE / "frame_analysis.json").read_text())
    pkg_one = {row["y"]: row for row in pkg_analysis["one_body"]}
    pkg_rects = {row["Y"]: row for row in pkg_analysis["rectangles"]}
    hs_pkg = [0.08, 0.04, 0.02, 0.01, 0.005]
    h_last4 = np.asarray(hs_pkg[-4:], dtype=float)
    h_last3 = np.asarray(hs_pkg[-3:], dtype=float)

    # 3a. Test the saved raw rectangle values with 5 alternative extrapolation bases:
    #     - Pade [1/1] on last 3
    #     - Pade [2/1] on last 4
    #     - 3-point (1, h, h*ln(h)) on last 3
    #     - 4-point (1, h, h^(3/2), h^2) on last 4
    #     - 4-point (1, h, h^2, h^2*ln(h)) on last 4
    #     And compare with our exact analytical reduction in terms of u'(y)!
    alt_basis_results = []
    for Y_v in (4.0, 8.0, 16.0, 32.0):
        raw_v = pkg_rects[Y_v]["raw"]
        v4 = np.asarray(raw_v[-4:], dtype=float)
        v3 = np.asarray(raw_v[-3:], dtype=float)

        p11 = pade_1_1_zero(hs_pkg, raw_v)
        p21 = pade_2_1_zero(hs_pkg, raw_v)
        fit_3pt_hlog = float(np.linalg.solve(np.column_stack([np.ones(3), h_last3, h_last3 * np.log(h_last3)]), v3)[0])
        fit_h32 = float(np.linalg.solve(np.column_stack([np.ones(4), h_last4, h_last4**1.5, h_last4**2]), v4)[0])
        fit_h2log = float(np.linalg.solve(np.column_stack([np.ones(4), h_last4, h_last4**2, h_last4**2 * np.log(h_last4)]), v4)[0])

        # Exact analytical reduction in the h -> 0 limit:
        # D_0(Y) = 8*(sqrt(2Y)/u'(2Y) - sqrt(Y)/u'(Y))*(sqrt(Y/2)/u'(Y/2) - sqrt(Y/4)/u'(Y/4))
        #          - 8*(sqrt(2Y) - sqrt(Y))*(sqrt(Y/2) - sqrt(Y/4))
        def g_dress(y_key: float) -> float:
            up = 1.0 + pkg_one[y_key]["jacobian_minus_one"]["cubic"]
            return float(np.sqrt(y_key) / up)

        D0_from_uprime = float(
            8.0 * (g_dress(2.0 * Y_v) - g_dress(Y_v)) * (g_dress(Y_v / 2.0) - g_dress(Y_v / 4.0))
            - 8.0 * (np.sqrt(2.0 * Y_v) - np.sqrt(Y_v)) * (np.sqrt(Y_v / 2.0) - np.sqrt(Y_v / 4.0))
        )

        alt_basis_results.append({
            "Y": Y_v,
            "pkg_quad": pkg_rects[Y_v]["value"],
            "pkg_cubic": pkg_rects[Y_v]["cubic"],
            "pade_1_1": p11,
            "pade_2_1": p21,
            "fit_3pt_hlog": fit_3pt_hlog,
            "fit_h32": fit_h32,
            "fit_h2log": fit_h2log,
            "exact_from_uprime": D0_from_uprime,
            "diff_cubic_vs_uprime": abs(pkg_rects[Y_v]["cubic"] - D0_from_uprime),
        })
        # All bases give a strictly negative value around -0.50 to -0.39
        assert all(val < -0.30 for val in (p11, p21, fit_3pt_hlog, fit_h32, fit_h2log, D0_from_uprime))

    assert alt_basis_results[0]["diff_cubic_vs_uprime"] < 1e-6  # at Y=4, matches to 1.2e-7!

    # 3b. Exact analytical large-Y expansion of D_0(Y):
    #     Using u'(y) = 1 - 1/(2*pi*y) - d_0/y^2 - 2*e_0/y^3 + O(y^-4),
    #     lim_{Y->oo} D_0(Y) = -(15*sqrt(2) - 20)/pi = -0.3861746185469503...
    Yv, d0_s, e0_s, eps_s = sp.symbols("Yv d0_s e0_s eps_s", positive=True)
    def g_asy(y_expr):
        u_p = 1 - 1 / (2 * sp.pi * y_expr) - d0_s / y_expr**2 - 2 * e0_s / y_expr**3
        return sp.sqrt(y_expr) / u_p

    rect_asy_expr = (
        8 * (g_asy(2 * Yv) - g_asy(Yv)) * (g_asy(Yv / 2) - g_asy(Yv / 4))
        - 8 * (sp.sqrt(2 * Yv) - sp.sqrt(Yv)) * (sp.sqrt(Yv / 2) - sp.sqrt(Yv / 4))
    )
    rect_series = sp.series(rect_asy_expr.subs(Yv, 1 / eps_s), eps_s, 0, 3).removeO()
    D0_inf_sym = sp.simplify(rect_series.subs(eps_s, 0))
    assert sp.simplify(D0_inf_sym - (-(15 * sp.sqrt(2) - 20) / sp.pi)) == 0
    D0_inf_float = float(-(15.0 * np.sqrt(2.0) - 20.0) / np.pi)

    d0_val = sp.Rational(1, 48) - 1 / (4 * sp.pi**2)
    e0_val = (4 + 6 * sp.zeta(3) - sp.pi**2) / (96 * sp.pi**3)
    c_D1_sym = sp.simplify(rect_series.coeff(eps_s, 1).subs({d0_s: d0_val, e0_s: e0_val}))
    c_D2_sym = sp.simplify(rect_series.coeff(eps_s, 2).subs({d0_s: d0_val, e0_s: e0_val}))
    c_D1_float = float(c_D1_sym.evalf(30))
    c_D2_float = float(c_D2_sym.evalf(30))

    # Compare the exact analytical 1/Y expansion against D0_from_uprime at Y=16 and Y=32
    # (note: at Y=4 the lowest corner is Y/4 = 1.0, which is O(1) rather than large-y,
    # so a 4-point quadratic 1/Y fit including Y=4 has truncation offset ~3.5e-3, whereas
    # on Y in {8,16,32} or cubic in 1/Y it converges to < 1.5e-3 of D0_inf, and at Y=32
    # the analytical 2-term 1/Y expansion matches D0_from_uprime to < 2e-5!)
    ys_arr = np.array([4.0, 8.0, 16.0, 32.0])
    d0_uprime_arr = np.array([r["exact_from_uprime"] for r in alt_basis_results])
    fit_invY_quad_all4 = float(np.polyfit(1.0 / ys_arr, d0_uprime_arr, 2)[-1])
    fit_invY_quad_last3 = float(np.polyfit(1.0 / ys_arr[1:], d0_uprime_arr[1:], 2)[-1])
    fit_invY_cubic_all4 = float(np.polyfit(1.0 / ys_arr, d0_uprime_arr, 3)[-1])
    asy_at_32 = D0_inf_float + c_D1_float / 32.0 + c_D2_float / (32.0**2)
    err_asy_32 = abs(alt_basis_results[-1]["exact_from_uprime"] - asy_at_32)
    assert err_asy_32 < 5e-5
    assert abs(fit_invY_quad_last3 - D0_inf_float) < 1.5e-3
    assert abs(fit_invY_cubic_all4 - D0_inf_float) < 1.5e-3

    # 3c. Test on 3 NEW independent point sets using our independent Fejer-I solver:
    #     And verify directly from the 4-term Fredholm phases \widehat{\Phi}(1,2) + \widehat{\Phi}(2,1)!
    h_ind = [s["h"] for s in ind_solves]
    b_val = float(np.exp(-0.5))

    def rem_pair(s_dict: dict, y1: float, y2: float) -> tuple[float, float]:
        sea = s_dict["sea"]
        p1, p2 = s_dict["pts"][y1], s_dict["pts"][y2]
        C12 = -2.0 * p1["k"] * (p2["P"] - 2.0 * b_val * np.sinh(p2["k"]))
        C21 = -2.0 * p2["k"] * (p1["P"] - 2.0 * b_val * np.sinh(p1["k"]))
        from_mq = (p1["M"] * p2["Q"] + p2["M"] * p1["Q"] - C12 - C21) / sea.h
        from_phi = (sea.four_phase(p1, p2) + sea.four_phase(p2, p1) - C12 - C21) / sea.h
        return from_mq, from_phi

    new_point_sets = [
        ("rect_A_(16,4)x(1,0.25)", 16.0, 4.0, 1.0, 0.25),
        ("rect_B_(8,2)x(0.5,0.1)", 8.0, 2.0, 0.5, 0.1),
        ("rect_C_(16,4)x(2.56,1.44)", 16.0, 4.0, 2.56, 1.44),
    ]
    new_rect_results = []
    for label, y_hi, y_lo, z_hi, z_lo in new_point_sets:
        mq_vals, phi_vals = [], []
        for s_dict in ind_solves:
            mq_11, phi_11 = rem_pair(s_dict, y_hi, z_hi)
            mq_01, phi_01 = rem_pair(s_dict, y_lo, z_hi)
            mq_10, phi_10 = rem_pair(s_dict, y_hi, z_lo)
            mq_00, phi_00 = rem_pair(s_dict, y_lo, z_lo)
            mq_vals.append(mq_11 - mq_01 - mq_10 + mq_00)
            phi_vals.append(phi_11 - phi_01 - phi_10 + phi_00)

        max_mq_vs_phi = max(abs(a - b) for a, b in zip(mq_vals, phi_vals))
        ext_nev = neville_zero(h_ind, phi_vals)
        ext_pade = pade_2_1_zero(h_ind, phi_vals)

        # Analytical formula from independent extrapolated u'(y)
        def ind_g(y_k: float) -> float:
            du_v = neville_zero(h_ind, [1.0 + s["pts"][y_k]["du_minus_ducl"] for s in ind_solves])
            return float(np.sqrt(y_k) / du_v)

        analytical_rect = float(
            8.0 * (ind_g(y_hi) - ind_g(y_lo)) * (ind_g(z_hi) - ind_g(z_lo))
            - 8.0 * (np.sqrt(y_hi) - np.sqrt(y_lo)) * (np.sqrt(z_hi) - np.sqrt(z_lo))
        )
        new_rect_results.append({
            "label": label,
            "corners": [y_hi, y_lo, z_hi, z_lo],
            "ext_neville4": ext_nev,
            "ext_pade21": ext_pade,
            "analytical_from_uprime": analytical_rect,
            "diff_ext_vs_analytical": abs(ext_nev - analytical_rect),
            "max_direct_phase_vs_MQ_diff": max_mq_vs_phi,
        })
        assert max_mq_vs_phi < 1e-9
        assert abs(ext_nev - analytical_rect) < 2e-3
        assert abs(ext_nev) > 0.08

    # 3d. Verify single-pair S_0(y,z) formula including the -D(q) endpoint contribution:
    #     S_0(y,z) = 8*sqrt(y*z)*(1/(u'(y)*u'(z)) - 1) - 4*sqrt(pi)*(sqrt(y)/u'(y) + sqrt(z)/u'(z)) + 2*pi
    s_4_1_vals = [rem_pair(s_dict, 4.0, 1.0)[1] for s_dict in ind_solves]
    s_4_1_ext = neville_zero(h_ind, s_4_1_vals)
    up_4 = 1.0 + pkg_one[4.0]["jacobian_minus_one"]["cubic"]
    up_1 = 1.0 + pkg_one[1.0]["jacobian_minus_one"]["cubic"]
    jac_only_4_1 = float(8.0 * np.sqrt(4.0 * 1.0) * (1.0 / (up_4 * up_1) - 1.0))
    full_S0_4_1 = float(
        jac_only_4_1 - 4.0 * np.sqrt(np.pi) * (np.sqrt(4.0) / up_4 + np.sqrt(1.0) / up_1) + 2.0 * np.pi
    )
    assert abs(s_4_1_ext - full_S0_4_1) < 1e-4
    assert abs(s_4_1_ext - jac_only_4_1) > 16.0  # Dropping -D(q) fails by > 16!

    print(f"  [PASS] Alternative extrapolation bases and 3 new point sets confirm nonzero mixed rectangle.", flush=True)
    print(f"  [PASS] Exact reduction to u'(y) matches Y=4 cubic rectangle to {alt_basis_results[0]['diff_cubic_vs_uprime']:.2e}", flush=True)
    print(
        f"  [PASS] Exact large-Y asymptote D_0(oo) = -(15*sqrt(2)-20)/pi = {D0_inf_float:.10f} "
        f"(2-term 1/Y series err at Y=32: {err_asy_32:.2e}; 1/Y fits: {fit_invY_quad_last3:.6f}, {fit_invY_cubic_all4:.6f})",
        flush=True,
    )
    print(f"  [PASS] Single-pair S_0(4,1) = {s_4_1_ext:.6f} vs formula with -D(q): {full_S0_4_1:.6f} (without -D(q): +{jac_only_4_1:.6f})", flush=True)

    return {
        "alternative_bases_on_table3": alt_basis_results,
        "exact_large_Y_asymptote": {
            "formula": "-(15*sqrt(2) - 20)/pi",
            "exact_value": D0_inf_float,
            "c_D1": c_D1_float,
            "c_D2": c_D2_float,
            "two_term_series_error_at_Y32": err_asy_32,
            "quadratic_1_over_Y_fit_Y8_16_32": fit_invY_quad_last3,
            "cubic_1_over_Y_fit_Y4_8_16_32": fit_invY_cubic_all4,
            "quadratic_1_over_Y_fit_all4_including_Y4_corner1": fit_invY_quad_all4,
        },
        "new_point_sets": new_rect_results,
        "single_pair_endpoint_verification": {
            "pair": [4.0, 1.0],
            "extrapolated_S0": s_4_1_ext,
            "full_formula_with_endpoint": full_S0_4_1,
            "no_endpoint_formula": jac_only_4_1,
            "error_of_full_formula": abs(s_4_1_ext - full_S0_4_1),
        },
        "seconds": time.monotonic() - t0,
    }


# ==============================================================================
# 4. Audit of the Two-Body Antisymmetric Freedom A_lambda (Point 4)
# ==============================================================================
def check_4_antisymmetric_freedom_audit() -> dict:
    print("\n=== Check 4: Physical Constraint Audit of the Two-Body Freedom A_lambda ===", flush=True)
    t0 = time.monotonic()

    y1, y2, Y, r, b1, b2 = sp.symbols("y1 y2 Y r b1 b2", positive=True)
    A_sym = y1 * y2 * (y1 - y2) / (y1 + y2)**3

    # 4a. Confirm algebraic properties claimed in Section 6
    assert sp.simplify(A_sym + A_sym.xreplace({y1: y2, y2: y1})) == 0
    assert sp.limit(A_sym, y2, 0) == 0
    A_scaled = sp.simplify(A_sym.subs({y1: Y, y2: r**2 * Y}))
    assert sp.simplify(A_scaled - r**2 * (1 - r**2) / (1 + r**2)**3) == 0

    # 4b. Test 1 (Cluster decomposition / Darboux frame separability):
    #     Any Darboux coordinate shift generated by asymptotic 1-body soliton charges (M_j, Pi_j)
    #     changes the phase by a finite bilinear \sum_{a=1}^K f_a(y_1) g_a(y_2) (Schmidt rank <= 2).
    #     By contrast, A_lambda has infinite Schmidt rank: even its 3x3 determinant is nonzero!
    pts_y = [sp.Integer(1), sp.Integer(2), sp.Integer(3)]
    pts_z = [sp.Integer(4), sp.Integer(5), sp.Integer(6)]
    M_3x3 = sp.Matrix([[A_sym.subs({y1: yi, y2: zj}) for zj in pts_z] for yi in pts_y])
    det_3x3 = sp.simplify(M_3x3.det())
    assert det_3x3 != 0

    # 4c. Test 2 (Microscopic Toda / Krein-Wiener-Hopf differential constraint):
    #     In the Fermi-edge limit, (\partial_{u_1} + \partial_{u_2}) \widehat{\Phi}_q(u_1, u_2)
    #     = 2 \psi_-(u_1) [\widetilde{\Psi}_-(u_2) - \widetilde{\Psi}_-(0)], which is rank-1 separable
    #     (vanishing 2x2 cross-determinant). For A_lambda, (\partial_{y_1} + \partial_{y_2}) A_lambda is
    #     non-zero and has non-zero 2x2 determinant!
    D_sum_A = sp.simplify(sp.diff(A_sym, y1) + sp.diff(A_sym, y2))
    assert sp.simplify(D_sum_A - (y1 - y2) * (y1**2 - 4 * y1 * y2 + y2**2) / (y1 + y2)**4) == 0
    Krein_2x2 = sp.Matrix([[D_sum_A.subs({y1: yi, y2: zj}) for zj in (sp.Integer(2), sp.Integer(5))]
                           for yi in (sp.Integer(1), sp.Integer(3))])
    assert sp.simplify(Krein_2x2.det()) != 0

    # 4d. Test 3 (Crossing / reflection parity in rapidity ratio r = beta_2 / beta_1 and analyticity):
    #     Under r -> -r (reversing sign of beta_2), A_scaled(r) = r^2(1-r^2)/(1+r^2)^3 is EVEN in r,
    #     whereas the physical Wiener-Hopf residual R_0(oo, r) = -(4/pi)(chi_2(r) - r) is ODD in r.
    #     Furthermore, A_scaled(r) has a spurious 3rd-order pole at r = +/- i (beta_1^2 + beta_2^2 = 0).
    assert sp.simplify(A_scaled.subs(r, -r) - A_scaled) == 0
    r_c = sp.Symbol("r_c")
    denom_roots = sp.solve(sp.denom(A_scaled.subs(r, r_c)), r_c)
    assert sp.I in denom_roots and -sp.I in denom_roots

    # 4e. Test 4 (Three-body Bethe consistency):
    #     Scalar 3-body Yang-Baxter and 3-body Bethe momentum conservation hold for any antisymmetric
    #     2-body phase, so they alone do not remove A_lambda; however, 3-body linear sea backflow
    #     F_{p_2,p_3} = F_{p_2} + F_{p_3} plus 1-body charge separability does.
    y3 = sp.Symbol("y3", positive=True)
    pair_sum_3body = sp.simplify(
        A_sym.xreplace({y1: y1, y2: y2}) + A_sym.xreplace({y1: y2, y2: y1})
        + A_sym.xreplace({y1: y2, y2: y3}) + A_sym.xreplace({y1: y3, y2: y2})
        + A_sym.xreplace({y1: y3, y2: y1}) + A_sym.xreplace({y1: y1, y2: y3})
    )
    assert pair_sum_3body == 0

    print(f"  [PASS] A_lambda 3x3 Schmidt determinant at (1,2,3)x(4,5,6) = {det_3x3} != 0 (excluded by 1-body charge separability).", flush=True)
    print(f"  [PASS] (d_y1 + d_y2) A_lambda 2x2 Krein determinant = {sp.simplify(Krein_2x2.det())} != 0 (violates Krein-WH equation).", flush=True)
    print(f"  [PASS] A_lambda(Y, r^2 Y) is even in r and has spurious triple poles at r = +/- i.", flush=True)

    return {
        "schmidt_3x3_det_at_123_456": str(det_3x3),
        "krein_sum_derivative": str(D_sum_A),
        "krein_2x2_separability_det": str(sp.simplify(Krein_2x2.det())),
        "even_under_r_reflection": True,
        "spurious_poles_in_r": ["+i", "-i"],
        "seconds": time.monotonic() - t0,
    }


# ==============================================================================
# 5. Continuum Non-Uniformity & Phase 2i Wiener-Hopf Factor Check (Sections 7-8)
# ==============================================================================
def check_5_continuum_and_wh_factor() -> dict:
    print("\n=== Check 5: Continuum Mode-Sum Non-Uniformity & Phase 2i WH Factor Product ===", flush=True)
    t0 = time.monotonic()

    # 5a. Continuum mode-sum momentum shift relative to P_{g,0} at fixed y = alpha^2 / hbar_B
    al, hB, c_const = sp.symbols("al hB c_const", positive=True)
    Pg0 = sp.Rational(2, 3) * sp.sin(al)**3 * sp.cos(al)
    Pg0_lead = sp.series(Pg0, al, 0, 4).removeO()
    assert sp.simplify(Pg0_lead - sp.Rational(2, 3) * al**3) == 0
    dPg = al * (sp.log(al) + c_const) / sp.pi
    # Leading term in al -> 0 at fixed y = al^2 / hB is 3*hB*(log(al)+c)/(2*pi*al^2) = 3*(log(al)+c)/(2*pi*y)
    leading_ratio = sp.simplify((hB * dPg / Pg0_lead) - 3 * hB * (sp.log(al) + c_const) / (2 * sp.pi * al**2))
    assert leading_ratio == 0

    # 5b. Phase 2i Wiener-Hopf factor product check and downstream comparison with Table 1
    def phi_wh(s_val: mp.mpc) -> mp.mpc:
        return (
            mp.sqrt(2 * mp.pi / s_val)
            * mp.gamma(1 + s_val / (2 * mp.pi))
            * mp.exp(-(s_val / (2 * mp.pi)) * mp.log(s_val / (2 * mp.pi * mp.e)))
        )

    wh_checks = []
    for om_str in ("0.2", "1.0", "3.0"):
        om = mp.mpf(om_str)
        prod_val = mp.re(phi_wh(mp.j * om) * phi_wh(-mp.j * om))
        exact_prod = mp.pi * mp.exp(om / 2) / mp.sinh(om / 2)
        printed_p2i = 2 * mp.pi**2 * mp.exp(mp.pi * om) / mp.sinh(mp.pi * om)
        err_exact = abs(prod_val - exact_prod)
        diff_printed = abs(prod_val - printed_p2i)
        assert err_exact < mp.mpf("1e-65")
        assert diff_printed > mp.mpf("1.0")
        wh_checks.append({
            "omega": om_str,
            "product": mp.nstr(prod_val, 25),
            "exact_pi_exp_over_sinh": mp.nstr(exact_prod, 25),
            "error_vs_exact": str(err_exact),
            "printed_eq3_middle": mp.nstr(printed_p2i, 25),
        })

    # Verify that \widetilde{\Phi}_{WH}(s) = \Phi_{WH}(s)/sqrt(2*pi) (Eq. 4 of Phase 2i)
    # is unaffected by the Eq. 3 middle-display typo and matches Table 1's u(y)-y and u'(y)-1!
    c_edge = float((1 - mp.log(8 * mp.pi)) / (2 * mp.pi))
    d0 = float(mp.mpf(1) / 48 - 1 / (4 * mp.pi**2))
    e0 = float((4 + 6 * mp.zeta(3) - mp.pi**2) / (96 * mp.pi**3))
    pkg_analysis = json.loads((HERE / "frame_analysis.json").read_text())
    pkg_one = {row["y"]: row for row in pkg_analysis["one_body"]}
    wh_du_comparison = []
    for y_v in (4.0, 8.0, 16.0, 32.0):
        du_wh = -float(np.log(y_v)) / (2.0 * np.pi) + c_edge + d0 / y_v + e0 / y_v**2
        du_pkg = pkg_one[y_v]["u_minus_y"]["cubic"]
        up_wh = -1.0 / (2.0 * np.pi * y_v) - d0 / y_v**2 - 2.0 * e0 / y_v**3
        up_pkg = pkg_one[y_v]["jacobian_minus_one"]["cubic"]
        wh_du_comparison.append({
            "y": y_v,
            "pkg_u_minus_y_cubic": du_pkg,
            "wh_asymptotic_to_O_y2": du_wh,
            "abs_diff_u_minus_y": abs(du_pkg - du_wh),
            "pkg_uprime_minus_1_cubic": up_pkg,
            "wh_uprime_minus_1_to_O_y3": up_wh,
            "abs_diff_uprime": abs(up_pkg - up_wh),
        })
    assert wh_du_comparison[1]["abs_diff_u_minus_y"] < 1.5e-6  # y=8: 1.01e-6 (edge_spread=3.45e-6)
    assert wh_du_comparison[1]["abs_diff_uprime"] < 5e-7       # y=8: u'-1 matches to ~2.4e-7
    assert wh_du_comparison[2]["abs_diff_uprime"] < 2e-7       # y=16: u'-1 matches to ~1.2e-7

    print(
        f"  [PASS] WH factor product matches pi*exp(|w|/2)/sinh(|w|/2) to < 1e-65; "
        f"Phase 2i tilde-Phi_WH matches Table 1 at y=8 (du diff {wh_du_comparison[1]['abs_diff_u_minus_y']:.2e}, u' diff {wh_du_comparison[1]['abs_diff_uprime']:.2e})",
        flush=True,
    )

    return {
        "wh_factor_product_checks": wh_checks,
        "wh_du_vs_table1": wh_du_comparison,
        "seconds": time.monotonic() - t0,
    }


def main() -> None:
    res1 = check_1_symbolic_and_high_prec_identities()
    res2 = check_2_dispersion_and_error_budgets()
    ind_solves = res2.pop("ind_solves_cache")
    res3 = check_3_mixed_remainder_and_exact_reduction(ind_solves)
    res4 = check_4_antisymmetric_freedom_audit()
    res5 = check_5_continuum_and_wh_factor()

    # Verify that no audited file was modified during execution
    final_hashes = {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in AUDITED_FILES
    }
    assert final_hashes == INITIAL_HASHES, "Audited file hashes changed during audit run!"

    out = {
        "precision_dps": mp.mp.dps,
        "audited_file_sha256": INITIAL_HASHES,
        "check_1_identities_and_jacobian": res1,
        "check_2_dispersion_and_budgets": res2,
        "check_3_mixed_remainder": res3,
        "check_4_antisymmetric_freedom": res4,
        "check_5_continuum_and_wh_product": res5,
        "recorded_package_script_runtimes_seconds": {
            "frame_edge_cached_validation": 1.62,
            "frame_edge_full_recompute_recorded_in_json": 325.81,
            "frame_identities_fresh_run": 140.18,
            "frame_analysis_fresh_run": 0.19,
        },
        "total_audit_runtime_seconds": time.monotonic() - START_TIME,
    }
    (HERE / "audit_frame_dictionary_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nTotal audit runtime: {out['total_audit_runtime_seconds']:.2f}s", flush=True)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()

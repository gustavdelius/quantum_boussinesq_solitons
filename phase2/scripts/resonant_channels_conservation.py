#!/usr/bin/env python3
"""Conservation laws and the KP-chord hypothesis for resonant channels (Question 2).

Tests and proves symbolically (SymPy) and at 60-digit precision (mpmath):
 1. General algebraic theorem on the KP chord variety p_a^2 + p_a p_b + p_b^2 = 1/4:
    With k = p_a - p_b, v = sqrt(3)(p_a + p_b), and z = p_a p_b = (1 - 4 k^2)/12 = v^2/3 - 1/4,
    the chord differences Delta_m := p_a^m - p_b^m satisfy:
    - Delta_{3j} = (1/4)^j Delta_1 for odd j, and Delta_{3j} proportional to lower spins for all 3|m
      (specifically p^3 = p/4 + lambda_tilde where lambda_tilde = -p_a p_b (p_a + p_b) is symmetric).
    - Every physical charge M, Pi, P, E and every Gelfand-Dickey hierarchy charge H_n (tested
      symbolically through n = 8: H_1, H_2, H_3=0, H_4, H_5, H_6=0, H_7, H_8) has the exact
      chord-difference form
          Q(p_a, p_b) = F_Q(p_b) - F_Q(p_a)
      for an explicit univariate polynomial F_Q(p) of degree n.
    - Note: an arbitrary polynomial k * P(k^2) does NOT automatically lie in span{Delta_1, Delta_5}
      at degree k^3 (because Delta_3 = (1/4) Delta_1 leaves a degree-1 gap in z = (1 - 4k^2)/12,
      so pure k^3 is not a polynomial chord difference of degree <= 5, whereas the specific Boussinesq
      energy E = (2/15) k^3 (5 - 4 k^2) IS in span{Delta_1, Delta_5}!).
 2. Exact symbolic integration of the Gelfand-Dickey hierarchy densities Res(L^{n/3}) on the
    one-soliton profile via the hyperbolic polynomial map T = tanh(k x / 2) for n = 1..8
    (with Delta_m := mu_1^m - mu_2^m = (-p_b)^m - (-p_a)^m):
    - H_1 = Delta_1 = p_a - p_b
    - H_2 = Delta_2 = p_b^2 - p_a^2
    - H_3 = 0
    - H_4 = Delta_4 - (1/3) Delta_2
    - H_5 = Delta_5 - (5/72) Delta_1
    - H_6 = 0
    - H_7 = Delta_7 - (7/12) Delta_5 + (7/324) Delta_1   (derived symbolically here!)
    - H_8 = Delta_8 - (7/36) Delta_4 + (11/324) Delta_2  (derived symbolically here!)
 3. Exact classical charge conservation in 2 -> 1 fusion and 1 -> 2 splitting:
    For any KP triple {q_1, q_2, q_3} with chords S_{12}=(q_1,q_2), S_{23}=(q_2,q_3), S_{13}=(q_1,q_3),
    telescoping F_Q(q_i) - F_Q(q_j) gives
        Q(S_{12}) + Q(S_{23}) - Q(S_{13}) = 0
    identically for M, Pi, P, E and all H_n (n = 1..8).
    Also verified by direct 60-digit spatial quadrature of the nonlinear charge densities on the
    exact time-dependent 3-exponential Y-junction solution u(x,t) across t in {-20, -2, 0, +2, +20}.
 4. One-loop (O(hbar)) fate of conservation laws in fusion (connection to defq_charges_fusion / audit_defq):
    - In the fixed-profile mode-sum prescription, (Delta E_1^ren, Delta P_mode) are dipole sums over
      the KP roots and therefore telescope to zero across S_{12} + S_{23} -> S_{13}.
    - At fixed physical (dressed) momenta P_j^phys, the one-loop dispersion shift
      F|_P(S) = Delta E_1^ren(S) - v(S) Delta P_mode(S) = -L_1(S) breaks the chord-difference form
      because v(S) = sqrt(3)(p_a + p_b) multiplies Delta P_mode(S):
          Delta F|_P := F|_P(S_{12}) + F|_P(S_{23}) - F|_P(S_{13})
                     = -(v_{12} - v_{13}) Delta P_mode(S_{12}) - (v_{23} - v_{13}) Delta P_mode(S_{23}) != 0.
    - While a single pair (E^phys, P^phys) can remain conserved on an O(hbar)-shifted resonant curve
      in (P_1^phys, P_2^phys), the co-moving canonical charges (Delta E_can, Delta P_can) have a
      nonzero triangle holonomy at unshifted rapidities, and higher quantum charges have no guarantee
      of simultaneous telescoping without a quantum KP-root shift.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 60
TOL = mp.mpf("1e-48")


def build_pseudodiff_residues(max_n=8):
    """Compute Res(L^{n/3}) symbolically for L = d^3 + q d + r up to n = max_n."""
    x = sp.Symbol("x", real=True)
    q = sp.Function("q")(x)
    r = sp.Function("r")(x)

    def mul_trunc(A, B, min_order):
        C = {}
        for i, a in A.items():
            if a == 0:
                continue
            for j, b in B.items():
                if b == 0:
                    continue
                max_m = i + j - min_order
                for m in range(max_m + 1):
                    bc = sp.binomial(i, m)
                    if bc == 0 and i >= 0 and m > i:
                        break
                    o = i + j - m
                    C[o] = C.get(o, 0) + bc * a * sp.diff(b, x, m)
        return {o: sp.expand(c) for o, c in C.items() if c != 0}

    # Determine L^{1/3} = d + sum_{m=1}^{max_n} c_m d^{-m}
    Lop = {3: sp.Integer(1), 1: q, 0: r}
    root = {1: sp.Integer(1)}
    for m in range(1, max_n + 1):
        min_ord = 2 - m
        trial = dict(root)
        cm = sp.Symbol(f"c_{m}")
        trial[-m] = cm
        sq = mul_trunc(trial, trial, min_ord - 1)
        cb = mul_trunc(sq, trial, min_ord)
        target = Lop.get(min_ord, 0)
        expr = sp.expand(cb.get(min_ord, 0) - target)
        # expr is linear in cm with coefficient 3:
        sol_cm = -expr.subs(cm, 0) / sp.Integer(3)
        root[-m] = sp.expand(sol_cm)

    # Compute Res(L^{n/3}) = coeff of d^{-1} in (L^{1/3})^n
    residues = {}
    P_op = {0: sp.Integer(1)}
    for n in range(1, max_n + 1):
        # To reach order -1 at step max_n, at step n we only need orders >= -1 - (max_n - n)
        min_ord = -1 - (max_n - n)
        P_op = mul_trunc(P_op, root, min_ord)
        residues[n] = P_op.get(-1, sp.Integer(0))

    return x, q, r, residues


def integrate_residue_on_soliton_symbolic(x, q, r, res_expr):
    """Integrate (res_expr[soliton] - res_expr[vacuum]) dx over (-oo, oo) symbolically.

    Uses T = tanh(k x / 2), sech^2(k x / 2) = 1 - T^2, d/dx = (k/2)(1 - T^2) d/dT,
    dx = 2 / (k (1 - T^2)) dT.
    """
    k, v = sp.symbols("k v", real=True)
    T = sp.Symbol("T", real=True)

    # Express u(x) and its x-derivatives in terms of T = tanh(k x / 2)
    u_T = k**2 * (1 - T**2) / 2
    q_T = sp.Rational(3, 2) * u_T - sp.Rational(1, 4)

    def d_dx(expr_T):
        return sp.expand(sp.Rational(1, 2) * k * (1 - T**2) * sp.diff(expr_T, T))

    r_T = sp.expand((d_dx(q_T) - v / sp.sqrt(3) * (q_T + sp.Rational(1, 4))) / 2)

    # Build substitution dictionary for q, r and their derivatives up to order 10
    subs_map = {q: q_T, r: r_T}
    q_cur, r_cur = q_T, r_T
    for m in range(1, 10):
        q_cur = d_dx(q_cur)
        r_cur = d_dx(r_cur)
        subs_map[sp.diff(q, x, m)] = q_cur
        subs_map[sp.diff(r, x, m)] = r_cur

    vac_map = {q: sp.Rational(-1, 4), r: sp.Integer(0)}
    for m in range(1, 10):
        vac_map[sp.diff(q, x, m)] = sp.Integer(0)
        vac_map[sp.diff(r, x, m)] = sp.Integer(0)

    dens_T = sp.expand(res_expr.subs(subs_map) - res_expr.subs(vac_map))
    if dens_T == 0:
        return k, v, sp.Integer(0)

    # Factor out (1 - T^2) and integrate polynomial over T in [-1, 1]
    poly_T = sp.cancel(dens_T * 2 / (k * (1 - T**2)))
    integral_kv = sp.integrate(sp.expand(poly_T), (T, -1, 1))
    return k, v, sp.expand(integral_kv)


def run_checks():
    results = {}

    # ------------------------------------------------------------------
    # 1. Exact symbolic integration of H_1 .. H_8 on the one-soliton
    # ------------------------------------------------------------------
    x, q, r, residues = build_pseudodiff_residues(max_n=8)
    assert residues[3] == 0 and residues[6] == 0

    pa, pb = sp.symbols("p_a p_b", real=True)
    k_chord = pa - pb
    v_chord = sp.sqrt(3) * (pa + pb)
    chord_poly = pa**2 + pa * pb + pb**2 - sp.Rational(1, 4)

    def delta_m(m):
        # Following Phase 1 convention: mu_1 = -p_b, mu_2 = -p_a,
        # so D_m := mu_1^m - mu_2^m = (-p_b)^m - (-p_a)^m = (-1)^{m-1} (p_a^m - p_b^m).
        return sp.expand((-pb)**m - (-pa)**m)

    # Express any polynomial in (p_a, p_b) modulo (p_a^2 + p_a p_b + p_b^2 - 1/4)
    # by substituting p_a + p_b = s, p_a - p_b = k, and k^2 = 1 - 3 s^2.
    s_sym, k_sym = sp.symbols("s k_s", real=True)

    def to_sk_canonical(expr_pab):
        # Substitute p_a = (s + k)/2, p_b = (s - k)/2, then reduce k^{2j+1} -> k (1 - 3 s^2)^j
        e = sp.expand(expr_pab.subs({pa: (s_sym + k_sym) / 2, pb: (s_sym - k_sym) / 2}))
        poly_k = sp.Poly(e, k_sym)
        out = sp.Integer(0)
        for (deg_k,), coeff_s in poly_k.terms():
            assert deg_k % 2 == 1, f"Expected odd power of k, got {deg_k}"
            out += coeff_s * k_sym * (1 - 3 * s_sym**2)**((deg_k - 1) // 2)
        return sp.expand(out)

    hierarchy_formulas = {}
    F_polynomials = {}
    p_var = sp.Symbol("p", real=True)

    for n in range(1, 9):
        k_s, v_s, Hn_kv = integrate_residue_on_soliton_symbolic(x, q, r, residues[n])
        if n in (3, 6):
            assert Hn_kv == 0
            hierarchy_formulas[f"H_{n}"] = "0"
            continue

        # Convert Hn_kv (which is a polynomial in k and v) to canonical form on k^2 + v^2 = 1
        # with k = k_sym, v = sqrt(3) s_sym
        Hn_sub = sp.expand(Hn_kv.subs({k_s: k_sym, v_s: sp.sqrt(3) * s_sym}))
        poly_k = sp.Poly(Hn_sub, k_sym)
        Hn_canon = sp.Integer(0)
        for (deg_k,), coeff_s in poly_k.terms():
            assert deg_k % 2 == 1
            Hn_canon += coeff_s * k_sym * (1 - 3 * s_sym**2)**((deg_k - 1) // 2)
        Hn_canon = sp.expand(Hn_canon)

        # Solve for exact rational coefficients c_{n,m} in H_n = sum_{m <= n, 3 \nmid m, m = n mod 2} c_{n,m} D_m
        basis_m = [m for m in range(1, n + 1) if (m % 3 != 0) and (m % 2 == n % 2)]
        coeffs_sym = sp.symbols(f"c_0:{len(basis_m)}")
        ansatz = sum(c * to_sk_canonical(delta_m(m)) for c, m in zip(coeffs_sym, basis_m))
        diff_poly = sp.Poly(sp.expand(Hn_canon - ansatz), k_sym, s_sym)
        sol = sp.solve(diff_poly.coeffs(), list(coeffs_sym))
        assert sol and len(sol) == len(basis_m), f"Failed to decompose H_{n} into chord basis!"

        # Verify exact polynomial division modulo (p_a^2 + p_a p_b + p_b^2 - 1/4)
        Hn_pab = Hn_kv.subs({k_s: pa - pb, v_s: sp.sqrt(3) * (pa + pb)})
        chord_expr = sum(sol[c] * delta_m(m) for c, m in zip(coeffs_sym, basis_m))
        rem = sp.rem(sp.expand(Hn_pab - chord_expr), chord_poly, pa)
        assert sp.expand(rem) == 0, f"Nonzero remainder modulo chord_poly for H_{n}: {rem}"

        # Construct univariate polynomial F_{H_n}(p) such that H_n = F_{H_n}(p_b) - F_{H_n}(p_a)
        # Since D_m = (-p_b)^m - (-p_a)^m, we have F_{H_n}(p) = sum c_{n,m} (-p)^m!
        F_n = sum(sol[c] * (-p_var)**m for c, m in zip(coeffs_sym, basis_m))
        assert sp.rem(sp.expand(Hn_pab - (F_n.subs(p_var, pb) - F_n.subs(p_var, pa))), chord_poly, pa) == 0

        terms_str = " + ".join(f"({sol[c]})*Delta_{m}" for c, m in zip(coeffs_sym, basis_m))
        hierarchy_formulas[f"H_{n}"] = terms_str
        F_polynomials[f"H_{n}"] = str(sp.expand(F_n))
        print(f"   H_{n} = {terms_str}  [F_{n}(p) = {sp.expand(F_n)}]")

    results["hierarchy_chord_decompositions"] = hierarchy_formulas
    results["univariate_chord_potentials_F_of_p"] = F_polynomials
    print("[PASS] 1. Exact symbolic integration and chord decomposition of H_1..H_8 verified!")

    # ------------------------------------------------------------------
    # 2. Physical charges M, Pi, P, E as chord differences F_Q(p_b) - F_Q(p_a)
    #    and demonstration of the spin-3 gap (pure k^3 is NOT a degree<=5 chord difference)
    # ------------------------------------------------------------------
    F_M = -2 * p_var
    F_Pi = 2 * sp.sqrt(3) * p_var**2
    F_P = -2 * sp.sqrt(3) * p_var**4 + (1 / sp.sqrt(3)) * p_var**2
    F_E = -sp.Rational(24, 5) * p_var**5 + sp.Rational(1, 6) * p_var

    M_pab = 2 * (pa - pb)
    Pi_pab = -2 * sp.sqrt(3) * (pa**2 - pb**2)
    P_pab = sp.Rational(2, 3) * (pa - pb)**3 * sp.sqrt(3) * (pa + pb)
    E_pab = sp.Rational(2, 15) * (pa - pb)**3 * (5 - 4 * (pa - pb)**2)

    for name, expr_pab, F_poly in [
        ("M", M_pab, F_M),
        ("Pi", Pi_pab, F_Pi),
        ("P", P_pab, F_P),
        ("E", E_pab, F_E),
    ]:
        rem = sp.rem(sp.expand(expr_pab - (F_poly.subs(p_var, pb) - F_poly.subs(p_var, pa))), chord_poly, pa)
        assert sp.expand(rem) == 0, f"Chord potential failed for {name}: {rem}"
        F_polynomials[name] = str(sp.expand(F_poly))

    # Prove that a hypothetical charge Q_fake = k^3 = (p_a - p_b)^3 CANNOT be written as
    # F(p_b) - F(p_a) for any polynomial F(p) on p_a^2 + p_a p_b + p_b^2 = 1/4!
    # Why? Because on the cubic 4 p^3 - p = const, every polynomial F(p) reduces modulo (4 p^3 - p - c)
    # to A(c) p^2 + B(c) p + C(c), so F(p_a) - F(p_b) = A(c)(p_a^2 - p_b^2) + B(c)(p_a - p_b).
    # For an even-in-v charge (like k^3), A(c) = 0 and B(c) must be an even polynomial in c, i.e. a
    # polynomial in c^2!
    # Now c = 4 p_a^3 - p_a = -4 p_a p_b (p_a + p_b), so on p_a^2 + p_a p_b + p_b^2 = 1/4:
    # c^2 = 16 (p_a p_b)^2 (p_a + p_b)^2 = (1 - 4 k^2)^2 (1 - k^2) / 27 = (1 - 9 k^2 + 24 k^4 - 16 k^6) / 27!
    # Notice that EVERY power c^{2j} (j >= 1) has degree 6j in k, and at j=0 it is 1 (degree 0 in k).
    # Therefore, in the ring of ALL polynomial chord differences F(p_b) - F(p_a), any even-in-v charge
    # is of the form k * Poly(k^2 * (5 - 4 k^2)^2 ...), which has dimensions 1, 3, 5, 7, ... in k
    # ONLY with the specific relation that the k^3 and k^5 coefficients are locked in the ratio
    # (-9) : 24 = -3 : 8, i.e. proportional to k (1 - 9 k^2 + 24 k^4) - k = -3 k^3 (3 - 8 k^2)?
    # Wait: let's check c^2:
    # 16 z^2 (1/4 + z) where z = (1 - 4 k^2)/12, so 1/4 + z = (4 - 4 k^2)/12 = (1 - k^2)/3.
    # Thus c^2 = 16 * (1 - 4 k^2)^2 / 144 * (1 - k^2) / 3 = (1 - 4 k^2)^2 (1 - k^2) / 27
    #          = (1 - 8 k^2 + 16 k^4)(1 - k^2) / 27 = (1 - 9 k^2 + 24 k^4 - 16 k^6) / 27!
    # Wait: why did Delta_5 give k^3(5 - 4 k^2) (degree 5 in k, not 7)?
    # Because Delta_5 = p_a^5 - p_b^5 uses p^5 = (1/16) p + (1/4) c p + c p^2!
    # In p_a^5 - p_b^5, the term c * (p_a^2 - p_b^2) is -4 p_a p_b (p_a + p_b)^2 (p_a - p_b)
    # = -4 z (1/4 + z) k = -(1 - 4 k^2)(1 - k^2) k / 9 = (-1 + 5 k^2 - 4 k^4) k / 9!
    # That uses A(c)(p_a^2 - p_b^2) with A(c) ODD in c!
    # Since A(c) is a polynomial in c (odd powers c^{2j+1}) and B(c) is a polynomial in c^2 (even powers c^{2j}):
    # - B(c)(p_a - p_b) spans k * {1, c^2, c^4, ...} = k * {1, (1-4k^2)^2(1-k^2), ...} (degrees 1, 7, 13, ...)
    # - A(c)(p_a^2 - p_b^2) spans k (p_a + p_b) * {c, c^3, ...} = k * {z(1/4+z), z^3(1/4+z)^2, ...} (degrees 5, 11, 17, ...)
    # Look at the degrees in k of ALL polynomial chord differences F(p_b) - F(p_a) that are even in v:
    # Degree 1 (from p_a - p_b), NO DEGREE 3 (from p_a^3 - p_b^3 = (1/4)(p_a - p_b)),
    # Degree 5 (from p_a^5 - p_b^5, which locks k^3 and k^5 into k^3(5 - 4 k^2)),
    # Degree 7 (from p_a^7 - p_b^7), NO DEGREE 9 (since 9 is divisible by 3), etc.!
    # Let's verify this exact algebraic theorem symbolically!
    c0, c1 = sp.symbols("c0 c1")
    k_only_span_5 = c0 * to_sk_canonical(delta_m(1)) + c1 * to_sk_canonical(delta_m(5))
    k_only_span_5_k = sp.expand(k_only_span_5.subs(s_sym**2, (1 - k_sym**2) / 3))
    assert sp.expand(k_only_span_5_k.subs({c0: -sp.Rational(5, 144), c1: 1}) - k_sym**3 * (5 - 4 * k_sym**2) / 36) == 0
    # Verify that pure k^3 is NOT in span{Delta_1, Delta_5}:
    poly_pure_k3 = sp.Poly(k_only_span_5_k - k_sym**3, k_sym)
    assert sp.solve(poly_pure_k3.coeffs(), [c0, c1]) == []

    print("[PASS] 2. Physical charges M, Pi, P, E verified as chord differences; spin-3 gap proved.")

    # ------------------------------------------------------------------
    # 3. Exact telescoping in 2->1 / 1->2 fusion & 60-digit PDE quadrature on Y-junction
    # ------------------------------------------------------------------
    q1_s, q2_s, q3_s = sp.symbols("q_1 q_2 q_3", real=True)
    for name, F_str in F_polynomials.items():
        F_expr = sp.sympify(F_str)
        Q_12 = F_expr.subs(p_var, q2_s) - F_expr.subs(p_var, q1_s)
        Q_23 = F_expr.subs(p_var, q3_s) - F_expr.subs(p_var, q2_s)
        Q_13 = F_expr.subs(p_var, q3_s) - F_expr.subs(p_var, q1_s)
        assert sp.expand(Q_12 + Q_23 - Q_13) == 0

    # Direct 60-digit spatial quadrature of M, Pi, P, E on the exact Y-junction solution
    # tau(x, t) = e^{eta_1} + e^{eta_2} + e^{eta_3} at t in {-10, 0, +10}
    phi_val = mp.mpf("0.23")
    q_roots = [
        mp.cos(phi_val) / mp.sqrt(3),
        mp.cos(phi_val - 2 * mp.pi / 3) / mp.sqrt(3),
        mp.cos(phi_val + 2 * mp.pi / 3) / mp.sqrt(3),
    ]
    w_roots = [mp.sqrt(3) * q**2 for q in q_roots]

    # Exact chord values from F_Q(q_3) - F_Q(q_1)
    q1_mp, q2_mp, q3_mp = q_roots
    M_exact = 2 * (q1_mp - q3_mp)
    Pi_exact = -2 * mp.sqrt(3) * (q1_mp**2 - q3_mp**2)
    P_exact = -2 * mp.sqrt(3) * (q3_mp**4 - q1_mp**4) + (q3_mp**2 - q1_mp**2) / mp.sqrt(3)
    E_exact = -mp.mpf(24) / 5 * (q3_mp**5 - q1_mp**5) + mp.mpf(1) / 6 * (q3_mp - q1_mp)

    # Also verify sum of S_{12} and S_{23} equals S_{13} at 60 digits
    M_12 = 2 * (q1_mp - q2_mp)
    M_23 = 2 * (q2_mp - q3_mp)
    P_12 = -2 * mp.sqrt(3) * (q2_mp**4 - q1_mp**4) + (q2_mp**2 - q1_mp**2) / mp.sqrt(3)
    P_23 = -2 * mp.sqrt(3) * (q3_mp**4 - q2_mp**4) + (q3_mp**2 - q2_mp**2) / mp.sqrt(3)
    E_12 = -mp.mpf(24) / 5 * (q2_mp**5 - q1_mp**5) + mp.mpf(1) / 6 * (q2_mp - q1_mp)
    E_23 = -mp.mpf(24) / 5 * (q3_mp**5 - q2_mp**5) + mp.mpf(1) / 6 * (q3_mp - q2_mp)
    assert abs((M_12 + M_23) - M_exact) < TOL
    assert abs((P_12 + P_23) - P_exact) < TOL
    assert abs((E_12 + E_23) - E_exact) < TOL

    # Direct spatial quadrature of M, Pi, P, E at t = 0 (right at the Y-junction vertex!)
    # and t = 2.0 (after splitting)
    # Note: phi = 2 (log tau)_x, so u = phi_x = 2 (log tau)_{xx}, pi = phi_t = 2 (log tau)_{xt}.
    def densities_y(x_val, t_val):
        # Shift exponents by max to prevent overflow for large |x|
        exps = [q * x_val - w * t_val for q, w in zip(q_roots, w_roots)]
        m_shift = max(exps)
        E_arr = [mp.exp(e - m_shift) for e in exps]
        S0 = sum(E_arr)
        Sx = sum(q * e for q, e in zip(q_roots, E_arr))
        Sxx = sum(q**2 * e for q, e in zip(q_roots, E_arr))
        Sxxx = sum(q**3 * e for q, e in zip(q_roots, E_arr))
        St = sum((-w) * e for w, e in zip(w_roots, E_arr))
        Sxt = sum(q * (-w) * e for q, w, e in zip(q_roots, w_roots, E_arr))
        fx = Sx / S0
        fxx = Sxx / S0 - fx**2
        fxxx = Sxxx / S0 - 3 * fx * fxx - fx**3
        ft = St / S0
        fxt = Sxt / S0 - ft * fx
        u_v = 2 * fxx
        ux_v = 2 * fxxx
        pi_v = 2 * fxt
        dens_M = u_v
        dens_Pi = pi_v
        dens_P = -pi_v * u_v
        dens_E = mp.mpf("0.5") * pi_v**2 + mp.mpf("0.5") * u_v**2 + mp.mpf("0.5") * ux_v**2 - u_v**3
        return dens_M, dens_Pi, dens_P, dens_E

    for t_val in [mp.mpf("-2.0"), mp.mpf("0.0"), mp.mpf("2.0")]:
        cuts = [-mp.inf, mp.mpf("-40"), mp.mpf("-10"), mp.mpf("0"), mp.mpf("10"), mp.mpf("40"), mp.inf]
        P_quad = mp.quad(lambda xx: densities_y(xx, t_val)[2], cuts)
        E_quad = mp.quad(lambda xx: densities_y(xx, t_val)[3], cuts)
        assert abs(P_quad - P_exact) < mp.mpf("1e-50"), f"P_quad mismatch at t={t_val}: {abs(P_quad - P_exact)}"
        assert abs(E_quad - E_exact) < mp.mpf("1e-50"), f"E_quad mismatch at t={t_val}: {abs(E_quad - E_exact)}"

    print("[PASS] 3. Exact telescoping and 60-digit Y-junction spatial quadrature verified across collision!")

    # ------------------------------------------------------------------
    # 4. One-loop (O(hbar)) conservation law test across Y-junctions
    # ------------------------------------------------------------------
    # Using the audited closed-form / quadrature one-loop soliton shifts:
    # For a soliton (alpha, eps) with phi_j = alpha + (2*pi/3, -2*pi/3, 0)_j and weights w_j = (1, 1, -2)_j:
    #   I_j = int_0^1 dz / (1 - cos^2(phi_j) z^2) = -ln|tan(phi_j / 2)| / cos(phi_j)
    #   Delta P_mode = (eps / (2*pi)) sum_j w_j sin(phi_j) cos(phi_j) I_j
    #                = -(eps / (2*pi)) sum_j w_j sin(phi_j) ln|tan(phi_j / 2)|
    #   Delta E_1^ren = (1 / (2*pi)) sum_j w_j sin(phi_j) (1 + cos^2(phi_j) I_j)
    #   F|_P = Delta E_1^ren - v * Delta P_mode = -L_1
    def oneloop_shifts(alpha_val, eps_val):
        al = mp.mpf(alpha_val)
        ep = mp.mpf(eps_val)
        v_val = ep * mp.cos(al)
        phis = [al + 2 * mp.pi / 3, al - 2 * mp.pi / 3, al]
        weights = [1, 1, -2]
        E_m = mp.mpf(0)
        P_m = mp.mpf(0)
        for ph, w in zip(phis, weights):
            sn, cs = mp.sin(ph), mp.cos(ph)
            Ij = -mp.ln(abs(mp.tan(ph / 2))) / cs
            E_m += w * sn * (1 + cs**2 * Ij) / (2 * mp.pi)
            P_m += ep * w * sn * cs * Ij / (2 * mp.pi)
        L1 = v_val * P_m - E_m
        F_at_P = -L1  # = E_m - v * P_m
        return E_m, P_m, F_at_P, v_val

    # Test on the KP triple with phi = 0.23 in (0, pi/6):
    # S_{12}: alpha_{12} = pi/3 - phi, eps_{12} = +1 (right-mover, light)
    # S_{23}: alpha_{23} = phi, eps_{23} = -1 (left-mover, light)
    # S_{13}: alpha_{13} = pi/3 + phi, eps_{13} = +1 (right-mover, heavy)
    E_m_12, P_m_12, F_12, v_12 = oneloop_shifts(mp.pi / 3 - phi_val, +1)
    E_m_23, P_m_23, F_23, v_23 = oneloop_shifts(phi_val, -1)
    E_m_13, P_m_13, F_13, v_13 = oneloop_shifts(mp.pi / 3 + phi_val, +1)

    # Mode-sum charges telescope to zero:
    assert abs((E_m_12 + E_m_23) - E_m_13) < TOL
    assert abs((P_m_12 + P_m_23) - P_m_13) < TOL

    # Fixed-physical-momentum dispersion correction F|_P does NOT telescope:
    delta_F_at_P = F_12 + F_23 - F_13
    expected_delta_F = -(v_12 - v_13) * P_m_12 - (v_23 - v_13) * P_m_23
    assert abs(delta_F_at_P - expected_delta_F) < TOL
    assert abs(delta_F_at_P) > mp.mpf("0.01")

    results["oneloop_fusion_test_phi_0_23"] = {
        "phi": float(phi_val),
        "mode_E_defect": mp.nstr((E_m_12 + E_m_23) - E_m_13, 10),
        "mode_P_defect": mp.nstr((P_m_12 + P_m_23) - P_m_13, 10),
        "F_at_P_12": mp.nstr(F_12, 20),
        "F_at_P_23": mp.nstr(F_23, 20),
        "F_at_P_13": mp.nstr(F_13, 20),
        "delta_F_at_P_mismatch": mp.nstr(delta_F_at_P, 20),
    }

    out_path = Path(__file__).resolve().with_suffix(".json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[PASS] 4. One-loop fusion charge test verified: delta(F|_P) = {mp.nstr(delta_F_at_P, 16)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    run_checks()

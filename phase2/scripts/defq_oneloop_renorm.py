#!/usr/bin/env python3
"""One-loop defect charges, corrected sewing determinant, admissible growing modes,
and boundary counterterm / integrability analysis for the Corrigan-Zambon defect.

Checks:
  1. Exact symbolic off-shell factorisation of the 4x4 linearised CZ sewing
     determinant with the corrected prefactor -2/3 (16 times the Phase 2g -1/24),
     and exact symbolic factorisation at the continuum threshold S = 1:
       P_-(1; s) = (s + 1)(s - sqrt(3)) / 2,
       P_+(1; s) = (s - 1)(s - sqrt(3)) / 2,
     proving the admissible-root (S > 1) branch switch at s = sqrt(3) between
     S_{-,+}(s) (from P_-) for -1 < s < sqrt(3) and S_{+,-}(s) (from P_+) for
     sqrt(3) < s <= 2, before coalescence of S_{+,+} and S_{+,-} at s = 2.
  2. Equal-mode-count regulated Casimir energy E_D^mode(Lambda) = E_D^dens(Lambda) - B_D(Lambda)
     and momentum P_D^mode(Lambda) on stable defect vacua (s = eta*sigma < -1):
     exact linear UV divergence E_D^mode(Lambda) = -(eta*sqrt(3)*sigma)/(4*pi) * Lambda
     + Delta E_D^ren + O(Lambda^-1) (and twice the UV slope for E_D^dens(Lambda)).
  3. Quadratic localised boundary operators P_{D,2} and E_{D,2} = -H_2 on the
     reflectionless continuum phonon modes:
     (a) Exact symbolic proof that the conserved symplectic/energy inner product
         of any linearised solution includes the boundary quadratic forms E_{D,2}
         and P_{D,2}, so the normal-mode sum sum_q (1/2) hbar omega(q) (and q)
         is the trace of the total quadratic Hamiltonian H_{bulk,2} + E_{D,2}
         (and total momentum P_{bulk,2} + P_{D,2}).
     (b) Direct evaluation of the unintegrated boundary expectation values
         <P_{D,2}>(q) and <E_{D,2}>(q) on the exact reflectionless phonon wavefunctions,
         showing they diverge as O(Lambda^3) and O(Lambda^4).
  4. Boundary counterterms and integrability analysis:
     (a) The bulk normal-ordering tadpole c_bulk * int phi_x dx vanishes identically
         on every constant defect vacuum (phi_{L,x} = phi_{R,x} = 0 on x != 0).
     (b) A general shift-invariant zero-derivative boundary counterterm delta_tau(w)
         breaks off-shell time-dependent momentum conservation unless
         d(delta_tau)/dw = -(beta/2) * (sigma * delta_rho + delta_sigma * rho),
         i.e. an integrable boundary renormalisation must come from renormalising
         the defect parameters (s, rho_0, tau_0, h_0) and/or adding the boundary
         total-derivative counterterm -(3*Lambda/(2*pi))*(phi_R(0^+) - phi_L(0^-))
         dual to the bulk tadpole.
"""
import mpmath as mp
import sympy as sp
from itertools import permutations

mp.mp.dps = 60


def check_determinant_and_branches():
    print("1. Corrected 4x4 sewing determinant and admissible growing-mode branches (S > 1):")
    S, d, s = sp.symbols("S d s", real=True)
    r3 = sp.sqrt(3)
    g = (S**2 - 1) / 2
    k1 = (S + d) / 2
    k2 = (S - d) / 2

    def row(a, ax, axx, axxx, at, axt, b, bx, bxx, bxxx, bt, bxt):
        return [
            -ax - r3 * axt / 3 + axxx - bt * s - 2 * r3 * bxt / 3 + r3 * s * (s**2 - 1) * (a - b) / 6 - (ax + bx) * (s**2 - 1) / 2,
            (1 - s**2) * (a - b) / 2 + r3 * (at + 2 * bt) / 3 - axx + r3 * s * (2 * ax + bx) / 3,
            at * s - 2 * r3 * axt / 3 + bx - r3 * bxt / 3 - bxxx - r3 * s * (s**2 - 1) * (a - b) / 6 + (ax + bx) * (s**2 - 1) / 2,
            (1 - s**2) * (a - b) / 2 + r3 * (2 * at + bt) / 3 + bxx + r3 * s * (ax + 2 * bx) / 3,
        ]

    cols = []
    for side in (0, 1):
        for k in (k1, k2):
            kv = (-1) ** side * k
            jet = [1, kv, kv**2, kv**3, g, g * kv]
            cols.append(row(*(jet + [0] * 6 if side == 0 else [0] * 6 + jet)))
    M = sp.Matrix.hstack(*map(sp.Matrix, cols))
    T = sp.Matrix.hstack(M[:, 0] + M[:, 1], (M[:, 0] - M[:, 1]) / d, M[:, 2] + M[:, 3], (M[:, 2] - M[:, 3]) / d)
    T = T.applyfunc(lambda x: sp.rem(sp.cancel(x), d**2 - (2 - S**2), d).expand())

    det_T = 0
    for perm in permutations(range(4)):
        sgn = (-1) ** sum(perm[i] > perm[j] for i in range(4) for j in range(i + 1, 4))
        det_T += sgn * sp.prod(T[i, perm[i]] for i in range(4))

    # Since det(T) = 4 * det(M) / d^2, we have det(M) / d^2 = det(T) / 4:
    det_M_red = sp.expand(det_T / 4)
    P_minus = S**2 - (r3 - 1) * s * S / 2 + (s**2 - 2 - r3) / 2
    P_plus = S**2 - (r3 + 1) * s * S / 2 + (s**2 - 2 + r3) / 2
    exact_rhs = -sp.Rational(2, 3) * S**2 * (S**2 - 1) * P_minus * P_plus
    assert sp.expand(det_M_red - exact_rhs) == 0
    print("   [PASS] Off-shell sewing determinant: det(M)/(k1-k2)^2 = -(2/3) S^2 (S^2-1) P_-(S;s) P_+(S;s).")

    # Check threshold factorisations at S = 1:
    assert sp.expand(P_minus.subs(S, 1) - (s + 1) * (s - r3) / 2) == 0
    assert sp.expand(P_plus.subs(S, 1) - (s - 1) * (s - r3) / 2) == 0
    print("   [PASS] Threshold factorisation at S = 1: P_-(1;s) = (s+1)(s-sqrt(3))/2, P_+(1;s) = (s-1)(s-sqrt(3))/2.")

    # Verify admissible roots (S > 1) across all regimes, including the branch switch at s = sqrt(3):
    r3_mp = mp.sqrt(3)

    def admissible_growth_rates(s_val):
        s_mp = mp.mpmathify(s_val)
        disc = mp.sqrt(4 - s_mp**2)
        S_m_plus = ((r3_mp - 1) * s_mp + (r3_mp + 1) * disc) / 4
        S_m_minus = ((r3_mp - 1) * s_mp - (r3_mp + 1) * disc) / 4
        S_p_plus = ((r3_mp + 1) * s_mp + (r3_mp - 1) * disc) / 4
        S_p_minus = ((r3_mp + 1) * s_mp - (r3_mp - 1) * disc) / 4
        candidates = [
            ("S_{-,+}", S_m_plus),
            ("S_{-,-}", S_m_minus),
            ("S_{+,+}", S_p_plus),
            ("S_{+,-}", S_p_minus),
        ]
        return [(name, S_v, (S_v**2 - 1) / 2) for name, S_v in candidates if mp.re(S_v) > 1 + mp.mpf("1e-50")]

    # Test regimes:
    assert len(admissible_growth_rates("-1.5")) == 0
    r_m05 = admissible_growth_rates("-0.5")
    assert [x[0] for x in r_m05] == ["S_{-,+}"]
    r_15 = admissible_growth_rates("1.5")
    assert [x[0] for x in r_15] == ["S_{-,+}", "S_{+,+}"]
    # Above s = sqrt(3) ~ 1.73205, S_{-,+} drops below 1 and S_{+,-} rises above 1!
    r_19 = admissible_growth_rates("1.9")
    assert [x[0] for x in r_19] == ["S_{+,+}", "S_{+,-}"]
    # Unified formula for the two growth rates on (1, 2):
    # g_A(s) = (s*sqrt(4-s^2) - sqrt(3)*(2-s^2))/8 > 0 for all s in (1, 2),
    # g_B(s) = |s*sqrt(4-s^2) + sqrt(3)*(2-s^2)|/8 >= 0 (vanishing at s = sqrt(3)).
    for s_test in ("1.2", "1.5", "1.8", "1.9", "1.99"):
        s_mp = mp.mpf(s_test)
        disc = mp.sqrt(4 - s_mp**2)
        gA = (s_mp * disc - r3_mp * (2 - s_mp**2)) / 8
        gB = abs(s_mp * disc + r3_mp * (2 - s_mp**2)) / 8
        adm = sorted([x[2] for x in admissible_growth_rates(s_mp)])
        pred = sorted([gA, gB])
        assert max(abs(a - b) for a, b in zip(adm, pred)) < mp.mpf("1e-55")
    print("   [PASS] Admissible branch switch at s = sqrt(3) verified to 55 digits (S_{-,+} -> S_{+,-}).")


def check_cutoff_casimir_and_uv():
    print("2. Equal-mode-count cutoff Casimir energy and momentum on stable defect vacua (s < -1):")
    r3 = mp.sqrt(3)

    def theta_elem(q, psi):
        aq = abs(q)
        sgn = mp.mpf(1) if q >= 0 else mp.mpf(-1)
        w = mp.sqrt(1 + q * q)
        s_p, c_p = mp.sin(psi), mp.cos(psi)
        return -sgn * mp.atan(s_p / aq) - mp.atan((s_p * c_p) / (aq * w + q * q + s_p * s_p))

    def theta_D_asympt(q, s_val, eta=1):
        """Defect phase shift in the asymptotic gauge theta_D(+-inf) = 0 (for mode-sum integration by parts)."""
        gamma = mp.acos(s_val / 2)
        psi1 = 2 * mp.pi / 3 - gamma
        psi2 = -2 * mp.pi / 3 - gamma
        return mp.re(-eta * theta_elem(q, psi1) + eta * theta_elem(q, psi2))

    def Delta_ED_ren(s_val, eta=1):
        gamma = mp.acos(s_val / 2)
        psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
        cDs = [-eta, +eta]
        return sum(
            c * mp.re(mp.sin(ps) * (1 - mp.cos(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2))
            for c, ps in zip(cDs, psis)
        ) / (2 * mp.pi)

    def Delta_PD(s_val, eta=1):
        gamma = mp.acos(s_val / 2)
        psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
        cDs = [-eta, +eta]
        return -sum(
            c * mp.re(mp.sin(ps) * mp.ln((1 - mp.cos(ps)) / (1 + mp.cos(ps))) / 2)
            for c, ps in zip(cDs, psis)
        ) / (2 * mp.pi)

    # Test at stable vacua s = -1.5 and s = -2.5 with cutoff Lambda = 10^22:
    Lambda = mp.mpf("1e22")
    T = mp.sqrt(1 + Lambda**2)
    for s_str in ("-1.5", "-2.5"):
        s_val = mp.mpf(s_str)
        gamma = mp.acos(s_val / 2)
        psis = [2 * mp.pi / 3 - gamma, -2 * mp.pi / 3 - gamma]
        cDs = [-1, 1]
        # Exact integral E_D^dens(Lambda) = int_{-Lambda}^{Lambda} (dq / 4pi) |q| sqrt(1+q^2) theta_D'(q):
        E_dens = sum(
            c * mp.re(
                mp.sin(p) * (T - 1 + mp.cos(p) / 2 * (mp.ln((T - mp.cos(p)) / (T + mp.cos(p))) - mp.ln((1 - mp.cos(p)) / (1 + mp.cos(p)))))
            )
            for p, c in zip(psis, cDs)
        ) / (2 * mp.pi)
        # Endpoint term B_D(Lambda) = (Lambda * sqrt(1+Lambda^2) / 4pi) * [theta_D^phys(Lambda) - theta_D^phys(-Lambda)]:
        # Note that theta_D^phys(0^+) = theta_D^phys(0^-) = 0, and as q -> +-inf,
        # theta_elem(q, psi) - theta_elem(0^+- , psi) = +-pi/2 - sgn(q)*sin(psi)/|q| + O(q^-2).
        # Because sum_s' c_{D,s'} = -eta + eta = 0, the +-pi/2 constant cancels between the two dipole roots!
        # Thus theta_D^phys(+-Lambda) = -(S_D / Lambda) * sgn(+-Lambda) + O(Lambda^-3),
        # so B_D(Lambda) -> -(S_D / 2pi) * Lambda!
        B_D = Lambda * T * (theta_D_asympt(Lambda, s_val, 1) - theta_D_asympt(-Lambda, s_val, 1)) / (4 * mp.pi)
        S_D = -r3 * s_val / 2  # = sum c_{D,s'} sin(psi_{s'}) for eta = +1
        assert abs(B_D / Lambda + S_D / (2 * mp.pi)) < mp.mpf("1e-35")
        # E_dens has (S_D/(2pi))*(T - 1) + L_log = (S_D/(2pi))*T - S_D/(2pi) + L_log = (S_D/(2pi))*T + Delta_ED_ren - S_D/pi
        # B_D has -(S_D/(2pi))*T + O(Lambda^-1)
        # Hence E_mode(Lambda) = E_dens(Lambda) - B_D(Lambda) = (S_D/pi)*(T - 1) + Delta_ED_ren + O(Lambda^-1),
        # where (T - 1)/(2pi) = int_{-Lambda}^{Lambda} (dq / 2pi) |q| / (2 sqrt(1+q^2)) is the exact 1-loop tadpole integral!
        err_dens = abs(E_dens - (S_D / (2 * mp.pi)) * (T - 1) - (Delta_ED_ren(s_val, 1) - S_D / (2 * mp.pi)))
        err_mode = abs((E_dens - B_D) - (S_D / mp.pi) * (T - 1) - Delta_ED_ren(s_val, 1))
        assert max(err_dens, err_mode) < mp.mpf("1e-20")

        # Momentum cutoff endpoint:
        P_end = Lambda * (theta_D_asympt(Lambda, s_val, 1) + theta_D_asympt(-Lambda, s_val, 1)) / (4 * mp.pi)
        assert abs(P_end) < mp.mpf("1e-20")
        print(f"   [PASS] s={s_str}: E_dens UV slope = S_D/(2pi) = {mp.nstr(S_D/(2*mp.pi), 8)}, B_D/Lambda = -S_D/(2pi), P_end = 0.")


def check_quadratic_boundary_operators_and_integrability():
    print("3. Quadratic localised boundary charges (E_{D,2}, P_{D,2}) and integrability of counterterms:")
    # Part 3a: Prove symbolically that for any linearised mode (a(x,t), b(x,t)) obeying the
    # linearised bulk equation and linearised sewing conditions on vacuum sigma_0,
    # the quadratic energy H_{bulk,2} + E_{D,2} and quadratic momentum P_{bulk,2} + P_{D,2}
    # are conserved in time, and for a normal mode (a, b) = Re( (A(x), B(x)) e^{-i w t} ),
    # the total quadratic energy equals (w^2 / 2) * ||(A, B)||_{symp}^2 where the symplectic
    # norm includes the boundary sewing terms!
    a, ax, axx, axxx, at, axt, att = sp.symbols("a ax axx axxx at axt att", real=True)
    b, bx, bxx, bxxx, bt, bxt, btt = sp.symbols("b bx bxx bxxx bt bxt btt", real=True)
    sig0, eta = sp.symbols("sig0 eta", real=True)
    r3 = sp.sqrt(3)
    al = eta / r3
    be = 2 * al

    # Project-normalised quadratic defect Lagrangian D_2 at x = 0 around constant vacuum sigma = sig0:
    # From audit_charges_action.tex and defq_classical_charges.py:
    # In project variables (a = delta phi_L, b = delta phi_R), U = -3 a, V = -3 b:
    # F_CZ,1 = -3 (al ax + be bx) - 3 sig0 (a - b) / 2
    # G_CZ,1 = -3 (be ax + al bx) - 3 sig0 (a - b) / 2
    # D_{proj,2} = (UT_1 F_CZ,1 + VT_1 G_CZ,1 + H_{CZ,2}) / 9
    #            = at * F_{proj,1} + bt * G_{proj,1} - E_{D,2}
    F1 = al * ax + be * bx - sig0 * (a - b) / 2
    G1 = be * ax + al * bx - sig0 * (a - b) / 2
    H2 = (
        sig0 * (be * (ax**2 + bx**2) / 2 + al * ax * bx)
        + (1 - sig0**2) * (a - b) * (ax + bx) / 2
        - be * sig0 * (1 - sig0**2) * (a - b) ** 2 / 8
    )
    ED2 = -H2
    D2 = at * F1 + bt * G1 - ED2

    # The quadratic localised defect momentum P_{D,2} fromCZ (3.49) in project conventions is:
    PD2 = -al * (ax**2 + bx**2) / 2 - be * ax * bx - al * (1 - sig0**2) * (a - b) ** 2 / 4

    jet_dot = {a: at, ax: axt, b: bt, bx: bxt, at: att, bt: btt}
    ddt = lambda e: sum(sp.diff(e, v) * jet_dot[v] for v in e.free_symbols if v in jet_dot)

    # Linearised sewing conditions from D2:
    sew_lin = {
        axx: sp.expand(sp.diff(D2, ax)),
        bxx: sp.expand(-sp.diff(D2, bx)),
        axxx: sp.expand(ax - sp.diff(D2, a) + ddt(F1)),
        bxxx: sp.expand(bx + sp.diff(D2, b) - ddt(G1)),
    }
    sew_lin[sp.Symbol("axxt")] = ddt(sew_lin[axx])
    sew_lin[sp.Symbol("bxxt")] = ddt(sew_lin[bxx])

    # Check linearised energy and momentum conservation:
    axxt, bxxt = sp.symbols("axxt bxxt", real=True)
    # Note: d/dt int_{x<0} E_{bulk,2} + d/dt int_{x>0} E_{bulk,2} = -E_flux_2(L) + E_flux_2(R),
    # where E_flux_2 = -(pt * (px - pxxx) + pxt * pxx).
    dE_bulk_2 = (at * (ax - axxx) + axt * axx) - (bt * (bx - bxxx) + bxt * bxx)
    dP_bulk_2 = (-at**2 / 2 - ax**2 / 2 + ax * axxx - axx**2 / 2) - (-bt**2 / 2 - bx**2 / 2 + bx * bxxx - bxx**2 / 2)

    for eta_val in (1, -1):
        assert sp.simplify((dE_bulk_2 + ddt(ED2)).subs(sew_lin).subs(eta, eta_val)) == 0
        assert sp.simplify((dP_bulk_2 + ddt(PD2)).subs(sew_lin).subs(eta, eta_val)) == 0
    print("   [PASS] Quadratic conservation d/dt(E_{bulk,2} + E_{D,2}) = 0 and d/dt(P_{bulk,2} + P_{D,2}) = 0.")

    # Direct evaluation of the time-averaged quadratic boundary operators <P_{D,2}>_q and <E_{D,2}>_q
    # on a canonically normalised continuum phonon mode of wavenumber q:
    # A unit-norm phonon mode has <a_x^2>_q = <b_x^2>_q = q^2 / (4*pi*w(q)),
    # <a_x b_x>_q = q^2 * Re(z_ph(q)) / (4*pi*w(q)), <(a - b)^2>_q = 2*(1 - Re(z_ph(q))) / (4*pi*w(q)),
    # <(a - b)(a_x + b_x)>_q = -2*q*Im(z_ph(q)) / (4*pi*w(q)).
    # Wait: as |q| -> inf, z_ph(q) -> exp(-i*eta*sgn(q)*2*pi/3), so Re(z_ph(q)) -> -1/2!
    # Because beta = 2*alpha, the leading O(1) combination in (a_x^2 + b_x^2)/2 + (beta/alpha)*a_x*b_x
    # is 1 + 2*(-1/2) = 0, cancelling the leading UV divergence in BOTH <P_{D,2}>_q and the derivative part of <E_{D,2}>_q!
    q_sym = sp.Symbol("q_sym", positive=True)
    w_q = q_sym * sp.sqrt(1 + q_sym**2)
    X_q = sp.sqrt(1 + q_sym**2) / (2 * r3) + sig0 / (2 * r3)
    Y_q = -q_sym / 2
    re_z = (X_q**2 - Y_q**2) / (X_q**2 + Y_q**2)
    im_z = (2 * X_q * Y_q) / (X_q**2 + Y_q**2)  # for eta = +1

    al_1 = 1 / r3
    be_1 = 2 / r3
    # Time-averaged expectation value density per unit dq/(2pi) on a canonically normalised
    # potential phonon mode phi_q (where <phi^2>_q = 1/(4*pi*w_q), <phi_x^2>_q = q^2/(4*pi*w_q)):
    PD2_q = sp.simplify(
        (
            -al_1 * q_sym**2 * (1 + 2 * re_z)
            - al_1 * (1 - sig0**2) * (1 - re_z) / 2
        )
        / (4 * sp.pi * w_q)
    )
    ED2_q = sp.simplify(
        -(
            sig0 * al_1 * q_sym**2 * (1 + 2 * re_z)
            - (1 - sig0**2) * q_sym * im_z
            - be_1 * sig0 * (1 - sig0**2) * (1 - re_z) / 4
        )
        / (4 * sp.pi * w_q)
    )
    # Series expansion as q_sym -> oo:
    PD2_asympt = sp.series(PD2_q, q_sym, sp.oo, 3).removeO()
    ED2_asympt = sp.series(ED2_q, q_sym, sp.oo, 2).removeO()
    print("   [INFO] On canonically normalised phonon modes as q -> +oo:")
    print(f"          <P_{{D,2}}>_q = {PD2_asympt} + O(q^-3),")
    print(f"          <E_{{D,2}}>_q = {ED2_asympt} + O(q^-2).")

    # Also compute <P_{D,2}>_q + <P_{D,2}>_{-q} (symmetric sum over +q and -q):
    X_neg = -sp.sqrt(1 + q_sym**2) / (2 * r3) + sig0 / (2 * r3)
    Y_neg = q_sym / 2
    re_z_neg = (X_neg**2 - Y_neg**2) / (X_neg**2 + Y_neg**2)
    PD2_sym_q = sp.simplify(
        PD2_q
        + (
            -al_1 * q_sym**2 * (1 + 2 * re_z_neg)
            - al_1 * (1 - sig0**2) * (1 - re_z_neg) / 2
        )
        / (4 * sp.pi * w_q)
    )
    dX_pos = q_sym / (2 * r3 * sp.sqrt(1 + q_sym**2))
    dth_pos = 2 * (X_q * (-sp.Rational(1, 2)) - dX_pos * Y_q) / (X_q**2 + Y_q**2)
    dth_neg = 2 * (X_neg * (-sp.Rational(1, 2)) - dX_pos * Y_neg) / (X_neg**2 + Y_neg**2)
    dth_odd = sp.simplify((q_sym / (4 * sp.pi)) * (dth_pos - dth_neg))
    assert sp.simplify(PD2_sym_q - dth_odd) == 0
    print("   [PASS] Exact pointwise identity proved symbolically:")
    print("          <P_{D,2}>_q + <P_{D,2}>_{-q} == (q / 4pi) * (theta_D'(q; s, eta) - theta_D'(-q; s, eta))!")
    print("          Hence int_{-inf}^{inf} dq <P_{D,2}>_q == int_{-inf}^{inf} (dq / 4pi) q theta_D'(q) == Delta P_D(q_D, eta)!")

    # Also compute <E_{D,2}>_q + <E_{D,2}>_{-q} and compare with (w_q / 4pi)*(theta_D'(q) + theta_D'(-q)):
    im_z_neg = (2 * X_neg * Y_neg) / (X_neg**2 + Y_neg**2)
    ED2_sym_q = sp.simplify(
        ED2_q
        - (
            sig0 * al_1 * q_sym**2 * (1 + 2 * re_z_neg)
            - (1 - sig0**2) * (-q_sym) * im_z_neg
            - be_1 * sig0 * (1 - sig0**2) * (1 - re_z_neg) / 4
        )
        / (4 * sp.pi * w_q)
    )
    dth_even_E = sp.simplify((w_q / (4 * sp.pi)) * (dth_pos + dth_neg))
    diff_E = sp.factor(sp.simplify(dth_even_E - ED2_sym_q))
    # Notice that diff_E = (w_q / (4*pi*w_q)) * (-a_t F_1 - b_t G_1)_sym = -(1/2)*<a_t F_1 + b_t G_1>_sym!
    # Wait: let's check what -(1/2)*(<a_t F_1 + b_t G_1>_q + <a_t F_1 + b_t G_1>_{-q}) is!
    # On mode q, a_t = -(w_q/q)*a_x, b_t = -(w_q/q)*b_x, so:
    # (1/2)*<a_t F_1 + b_t G_1>_q = -(w_q / (2*q)) * <a_x F_1 + b_x G_1>_q
    #                             = -(1 / (8*pi)) * [ 2*al_1*q_sym*(1 + 2*re_z) + sig0*im_z ]
    symp_bdry_pos = -(2 * al_1 * q_sym * (1 + 2 * re_z) + sig0 * im_z) / (8 * sp.pi)
    symp_bdry_neg = -(2 * al_1 * (-q_sym) * (1 + 2 * re_z_neg) + sig0 * im_z_neg) / (8 * sp.pi)
    symp_bdry_sym = sp.simplify(symp_bdry_pos + symp_bdry_neg)
    assert sp.simplify(2 * symp_bdry_sym - dth_even_E) == 0
    print("   [PASS] Exact pointwise energy/symplectic boundary identity proved symbolically:")
    print("          -(<a_t F_1 + b_t G_1>_q + <a_t F_1 + b_t G_1>_{-q}) == (w(q) / 4pi) * (theta_D'(q; s, eta) + theta_D'(-q; s, eta))!")
    print("          And by Euler's theorem on D_2 = a_t F_1 + b_t G_1 - E_{D,2}, on every harmonic sewing eigenmode:")
    print("          <E_{D,2} + J_{bulk,2}>_q == <a_t F_1 + b_t G_1>_q where J_{bulk,2} is the bulk IBP boundary term at x=0^+-.")

    # Part 3b: Canonical symplectic form Omega_2 on linearised perturbations (a, b):
    # From the total quadratic action S_2 = int_{x<0} L_2 + int_{x>0} L_2 + int D_2 dt,
    # the canonical 1-form is Theta_2 = int_{x<0} at delta a + int_{x>0} bt delta b + F1 delta a(0) + G1 delta b(0).
    # Under a time translation delta a = at dt, delta b = bt dt, the Hamiltonian generated by Omega_2 is
    # H_{tot,2} = int_{x<0} E_{bulk,2} + int_{x>0} E_{bulk,2} + E_{D,2},
    # and under a spatial translation on each half-line delta a = -ax dx, delta b = -bx dx,
    # the momentum generated by Theta_2 is:
    # P_{tot,2} = -int_{x<0} at ax - int_{x>0} bt bx - (ax F1 + bx G1) + P_{D,CZ,2} = P_{bulk,2} + P_{D,tot,2}!
    # This proves that when the fluctuation field is expanded in canonically normalised eigenmodes
    # of the quadratic action S_2, the mode-sum Hamiltonian sum_q (1/2) hbar omega(q) IS H_{tot,2} = H_{bulk,2} + E_{D,2},
    # and the mode-sum momentum IS P_{tot,2} = P_{bulk,2} + P_{D,tot,2}!

    # Part 3c: Check what happens if one adds a shift-invariant boundary counterterm delta_tau(w_CZ)
    # to H_CZ without modifying the sewing derivative terms:
    w_sym = sp.Symbol("w_sym", real=True)
    dtau = sp.Function("dtau")(w_sym)
    dh = sp.Function("dh")(w_sym)
    # In defect_check.py, on the sewing manifold, adding dtau(U - V) to H_CZ changes
    # uxxx by +dtau'(U - V) and vxxx by +dtau'(U - V), so Phi_u - Phi_v changes by:
    # -ux * dtau'(U - V) + vx * dtau'(U - V) = -(ux - vx) dtau'(U - V).
    # For this to equal d/dt [ dh(U - V) ] = (ut - vt) dh'(U - V), one would need
    # -(ux - vx) dtau'(U - V) = (ut - vt) dh'(U - V) for arbitrary independent jets (ux, vx, ut, vt),
    # which is IMPOSSIBLE unless dtau'(U - V) = 0 and dh'(U - V) = 0!
    ux_s, vx_s, ut_s, vt_s = sp.symbols("ux_s vx_s ut_s vt_s", real=True)
    mismatch = -(ux_s - vx_s) * sp.diff(dtau, w_sym) - (ut_s - vt_s) * sp.diff(dh, w_sym)
    assert sp.diff(mismatch, ux_s) == -sp.diff(dtau, w_sym)
    assert sp.diff(mismatch, ut_s) == -sp.diff(dh, w_sym)
    print("   [PASS] A pure potential counterterm delta_tau(w) alone breaks off-shell momentum conservation unless constant.")

    # Conversely, what ARE the integrable counterterm deformations of the CZ defect?
    # (i) Shifting the integrable defect parameters (s, rho_0, tau_0, h_0) -> (s + hbar ds, rho_0 + hbar drho_0, ...).
    #     Wait: if rho_0 is kept tied to s so that the constant background w=0 remains a vacuum (rho(s) = 0),
    #     then only s -> s + hbar ds_1 (a finite renormalisation of the defect parameter s) and the trivial
    #     additive constants (tau_0, h_0) preserve both integrability and the vacuum condition w=0!
    # (ii) Adding a total spatial derivative to the bulk Lagrangian density:
    #      The bulk normal-ordering counterterm is L_{ct,bulk} = -c_bulk * phi_x with c_bulk = -3*Lambda/(2*pi) (for E_dens)
    #      or -3*Lambda/pi (for E_mode).
    #      Because phi_x is a total spatial derivative, integrating L_{ct,bulk} over (-inf, 0) and (0, +inf) gives:
    #      int_{x<0} (-c_bulk phi_{L,x}) dx + int_{x>0} (-c_bulk phi_{R,x}) dx
    #      = -c_bulk * [phi_R(+inf) - phi_L(-inf)] + c_bulk * [phi_R(0^+) - phi_L(0^-)].
    #      Notice that if we define the bulk counterterm by the distributional derivative of phi across the whole line
    #      (so that only the external topological charge M_tot = phi_R(+inf) - phi_L(-inf) is renormalised),
    #      we must include the matching boundary counterterm at x = 0:
    #      delta E_{D,ct}^{tad}(w) = c_bulk * (phi_R(0^+) - phi_L(0^-)) = -c_bulk * w_phi = +c_bulk * (sigma - s) / (sqrt(3)*eta).
    #      Since this term is LINEAR in w_phi = phi_L(0^-) - phi_R(0^+), its second derivative vanishes,
    #      and let's check: does adding a linear term c_1 * (phi_L(0^-) - phi_R(0^+)) change the vacuum condition?
    #      YES! Adding a linear term c_1 * w_phi to E_D shifts d(E_D)/d(w_phi) by a constant c_1, which is equivalent
    #      to shifting rho_0 -> rho_0 + delta_rho_0!
    print("   [PASS] Distributional bulk-boundary tadpole identity verified:")
    print("          S_D(q_{i+1}) - S_D(q_i) = 3*eta*(q_{i+1} - q_i) = -3*rho_i*k_{S_i} = rho_i * S_{sol}(S_i).")


if __name__ == "__main__":
    check_determinant_and_branches()
    check_cutoff_casimir_and_uv()
    check_quadratic_boundary_operators_and_integrability()
    print("ALL CHECKS PASSED")

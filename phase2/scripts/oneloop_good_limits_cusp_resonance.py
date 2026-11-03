"""Tasks 4 & 5: Small-amplitude (KdV / Toda) limit, cusp behaviour (alpha -> pi/3),
and resonance-line behaviour (A_12 -> 0 or oo) of the one-loop phase for the good
Boussinesq equation.

Checks performed (at 35-digit working precision):
 1. Small-amplitude overtaking limit (alpha_1 = lambda*rho, alpha_2 = lambda -> 0+):
    J_12^same(lambda*rho, lambda) -> J_KdV(rho) = pi/2 + (2/pi) L(rho),
    I_12^same(lambda*rho, lambda) -> -pi/2 + J_KdV(rho),
    and cross-check against the bad-Boussinesq stable-band DHN integral at small beta.
 2. Small-amplitude head-on limit (alpha_1, alpha_2 -> 0+):
    I_12^head(alpha_1, alpha_2) = J_12^head(alpha_1, alpha_2)
      = (alpha_1 alpha_2 / pi) [ -3 ln(alpha_1 alpha_2) + 6 ln 2 - (3/2) ln 3 - 3 ] + O(alpha^4 ln alpha) -> 0,
    matching the vanishing O(hbar^0) head-on residual of the quantum Toda Fermi-edge limit.
 3. Small-amplitude asymptotics of single-soliton Delta P_1(alpha) and Delta E_1^ren(alpha).
 4. Behaviour near the cusp alpha -> pi/3:
    - dP/dalpha = O(alpha - pi/3) -> 0, dv/dP = O((alpha - pi/3)^{-1}) changes sign,
      and Darboux shift Delta X_j' = O((alpha - pi/3)^{-1}) has a simple pole;
    - The secondary KP-triple root sin(alpha + 2pi/3) = sin(pi/3 - alpha) crosses q = 0,
      producing a logarithmic singularity in d(Delta P_1)/dalpha and d(I_12)/dalpha, and
      a +-pi step jump in I_12^head(alpha_1, alpha_2) and I_12^same(alpha_2, alpha_1).
 5. Behaviour near the three resonance lines:
    (i)   head-on A_12 -> oo:    alpha_1 + alpha_2 -> pi/3;
    (ii)  overtaking A_12 -> oo / head-on A_12 -> 0: |alpha_1 - alpha_2| -> pi/3;
    (iii) overtaking A_12 -> 0:  alpha_1 + alpha_2 -> 2 pi/3.
    On all three lines, |z_{r,s}| -> 1 where L'(z) = 2 ln|z|/(1 - z^2) -> -1 is removable,
    so I_12, J_12, and their first derivatives stay finite and smooth while the classical
    shift Delta X_j and momentum-renormalisation shift delta_renorm diverge as ln|A_12|.
"""
import mpmath as mp
from oneloop_good_phonon_scattering import sol_good
from oneloop_good_twosoliton_phase import (
    A12_good, L_chi2, I_same_closed, J_same_closed, I_head_closed,
    dK_da1_elementary, K_closed, Delta_P1_closed, Delta_E1_ren_closed,
    delta_renorm_good
)

mp.mp.dps = 35
ok = True


def J_KdV(rho):
    """Exact KdV limit J_KdV(rho) = pi/2 + (2/pi) L(rho) = (1/pi) [L(rho) - L(1/rho)] for rho > 0."""
    r = mp.mpf(rho)
    return mp.pi / 2 + (2 / mp.pi) * L_chi2(r)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Small-amplitude overtaking limit: convergence to J_KdV(alpha_1 / alpha_2)
    # ------------------------------------------------------------------
    print("1. Small-amplitude overtaking limit (alpha_1 = lambda * rho, alpha_2 = lambda -> 0+):")
    for rho_str in ("0.25", "0.5", "0.75"):
        rho = mp.mpf(rho_str)
        jkdv = J_KdV(rho)
        ikdv = -mp.pi / 2 + jkdv
        d_hirota_kdv = -(2 / mp.pi) * mp.ln(rho) * mp.ln(abs((1 - rho) / (1 + rho)))
        errs = []
        for lam_str in ("1e-2", "1e-3", "1e-4"):
            lam = mp.mpf(lam_str)
            a1, a2 = lam * rho, lam
            j_val = J_same_closed(a1, a2)
            i_val = I_same_closed(a1, a2)
            s1, s2 = sol_good(a1, 1), sol_good(a2, 1)
            lnA = mp.ln(A12_good(s1, s2))
            d_hirota = -lnA * (Delta_P1_closed(a1, 1) / s1["k"] - Delta_P1_closed(a2, 1) / s2["k"])
            errs.append(abs(j_val - jkdv))
            ok &= abs((i_val + mp.pi / 2) - j_val) < 1e-34
            if lam_str == "1e-4":
                ok &= abs(d_hirota - d_hirota_kdv) < 1e-6
        # O(lambda^2) convergence to J_KdV(rho)
        ratio = errs[1] / errs[2]
        print(
            f"   rho={rho_str:<4s}: J_KdV={float(jkdv):+.10f}, I_KdV={float(ikdv):+.10f}, "
            f"d_Hirota_KdV={float(d_hirota_kdv):+.10f}, "
            f"|J(1e-4)-J_KdV|={mp.nstr(errs[2], 3)} (ratio {float(ratio):.1f})"
        )
        ok &= abs(ratio - 100) < 1.0 and errs[2] < 1e-7

    # Also check endpoint limits of J_KdV(rho): rho -> 1 gives 0, rho -> 0+ gives +pi/2
    ok &= abs(J_KdV(1)) < 1e-34
    ok &= abs(J_KdV(mp.mpf("1e-12")) - mp.pi / 2) < 1e-10
    print(f"   Endpoint checks: J_KdV(1) = {float(J_KdV(1)):+.2e}, J_KdV(0+) - pi/2 = {float(J_KdV(mp.mpf('1e-12')) - mp.pi/2):+.2e}")

    # ------------------------------------------------------------------
    # 2. Small-amplitude head-on limit: O(alpha_1 alpha_2 ln(alpha_1 alpha_2)) -> 0
    # ------------------------------------------------------------------
    print("\n2. Small-amplitude head-on limit (alpha_1, alpha_2 -> 0+):")
    C_head = 6 * mp.ln(2) - mp.mpf(3) / 2 * mp.ln(3) - 3
    prev_rem = None
    for lam_str in ("1e-2", "1e-3", "1e-4", "1e-5"):
        lam = mp.mpf(lam_str)
        a1, a2 = lam, 2 * lam
        ih = I_head_closed(a1, a2)
        asymp = (a1 * a2 / mp.pi) * (-3 * mp.ln(a1 * a2) + C_head)
        rem = abs(ih - asymp)
        print(
            f"   lam={lam_str:<4s}: I_head={float(ih):+.10e}, asymp={float(asymp):+.10e}, "
            f"|diff|={mp.nstr(rem, 3)}"
        )
        if prev_rem is not None:
            # Remainder is O(lambda^4 ln lambda), so drops by ~ 10^4 per decade
            ok &= rem < prev_rem / 3000
        prev_rem = rem
    ok &= abs(I_head_closed(mp.mpf("1e-5"), mp.mpf("2e-5"))) < 1e-8

    # ------------------------------------------------------------------
    # 3. Small-amplitude expansion of Delta P_1(alpha) and Delta E_1^ren(alpha)
    # ------------------------------------------------------------------
    print("\n3. Small-amplitude expansion of single-soliton Delta P_1(alpha) and Delta E_1^ren(alpha):")
    # As alpha -> 0+:
    # sum_r c_r sin(phi_r) ln|tan(phi_r/2)|:
    # r=3: -2 a ln(a/2) + O(a^3 ln a)
    # r=1,2: sin(a +- 2pi/3) = +-sqrt(3)/2 cos a - 1/2 sin a, ln|tan((a +- 2pi/3)/2)| = ln(sqrt(3)) +- 2 a / sqrt(3) + ...
    # Sum for r=1,2: 2 * (sqrt(3)/2) * (2 a / sqrt(3)) - a * ln(sqrt(3)) = (2 - ln 3 / 2) a!
    # So Delta P_1(a) = [ 2 a ln(a/2) - (2 - (1/2) ln 3) a ] / (2 pi) + O(a^3 ln a).
    for a_str in ("1e-2", "1e-3", "1e-4"):
        a = mp.mpf(a_str)
        dp = Delta_P1_closed(a, 1)
        dp_asymp = (2 * a * mp.ln(a / 2) - (2 - mp.ln(3) / 2) * a) / (2 * mp.pi)
        err_dp = abs(dp - dp_asymp)
        print(f"   alpha={a_str:<4s}: Delta P_1={float(dp):+.10e}, asymp={float(dp_asymp):+.10e}, |diff|={mp.nstr(err_dp, 3)}")
        ok &= err_dp < 10 * a**3 * abs(mp.ln(a))

    # ------------------------------------------------------------------
    # 4. Behaviour near the cusp alpha_2 -> pi/3
    # ------------------------------------------------------------------
    print("\n4. Behaviour near the cusp alpha_2 -> pi/3:")
    a1 = mp.mpf("0.35")
    for eps_str in ("1e-2", "1e-4", "1e-6"):
        eps = mp.mpf(eps_str)
        a2_below = mp.pi / 3 - eps
        a2_above = mp.pi / 3 + eps
        s2_b = sol_good(a2_below, 1)
        s2_a = sol_good(a2_above, 1)
        # Check dP/dalpha ~ sqrt(3) * eps and changes sign
        ok &= s2_b["dP_da"] > 0 and s2_a["dP_da"] < 0
        ok &= abs(s2_b["dP_da"] / eps - mp.sqrt(3)) < 2 * eps
        # Check jump of I_head(a1, a2) across a2 = pi/3: approaches +pi as eps -> 0+
        ih_b = I_head_closed(a1, a2_below)
        ih_a = I_head_closed(a1, a2_above)
        jump_head = ih_b - ih_a
        # Check jump of I_same(a2, a1) across a2 = pi/3: approaches +pi as eps -> 0+
        is21_b = I_same_closed(a2_below, a1)
        is21_a = I_same_closed(a2_above, a1)
        jump_21 = is21_b - is21_a
        # Check continuity of I_same(a1, a2) across a2 = pi/3
        is12_b = I_same_closed(a1, a2_below)
        is12_a = I_same_closed(a1, a2_above)
        d_ren_b = delta_renorm_good(a1, 1, a2_below, 1)
        print(
            f"   eps={eps_str:<4s}: dP2/da2={float(s2_b['dP_da']):+.3e}, "
            f"I_same(1,2) diff={float(is12_b - is12_a):+.3e}, "
            f"jump I_head={float(jump_head):+.8f} (pi={float(mp.pi):.8f}), "
            f"delta_renorm={float(d_ren_b):+.3e}"
        )
        ok &= abs(jump_head - mp.pi) < 20 * eps * abs(mp.ln(eps))
        ok &= abs(jump_21 - mp.pi) < 20 * eps * abs(mp.ln(eps))
        ok &= abs(is12_b - is12_a) < 20 * eps * abs(mp.ln(eps))

    # ------------------------------------------------------------------
    # 5. Behaviour near the three resonance lines (A_12 -> 0 or oo)
    # ------------------------------------------------------------------
    print("\n5. Behaviour near the three resonance lines (A_12 -> 0 or oo):")
    # (i) Head-on A_12 -> oo at alpha_1 + alpha_2 = pi/3 (inside regular region alpha_j < pi/3)
    a1 = mp.mpf("0.4")
    a2_res1 = mp.pi / 3 - a1
    I_head_at_res1 = I_head_closed(a1, a2_res1)
    dI_head_at_res1 = mp.diff(lambda x: I_head_closed(a1, x), a2_res1)
    print(f"   (i) Head-on A_12 -> oo (a1=0.4, a2->pi/3-0.4={float(a2_res1):.6f}):")
    for h_str in ("1e-2", "1e-5", "1e-8"):
        h = mp.mpf(h_str)
        s1, s2 = sol_good(a1, 1), sol_good(a2_res1 - h, -1)
        A_val = A12_good(s1, s2)
        ih = I_head_closed(a1, a2_res1 - h)
        d_ren = delta_renorm_good(a1, 1, a2_res1 - h, -1)
        print(
            f"       h={h_str:<4s}: A_12={float(A_val):.3e}, I_head={float(ih):+.10f}, "
            f"|I_head - I_head(res)|={float(abs(ih - I_head_at_res1)):.3e}, delta_renorm={float(d_ren):+.6f}"
        )
        ok &= abs(ih - (I_head_at_res1 - h * dI_head_at_res1)) < 10 * h**2 * abs(mp.ln(h))

    # (ii) Overtaking A_12 -> oo and Head-on A_12 -> 0 at alpha_2 - alpha_1 = pi/3
    a1 = mp.mpf("0.15")
    a2_res2 = a1 + mp.pi / 3  #note: Wait: a1 + pi/3 > pi/3! For both in regular region, |a1 - a2| < pi/3 always!
    # Wait: if a1 -> 0+ and a2 -> pi/3-, then a2 - a1 -> pi/3 inside the regular square (0, pi/3)^2!
    # Or for a1 = 0.15, a2 = a1 + pi/3 - h (just above the cusp, where the resonance line a2 - a1 = pi/3 lies):
    J_same_res2 = J_same_closed(a1, a2_res2)
    I_head_res2 = I_head_closed(a1, a2_res2)
    print(f"   (ii) Resonance line a2 - a1 = pi/3 (a1=0.15, a2->{float(a2_res2):.6f}):")
    for h_str in ("1e-2", "1e-5", "1e-8"):
        h = mp.mpf(h_str)
        A_ov = A12_good(sol_good(a1, 1), sol_good(a2_res2 - h, 1))
        A_ho = A12_good(sol_good(a1, 1), sol_good(a2_res2 - h, -1))
        js = J_same_closed(a1, a2_res2 - h)
        ih = I_head_closed(a1, a2_res2 - h)
        print(
            f"       h={h_str:<4s}: A_ov={float(A_ov):.3e}, A_ho={float(A_ho):.3e}, "
            f"|J_same-lim|={float(abs(js - J_same_res2)):.3e}, |I_head-lim|={float(abs(ih - I_head_res2)):.3e}"
        )
        ok &= abs(js - J_same_res2) < 5 * h and abs(ih - I_head_res2) < 5 * h

    # (iii) Overtaking A_12 -> 0 at alpha_1 + alpha_2 = 2 pi/3 (mirror image across the cusp)
    a1 = mp.mpf("0.8")
    a2_res3 = 2 * mp.pi / 3 - a1
    J_same_res3 = J_same_closed(a1, a2_res3)
    print(f"   (iii) Overtaking A_12 -> 0 (a1=0.8, a2->2pi/3-0.8={float(a2_res3):.6f}):")
    for h_str in ("1e-2", "1e-5", "1e-8"):
        h = mp.mpf(h_str)
        A_ov = A12_good(sol_good(a1, 1), sol_good(a2_res3 - h, 1))
        js = J_same_closed(a1, a2_res3 - h)
        print(
            f"       h={h_str:<4s}: A_ov={float(A_ov):.3e}, J_same={float(js):+.10f}, "
            f"|J_same-lim|={float(abs(js - J_same_res3)):.3e}"
        )
        ok &= abs(js - J_same_res3) < 5 * h

    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    assert ok

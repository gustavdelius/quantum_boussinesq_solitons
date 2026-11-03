"""Semiclassical soliton-defect transmission phase for the GOOD Boussinesq equation with the
Corrigan-Zambon (CZ, arXiv:2306.03772) type-I defect.  See phase2/defect_semiclassics.pdf.

Dictionary (c = lam = kappa = 1 in CZ):
  CZ field u = -3 phi (our U = phi_x = -u_x/3),  L_CZ = 9 L_ours,  so P_CZ = 9 P_ours, E_CZ = 9 E_ours.
  CZ soliton gamma = sech(th) = k = sin(alpha),  nu = tanh(th) = eps cos(alpha).
  Defect: s = sigma_0, orientation eta = sign(alpha_CZ) (alpha_CZ = eta/sqrt3).

Checks:
 1. dictionary: U = (k^2/2) sech^2, int U = M = 2k, P_CZ = 9 * (2/3) eps k^3 cos(alpha)
 2. CZ's z (5.5) equals [(p_b - q)/(p_a - q)]^eta, with soliton KP roots
      p_a = (k + eps cos(alpha)/sqrt3)/2,  p_b = (eps cos(alpha)/sqrt3 - k)/2,  defect root q = -s/(2 sqrt3)
 3. the delay Dx = -log(z^eps)/k, read off from peak positions of the exact CZ solution
 4. the phase  delta_D = (2/9)(ell - ell_D) log z_t - (2/3)(Pi M_D - Pi_D M),  z_t = z^eps,
      ell = -eps cos 3 alpha = ell(p_a) = ell(p_b),  ell(p) = -3 sqrt3 p (1 - 4 p^2),
      ell_D = ell(q) = (3 s - s^3)/2,  M_D = eps eta q,  Pi_D = eps eta (s^2 - 2)/(2 sqrt3),
    satisfies Dx = -d delta_D/dP   (40 digits, all four (eps, eta))
 5. defect vacua sigma in {s, 0, sigma_+-}, sigma_+-^2 + s sigma + s^2 = 3  <=>  roots q' with
    q'^2 + q q' + q^2 = 1/4 (same triple as q); absorption (z = 0 or infinity) iff q = p_a or p_b, and
    then the soliton's (E, P) equals the jump in the defect's (E_D, P_D) between vacua s and sigma_+-.
 6. type-I defect contributes Omega_D = (f_v - g_u) du ^ dv at x0 which vanishes on every vacuum sector
    (u - v fixed), so there are no defect-soliton cross terms in Omega.
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 40
r3 = mp.sqrt(3)
ok = True

# ---------------------------------------------------------------- 1. dictionary
for al in (mp.mpf('0.4'), mp.mpf('1.2')):
    for eps in (1, -1):
        k = mp.sin(al); nu = eps*mp.cos(al)
        ux = lambda x: -mp.mpf(3)/2*k**2*mp.sech(k*x/2)**2               # CZ u_x at t = 0
        ut = lambda x: -nu*ux(x)
        U = lambda x: -ux(x)/3
        e1 = abs(U(mp.mpf('0.3')) - k**2/2*mp.sech(k*mp.mpf('0.3')/2)**2)
        M = mp.quad(U, [-mp.inf, 0, mp.inf])
        PCZ = mp.quad(lambda x: -ut(x)*ux(x), [-mp.inf, 0, mp.inf])
        e2 = abs(M - 2*k); e3 = abs(PCZ - 9*mp.mpf(2)/3*eps*k**3*mp.cos(al))
        ok &= max(e1, e2, e3) < 1e-25
print('1. dictionary U=(k^2/2)sech^2, M=2k, P_CZ = 9 P_ours :', ok)


def roots(al, eps):
    k = mp.sin(al)
    return (k + eps*mp.cos(al)/r3)/2, (eps*mp.cos(al)/r3 - k)/2


def zCZ(al, s, eps, eta):
    sh, ch = eps*mp.cos(al)/mp.sin(al), 1/mp.sin(al)
    A = sh + s*ch
    return (A - eta*r3)/(A + eta*r3)


# ---------------------------------------------------------------- 2. z in root form (symbolic)
a, s_ = sp.symbols('alpha s', real=True)
R3 = sp.sqrt(3)
for eps in (1, -1):
    for eta in (1, -1):
        A = (eps*sp.cos(a) + s_)/sp.sin(a)
        z = (A - eta*R3)/(A + eta*R3)
        pa = (sp.sin(a) + eps*sp.cos(a)/R3)/2; pb = (eps*sp.cos(a)/R3 - sp.sin(a))/2; q = -s_/(2*R3)
        d = sp.simplify(sp.expand_trig(z - ((pb - q)/(pa - q))**eta))
        ell = -eps*sp.cos(3*a); lp = lambda p: -3*R3*p*(1 - 4*p**2)
        d2 = [sp.simplify(sp.expand_trig(ell - lp(p))) for p in (pa, pb)]
        d3 = sp.simplify(sp.expand(pa**2 + pa*pb + pb**2 - sp.Rational(1, 4)).rewrite(sp.cos))
        print('2. eps=%+d eta=%+d: z - root form = %s ; ell - ell(p_a,p_b) = %s ; p_a^2+p_ap_b+p_b^2-1/4 = %s'
              % (eps, eta, d, d2, sp.simplify(d3)))
        ok &= d == 0 and d2 == [0, 0] and sp.simplify(d3) == 0
print('   ell(q) = (3s - s^3)/2 :', sp.expand(-3*R3*(-s_/(2*R3))*(1 - 4*(s_/(2*R3))**2)))


# ---------------------------------------------------------------- 3. delay from the exact solution
def peak(f, x0):
    return mp.findroot(lambda x: mp.diff(f, x), x0, tol=mp.mpf(10)**-50, verify=False)


worst = 0
cases = [(mp.mpf(al), mp.mpf(s), eps, eta) for al in ('0.7', '1.2') for s in ('2.5', '-0.3', '-2.4')
         for eps in (1, -1) for eta in (1, -1)]
cases = [c for c in cases if zCZ(*c) > 0]
for al, s, eps, eta in cases:
    k = mp.sin(al); nu = eps*mp.cos(al); z = zCZ(al, s, eps, eta)
    T = mp.mpf(60)/nu*eps            # |T| chosen so that the soliton is far from x = 0
    def Ufun(t, side):               # side 'u' (x<0 solution) or 'v' (x>0 solution), our U = -u_x/3
        zz = 1 if side == 'u' else z
        return lambda x: 2*k**2*zz*mp.exp(k*(x - nu*t))/(1 + zz*mp.exp(k*(x - nu*t)))**2
    # right-mover: on u at t=-T, on v at t=+T; left-mover: on v at t=-T (x>0), on u at t=+T
    first, last = ('u', 'v') if eps == 1 else ('v', 'u')
    guess = lambda t, side: nu*t - (mp.log(z)/k if side == 'v' else 0) + mp.mpf('0.3')   # start off-peak
    Xi = peak(Ufun(-T, first), guess(-T, first)); Xf = peak(Ufun(T, last), guess(T, last))
    Dx = (Xf - nu*T) - (Xi + nu*T)
    pred = -mp.log(z**eps)/k
    worst = max(worst, abs(Dx - pred))
print('3. delay from exact solution vs -log(z^eps)/k (%d regular cases): max diff' % len(cases), mp.nstr(worst, 3))
ok &= worst < 1e-25


# ---------------------------------------------------------------- 4. the phase
def delta_D(al, s, eps, eta):
    pa, pb = roots(al, eps); q = -s/(2*r3)
    zt = ((pb - q)/(pa - q))**(eta*eps)
    k = mp.sin(al); M = 2*k; v = eps*mp.cos(al); Pi = -M*v
    ell = -eps*mp.cos(3*al); ellD = (3*s - s**3)/2
    MD = eps*eta*q; PiD = eps*eta*(s**2 - 2)/(2*r3)
    return mp.mpf(2)/9*(ell - ellD)*mp.log(zt) - mp.mpf(2)/3*(Pi*MD - PiD*M)


worst = 0
for s in (mp.mpf('0.7'), mp.mpf('2.6'), mp.mpf('-1.3'), mp.mpf('-3.2')):
    for eps in (1, -1):
        for eta in (1, -1):
            for al in (mp.mpf('0.3'), mp.mpf('0.8'), mp.mpf('1.25'), mp.pi/3 + mp.mpf('0.01')):
                z = zCZ(al, s, eps, eta)
                if z <= 0:
                    continue
                k = mp.sin(al)
                Dx = -mp.log(z**eps)/k
                dPda = mp.mpf(2)/3*eps*k*mp.sin(3*al)
                lhs = mp.diff(lambda x: delta_D(x, s, eps, eta), al)
                worst = max(worst, abs(lhs + Dx*dPda))
print('4. |d delta_D/d alpha + Dx dP/d alpha| max over 4x4x4 points:', mp.nstr(worst, 3))
ok &= worst < 1e-30
# orientation reversal: delta_D(-eta) = -delta_D(eta)
r = abs(delta_D(mp.mpf('0.6'), mp.mpf(2.5), 1, 1) + delta_D(mp.mpf('0.6'), mp.mpf(2.5), 1, -1))
print('   delta_D(eta=-1) = -delta_D(eta=+1):', mp.nstr(r, 3)); ok &= r < 1e-35

# ---------------------------------------------------------------- 5. vacua, absorption, energy balance
# CZ defect, c = lam = kappa = 1: beta = 2 alpha_CZ, mu = 2/(3 beta), sigma = s + mu w (w = u - v at x0)
# rho = rho0 + (sigma - sigma^3/3)/(2 mu), rho0 = -(s/(2 mu))(1 - s^2/3); dtau/dw = -beta sigma rho/2,
# dh/dw = -(beta - alpha) rho.  On a vacuum (u_x = v_x = 0): E_D = -tau, P_D = h.
worst = 0
for s in (mp.mpf('0.7'), mp.mpf('-1.3'), mp.mpf('1.9')):
    for eta in (1, -1):
        alCZ = eta/r3; beta = 2*alCZ; mu = mp.mpf(2)/(3*beta)
        rho = lambda sg: -(s/(2*mu))*(1 - s**2/3) + (sg - sg**3/3)/(2*mu)
        EDf = lambda sg: mp.quad(lambda x: beta*x*rho(x)/2/mu, [s, sg])     # -tau, d sigma = mu dw
        PDf = lambda sg: mp.quad(lambda x: -(beta - alCZ)*rho(x)/mu, [s, sg])
        sigs = [(-s + e*mp.sqrt(12 - 3*s**2))/2 for e in (1, -1)]
        for sg in sigs + [0]:
            assert abs(sg*rho(sg)) < 1e-30                                  # vacua: tau'(w) = 0
        q = -s/(2*r3)
        for sg in sigs:
            qp = -sg/(2*r3); assert abs(qp**2 + q*qp + q**2 - mp.mpf(1)/4) < 1e-30
        # absorption rapidities: q = p_a or p_b of some soliton (eps = +-1)
        jumps = [(EDf(sg), PDf(sg)) for sg in sigs]
        found = 0
        for eps in (1, -1):
            for which in (0, 1):
                f = lambda al: roots(al, eps)[which] - q
                for a0 in (0.2, 0.6, 1.0, 1.4):
                    try:
                        al = mp.findroot(f, mp.mpf(a0))
                    except Exception:
                        continue
                    if not (0 < al < mp.pi/2) or abs(f(al)) > 1e-30:
                        continue
                    k = mp.sin(al); v = eps*mp.cos(al)
                    A_ = (eps*mp.cos(al) + s)/mp.sin(al)
                    if not min(abs(A_ - r3), abs(A_ + r3)) < 1e-25:      # z = 0 or z = infinity
                        ok = False
                    P9 = 9*mp.mpf(2)/3*eps*k**3*mp.cos(al)
                    ux = lambda x: -mp.mpf(3)/2*k**2*mp.sech(k*x/2)**2
                    uxx = lambda x: mp.mpf(3)/2*k**3*mp.sech(k*x/2)**2*mp.tanh(k*x/2)
                    E9 = mp.quad(lambda x: (v*ux(x))**2/2 + ux(x)**2/2 + ux(x)**3/3 + uxx(x)**2/2, [-mp.inf, 0, mp.inf])
                    m = min(min(abs(E9 - sg_*dE) + abs(P9 - sg_*dP) for dE, dP in jumps) for sg_ in (1, -1))
                    worst = max(worst, m); found += 1
                    break
        print('5. s=%5s eta=%+d: vacua sigma_+- = %s ; %d absorbed solitons, |(E,P)_sol -+ jump(E_D,P_D)| <= %s'
              % (mp.nstr(s, 3), eta, [mp.nstr(x, 6) for x in sigs], found, mp.nstr(worst, 3)))
        ok &= found >= 1
ok &= worst < 1e-20

# ---------------------------------------------------------------- 6. Omega_D on vacuum sectors
u, v, w0 = sp.symbols('u v w0')
sg, mu_ = sp.symbols('s mu')
sigma = sg + mu_*(u - v)
f = -sigma**2/(4*mu_); g = -sigma**2/(4*mu_)          # f_v - g_u = sigma, shift-invariant gauge
print('6. f_v - g_u =', sp.simplify(sp.diff(f, v) - sp.diff(g, u)), '; Omega_D = (f_v-g_u) du^dv; on u - v = w0:',
      sp.simplify(((sp.diff(f, v) - sp.diff(g, u))*sp.Matrix([[1, 0], [1, 0]]).det())), '(du = dv => du^dv = 0)')
ok &= sp.simplify(sp.diff(f, v) - sp.diff(g, u) - sigma) == 0
assert ok
print('ALL CHECKS PASSED')

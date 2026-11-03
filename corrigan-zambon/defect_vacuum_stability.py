"""Linear stability of the Corrigan-Zambon defect vacua (good equation, c = lam = kappa = 1).
See phase2/defect_semiclassics.pdf, Sec. 8.

Linearise the sewing conditions (as in verify_defect_solutions.py) about a static vacuum u = w0, v = 0,
sigma_v = s + mu w0.  Bulk: a_tt = a_xx - a_xxxx (the cubic term drops at linear order).
Checks:
 1. static vacua: w0 = 0 (sigma = s) or sigma^2 + s sigma + s^2 = 3; sigma = 0 is NOT a vacuum
    (the u_xx sewing condition needs rho(sigma) = 0, and rho(0) = rho0 != 0 unless s in {0, +-sqrt3})
 2. phonon scattering on every real vacuum: R = 0, |T| = 1, and T equals CZ's linear-defect
    formula (4.6) with sigma0 -> sigma_vac, alpha + beta = sqrt3 eta:
       T = -(sqrt3 eta k^2 + i sigma k + i w)/(sqrt3 eta k^2 - i sigma k - i w)
 3. localised modes e^{kappa x - i w t}: a vacuum is linearly unstable (growing mode, Im w > 0) iff
    eta sigma > -1; growing modes have zero conserved energy (independent consistency check);
    the spectrum depends on (eta, sigma) only, not on s
 4. the vacua of one defect are sigma = -2 sqrt3 q' for the three roots q' of one KP triple; exactly one
    root has eta q' > 1/(2 sqrt3), so EXACTLY ONE vacuum per triple is linearly stable
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30
I = mp.mpc(0, 1)
U, UX, V, VX, UT, VT, UXT, VXT, UXX, UXXX, VXX, VXXX = sp.symbols('U UX V VX UT VT UXT VXT UXX UXXX VXX VXXX')
s = sp.Symbol('s')
a, ax, axx, axxx, at, axt, b, bx, bxx, bxxx, bt, bxt = syms = sp.symbols('a ax axx axxx at axt b bx bxx bxxx bt bxt')
e = sp.Symbol('e')
ok = True


def defect(eta):
    al = eta/sp.sqrt(3); be = 2*al; mu = sp.Rational(2, 3)/be
    sig = s + mu*(U - V); rho0 = -(s/(2*mu))*(1 - s**2/3); rho = rho0 + (sig - sig**3/3)/(2*mu)
    W = sp.Symbol('W')
    tau = sp.integrate((-be*sig*rho/2).subs(U, W + V), W).subs(W, U - V)
    f = g = -sig**2/(4*mu)
    F = al*UX + be*VX + f; G = be*UX + al*VX + g
    H = sig*(be*UX**2 + 2*al*UX*VX + be*VX**2)/2 + rho*(UX + VX) + tau
    D = UT*F + VT*G + H
    dt = lambda E: sum(sp.diff(E, z)*zt for z, zt in [(U, UT), (UX, UXT), (V, VT), (VX, VXT)])
    eqs = [-UX - UX**2 + UXXX + sp.diff(D, U) - dt(F), -UXX + sp.diff(D, UX),
           VX + VX**2 - VXXX + sp.diff(D, V) - dt(G), VXX + sp.diff(D, VX)]
    return eqs, H, mu, rho


# ---------------------------------------------------------------- 1. vacua
static = {UX: 0, VX: 0, UT: 0, VT: 0, UXT: 0, VXT: 0, UXX: 0, UXXX: 0, VXX: 0, VXXX: 0}
X = sp.Symbol('X')
for eta in (1, -1):
    eqs, H, mu, rho = defect(eta)
    for sv in (sp.Rational(7, 10), sp.Rational(-13, 10)):
        res = [sp.simplify(E.subs(static).subs({s: sv, V: 0})) for E in eqs]
        sol = set()
        for r in res:
            sol |= set(sp.solve(r, U))
        common = [w0 for w0 in sol if all(sp.simplify(r.subs(U, w0)) == 0 for r in res)]
        sigs = sorted([sp.nsimplify(sp.simplify(sv + mu*w0)) for w0 in common], key=float)
        expect = sorted([sv] + list(sp.solve(X**2 + sv*X + sv**2 - 3, X)), key=float)
        w_sig0 = -sv/mu
        r2 = sp.simplify(res[1].subs(U, w_sig0))
        print('1. eta=%+d s=%s: vacua sigma = %s ; expected %s ; u_xx-condition at sigma=0: %s'
              % (eta, sv, [float(x) for x in sigs], [float(x) for x in expect], r2))
        ok &= all(abs(float(x) - float(y)) < 1e-12 for x, y in zip(sigs, expect)) and len(sigs) == 3 and r2 != 0


# ---------------------------------------------------------------- linearisation
def lin(eta, sv, sig_vac):
    eqs, H, mu, rho = defect(eta)
    w0 = (sig_vac - sv)/mu
    sub = {U: w0 + e*a, V: e*b, UX: e*ax, UXX: e*axx, UXXX: e*axxx, UT: e*at, UXT: e*axt,
           VX: e*bx, VXX: e*bxx, VXXX: e*bxxx, VT: e*bt, VXT: e*bxt, s: sv}
    L = [sp.expand(sp.diff(E.subs(sub), e).subs(e, 0)) for E in eqs]
    H2 = sp.expand(sp.diff(H.subs(sub), e, 2).subs(e, 0)/2)
    return [sp.lambdify(syms, Li, 'mpmath') for Li in L], sp.lambdify(syms, H2, 'mpmath')


def col(Lf, side, kk, w):
    d = [1, kk, kk**2, kk**3]; dt = -I*w
    args = [d[0], d[1], d[2], d[3], dt*d[0], dt*d[1]] + [0]*6 if side == 'a' else [0]*6 + [d[0], d[1], d[2], d[3], dt*d[0], dt*d[1]]
    return [f(*args) for f in Lf]


def scatter(Lf, w):
    w = mp.mpf(w); k = mp.sqrt((-1 + mp.sqrt(1 + 4*w*w))/2); K = mp.sqrt((1 + mp.sqrt(1 + 4*w*w))/2)
    cols = [col(Lf, 'a', -I*k, w), col(Lf, 'a', K, w), col(Lf, 'b', I*k, w), col(Lf, 'b', -K, w)]
    M = mp.matrix([[cols[j][i] for j in range(4)] for i in range(4)])
    x = mp.lu_solve(M, mp.matrix([-c for c in col(Lf, 'a', I*k, w)]))
    return x[0], x[2], k


def kappas(w):
    r = mp.sqrt(1 + 4*w*w); ks = [mp.sqrt((1 + r)/2), mp.sqrt((1 - r)/2)]
    return [kk if mp.re(kk) > 0 else -kk for kk in ks]


def locmat(Lf, w):
    ks = kappas(w)
    cols = [col(Lf, 'a', ks[0], w), col(Lf, 'a', ks[1], w), col(Lf, 'b', -ks[0], w), col(Lf, 'b', -ks[1], w)]
    return mp.matrix([[cols[j][i] for j in range(4)] for i in range(4)])


def growing_modes(Lf):
    """roots of det in the upper half plane, excluding the spurious branch point w = i/2 (kappa1 = kappa2):
    fine scan of the imaginary axis plus Newton from a grid of complex starts"""
    found = []
    det = lambda w: mp.det(locmat(Lf, w))
    gs = mp.linspace(mp.mpf('0.002'), mp.mpf('0.498'), 125)
    vals = [mp.re(det(I*g)) for g in gs]
    for i in range(len(gs) - 1):
        if vals[i]*vals[i + 1] < 0:
            found.append(I*mp.findroot(lambda g: mp.re(det(I*g)), (gs[i], gs[i + 1]), solver='anderson'))
    for z0 in [mp.mpc(r, i_) for r in (-2, -0.8, -0.3, 0.3, 0.8, 2) for i_ in (0.2, 0.7, 1.3, 2.5)]:
        try:
            z = mp.findroot(det, z0, tol=1e-20, maxsteps=80)
        except Exception:
            continue
        if mp.im(z) > 1e-8 and abs(z - I/2) > 1e-6 and abs(det(z)) < 1e-12 and all(abs(z - f) > 1e-6 for f in found):
            found.append(z)
    return found


def mode_energy(Lf, H2f, g):
    """zero-energy test for a growing mode w = i g on the imaginary axis (real mode)"""
    w = I*g; ks = [mp.re(k_) for k_ in kappas(w)]
    M = mp.matrix([[mp.re(x) for x in row] for row in locmat(Lf, w).tolist()])
    Uu, Ss, Vv = mp.svd_r(M); c = [Vv[3, j] for j in range(4)]
    bulk = 0
    for p in (0, 1):
        for q in (0, 1):
            dens = (g*g + ks[p]*ks[q] + ks[p]**2*ks[q]**2)/2
            bulk += (c[p]*c[q] + c[2 + p]*c[2 + q])*dens/(ks[p] + ks[q])
    a0 = c[0] + c[1]; ax0 = c[0]*ks[0] + c[1]*ks[1]; b0 = c[2] + c[3]; bx0 = -(c[2]*ks[0] + c[3]*ks[1])
    h2 = H2f(a0, ax0, 0, 0, g*a0, g*ax0, b0, bx0, 0, 0, g*b0, g*bx0)
    return bulk - h2, bulk


# ---------------------------------------------------------------- 2. phonons
worst = 0
for eta in (1, -1):
    for sv in (sp.Rational(7, 10), sp.Rational(-13, 10), sp.Rational(5, 2)):
        for sg in [sv] + [r for r in sp.solve(X**2 + sv*X + sv**2 - 3, X) if r.is_real]:
            Lf, H2f = lin(eta, sv, sg)
            sgn = mp.mpf(sp.N(sg, 40))
            for w in ('0.3', '1.0', '2.5'):
                R, T, k = scatter(Lf, w)
                w_ = mp.mpf(w)
                Tcz = -(mp.sqrt(3)*eta*k**2 + I*sgn*k + I*w_)/(mp.sqrt(3)*eta*k**2 - I*sgn*k - I*w_)
                worst = max(worst, abs(R), abs(abs(T) - 1), abs(T - Tcz))
print('2. phonons on all real vacua: max(|R|, ||T|-1|, |T - T_CZ(4.6)|) =', mp.nstr(worst, 3))
ok &= worst < 1e-25

# ---------------------------------------------------------------- 3. stability map
print('3. growing localised modes (w in upper half plane), by (eta, sigma) and two values of s')
for eta in (1, -1):
    for sgv in ('-2.5', '-1.5', '-1.05', '-0.95', '-0.5', '0', '0.5', '0.95', '1.05', '1.5', '2.5'):
        sg = sp.nsimplify(sgv)
        results = []
        for sv in [sg] + [r for r in sp.solve(X**2 + sg*X + sg**2 - 3, X) if r.is_real][:1]:
            Lf, H2f = lin(eta, sv, sg)
            gm = growing_modes(Lf)
            for z in gm:
                if abs(mp.re(z)) < 1e-15:
                    E, Eb = mode_energy(Lf, H2f, mp.im(z))
                    ok &= abs(E) < 1e-20*max(1, abs(Eb))
            results.append(sorted([mp.nstr(z, 6) for z in gm]))
        stable = len(results[0]) == 0
        pred = eta*float(sg) < -1
        same = all(r == results[0] for r in results)
        print('   eta=%+d sigma=%5s: %s %s%s' % (eta, sgv, 'stable  ' if stable else 'UNSTABLE', results[0],
                                                '' if same else '  (s-DEPENDENT!)'))
        ok &= stable == pred and same

# ---------------------------------------------------------------- 4. one stable vacuum per triple
cnt = []
for qv in mp.linspace(-0.57, 0.57, 39):
    disc = mp.sqrt(1 - 3*qv*qv); tri = [qv, (-qv + disc)/2, (-qv - disc)/2]
    for eta in (1, -1):
        cnt.append(sum(1 for q in tri if eta*(-2*mp.sqrt(3)*q) < -1))
print('4. number of stable vacua (eta sigma < -1) per KP triple over a grid of q:', sorted(set(cnt)))
ok &= set(cnt) == {1}
assert ok
print('ALL CHECKS PASSED')

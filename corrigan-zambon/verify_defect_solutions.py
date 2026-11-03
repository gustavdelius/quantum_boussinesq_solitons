"""Check Corrigan-Zambon's exact soliton-defect solutions (arXiv:2306.03772 §5) against the sewing
conditions derived from the action (as in defect_check.py), with c = lam = kappa = 1, x0 = 0.

Defect: alpha = +-1/sqrt3, beta = 2 alpha, mu = 2 lam/(3 beta), sigma = s + mu (u - v),
rho = rho0 + (sigma - sigma^3/3)/(2 mu), rho0 = -(s/(2 mu))(1 - s^2/3)  (their 5.4, removes shifts),
d tau/du = -beta sigma rho/2,  f = int sigma dv, g = 0  (f_v - g_u = sigma).
[Note: their example f = g = -sigma^2/(2 mu c^2) in (3.43)/(3.47) gives f_v - g_u = 2 sigma/c^2,
 apparently a typo; f, g only enter through f_v - g_u up to a gauge term, so we use our choice.]
Solutions tested (u for x<0, v for x>0), E_a = exp(gamma_a (x - nu_a t)), nu = tanh th, gamma = sech th:
  one soliton  (5.2),(5.3),(5.5);  two solitons (5.6),(5.11);  merging (5.12)-(5.16).
"""
import sympy as sp
import mpmath as mp

x, t = sp.symbols('x t', real=True)
U, UX, V, VX, UT, VT, UXT, VXT = sp.symbols('U UX V VX UT VT UXT VXT')
lam = 1
sq3 = sp.sqrt(3)


def sewing_residuals(uexpr, vexpr, alpha, s):
    beta = 2*alpha
    mu = sp.Rational(2, 3)*lam/beta
    W = sp.Symbol('W')
    sigma = s + mu*(U - V)
    rho0 = -(s/(2*mu))*(1 - s**2/3)
    rho = rho0 + (sigma - sigma**3/3)/(2*mu)
    tau = sp.integrate((-beta*sigma*rho/2).subs(U, W + V), W).subs(W, U - V)
    f = sp.integrate(sigma, V); g = 0
    F = alpha*UX + beta*VX + f
    G = beta*UX + alpha*VX + g
    H = sigma*(beta*UX**2 + 2*alpha*UX*VX + beta*VX**2)/2 + rho*(UX + VX) + tau
    D = UT*F + VT*G + H
    dt = lambda e: sum(sp.diff(e, z)*zt for z, zt in [(U, UT), (UX, UXT), (V, VT), (VX, VXT)])
    eqs = [-UX - lam*UX**2 + sp.Symbol('UXXX') + sp.diff(D, U) - dt(F),
           -sp.Symbol('UXX') + sp.diff(D, UX),
           VX + lam*VX**2 - sp.Symbol('VXXX') + sp.diff(D, V) - dt(G),
           sp.Symbol('VXX') + sp.diff(D, VX)]
    subs = {U: uexpr, UX: sp.diff(uexpr, x), V: vexpr, VX: sp.diff(vexpr, x),
            UT: sp.diff(uexpr, t), VT: sp.diff(vexpr, t),
            UXT: sp.diff(uexpr, x, t), VXT: sp.diff(vexpr, x, t),
            sp.Symbol('UXX'): sp.diff(uexpr, x, 2), sp.Symbol('UXXX'): sp.diff(uexpr, x, 3),
            sp.Symbol('VXX'): sp.diff(vexpr, x, 2), sp.Symbol('VXXX'): sp.diff(vexpr, x, 3)}
    fs = [sp.lambdify(t, e.subs(subs).subs(x, 0), 'mpmath') for e in eqs]
    return max(abs(fn(mp.mpf(tt))) for fn in fs for tt in (-3, -0.7, 0, 0.4, 2.5))


def Ea(th, xs=0):
    return sp.exp(sp.sech(th)*(x - sp.tanh(th)*t) + xs)


def z1(th, s, sgn):           # (5.5), sgn = sign of alpha
    a = sp.sinh(th) + s*sp.cosh(th)
    return (a - sgn*sq3)/(a + sgn*sq3)


mp.mp.dps = 30
worst = 0
for sgn in (1, -1):
    alpha = sgn/sq3
    for th, s in [(sp.Rational(1), sp.Rational(1, 2)), (sp.Rational(-3, 5), sp.Rational(3, 2)), (sp.Rational(2, 5), -1)]:
        E = Ea(th); z = z1(th, s, sgn)
        u = -6*sp.sech(th)*E/(1 + E)
        v = -6*sp.sech(th)*z*E/(1 + z*E)
        r = sewing_residuals(u, v, alpha, s); worst = max(worst, r)
        print('one soliton  sgn(alpha)=%+d th=%s s=%s z=%.5f  residual %s' % (sgn, th, s, float(z), mp.nstr(r, 3)))
    # two solitons
    th1, th2, s = sp.Rational(1, 2), sp.Rational(-6, 5), sp.Rational(1, 3)
    E1, E2 = Ea(th1), Ea(th2, sp.Rational(1, 2))
    g1, g2 = sp.sech(th1), sp.sech(th2)
    S12 = (sp.sinh(th1 - th2)**2 - 3*(sp.cosh(th1) - sp.cosh(th2))**2) / \
          (sp.sinh(th1 - th2)**2 - 3*(sp.cosh(th1) + sp.cosh(th2))**2)
    za, zb = z1(th1, s, sgn), z1(th2, s, sgn)
    u = -6*(g1*E1 + g2*E2 + (g1 + g2)*S12*E1*E2)/(1 + E1 + E2 + S12*E1*E2)
    v = -6*(g1*za*E1 + g2*zb*E2 + (g1 + g2)*S12*za*zb*E1*E2)/(1 + za*E1 + zb*E2 + S12*za*zb*E1*E2)
    r = sewing_residuals(u, v, alpha, s); worst = max(worst, r)
    print('two solitons sgn(alpha)=%+d residual %s' % (sgn, mp.nstr(r, 3)))
    # merging / splitting
    th, s = sp.Rational(-3, 2), sp.Integer(2)
    w = [(1/sp.cosh(th) + sp.tanh(th)/sq3)/2, (-1/sp.cosh(th) + sp.tanh(th)/sq3)/2, -sp.tanh(th)/sq3]
    Es = [sp.exp(wa*(x - sq3*wa*t)) for wa in w]
    a = s*sp.cosh(th) + sp.sinh(th); b = s*sp.cosh(th) - 2*sp.sinh(th)
    zz = [b/(a + sq3), b/(a - sq3), 1] if sgn == 1 else [(a + sq3)/b, (a - sq3)/b, 1]
    u = -6*sum(wa*Ea_ for wa, Ea_ in zip(w, Es))/sum(Es)
    v = -6*sum(za_*wa*Ea_ for za_, wa, Ea_ in zip(zz, w, Es))/sum(za_*Ea_ for za_, Ea_ in zip(zz, Es))
    r = sewing_residuals(u, v, alpha, s); worst = max(worst, r)
    print('merging      sgn(alpha)=%+d residual %s' % (sgn, mp.nstr(r, 3)))
print('max residual:', mp.nstr(worst, 3))
# z < 0 cases have a singular v (pole crossing x = 0); there residuals are evaluated near the pole
assert worst < 1e-12
print('ALL CHECKS PASSED')

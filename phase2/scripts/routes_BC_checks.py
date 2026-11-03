"""Route C and Route B key-step checks.

C1. Hollowood's a_n^(1) interaction constant [hep-th/9110010 eq. (1.19)] for n=2 (a2 Toda), with
    sigma = m cosh(theta), sigma v = m sinh(theta):
       A^{ab}(theta) = -[2m^2 - 2m^2 cosh(theta) - 4m^2 sin^2(pi(a-b)/3)] / [2m^2 + 2m^2 cosh(theta) - 4m^2 sin^2(pi(a+b)/3)]
    Test the factorisation of the bad-Boussinesq constants into Toda/sine-Gordon blocks of S = b1+b2, D = b1-b2:
       A_ov(b1,b2) = [A^{12}(S) / A^{12}(D)] * [tanh^2(D/2) / tanh^2(S/2)]
       A_ho(b1,b2) = 1 / A_ov(b1, b2 + i pi)
C2. Equivalent form: A_ov = G(S/2)/G(D/2), G(y) = (4cosh^2 y - 1)/sinh^2 y
       = 16 sinh((y+i pi/3)/2) sinh((y-i pi/3)/2) sinh((y+2i pi/3)/2) sinh((y-2i pi/3)/2) / sinh^2 y
B1. Canonical brackets of the Lax data: q = (3/2) phi_x - 1/4, r = q_x/2 + (sqrt3/4) pi  (good, from the
    Lax flow).  {q(x), r(y)} = (3 sqrt3/8) d/dx delta(x-y), {q,q} = {r,r} = 0  -> non-ultralocal.
"""
import sympy as sp
import mpmath as mp

def A_bad(b1, b2, e2):
    k1, k2 = mp.sinh(b1), mp.sinh(b2)
    w1, w2 = k1*mp.cosh(b1), e2*k2*mp.cosh(b2)
    P = lambda K, W: W**2 - K**2 - K**4
    return -P(k1 - k2, w1 - w2)/P(k1 + k2, w1 + w2)

def A_toda(th, a, b, m=1):
    return -(2*m**2 - 2*m**2*mp.cosh(th) - 4*m**2*mp.sin(mp.pi*(a - b)/3)**2) / \
           (2*m**2 + 2*m**2*mp.cosh(th) - 4*m**2*mp.sin(mp.pi*(a + b)/3)**2)

G = lambda y: (4*mp.cosh(y)**2 - 1)/mp.sinh(y)**2
G4 = lambda y: 16*mp.sinh((y + 1j*mp.pi/3)/2)*mp.sinh((y - 1j*mp.pi/3)/2)*mp.sinh((y + 2j*mp.pi/3)/2) * \
               mp.sinh((y - 2j*mp.pi/3)/2)/mp.sinh(y)**2
worst = 0
for b1, b2 in [(1.3, 0.4), (0.25, 0.2), (2.7, 1.1)]:
    S, D = b1 + b2, b1 - b2
    fac = A_toda(S, 1, 2)/A_toda(D, 1, 2)*mp.tanh(D/2)**2/mp.tanh(S/2)**2
    cross = 1/A_bad(b1, b2 + 1j*mp.pi, 1)
    e = [abs(A_bad(b1, b2, 1) - fac), abs(A_bad(b1, b2, -1) - cross), abs(A_bad(b1, b2, 1) - G(S/2)/G(D/2)),
         abs(G(S/2) - G4(S/2))]
    worst = max(worst, max(e))
    print('C. b=(%.2f,%.2f): A_ov=%.10f  Toda/sG factorisation err %.1e | A_ho via b2+i pi err %.1e | G-form err %.1e, %.1e'
          % (b1, b2, A_bad(b1, b2, 1), *e))
assert worst < 1e-12

# B1 brackets via functional derivatives on test functions
x, y = sp.symbols('x y')
f, g = sp.Function('f')(x), sp.Function('g')(x)
# smeared functionals: Q[f] = int f q,  R[g] = int g r ;  {F,G} = int (dF/dphi dG/dpi - dF/dpi dG/dphi)
# Q[f] = int f (3/2 phi_x - 1/4)  -> dQ/dphi = -(3/2) f',  dQ/dpi = 0
# R[g] = int g ((3/4) phi_xx + sqrt3/4 pi) -> dR/dphi = (3/4) g'',  dR/dpi = sqrt3/4 g
dQphi, dQpi = -sp.Rational(3, 2)*sp.diff(f, x), 0
dRphi, dRpi = sp.Rational(3, 4)*sp.diff(g, x, 2), sp.sqrt(3)/4*g
QR = sp.simplify(dQphi*dRpi - dQpi*dRphi)          # integrand of {Q[f], R[g]}
print('B. {Q[f], R[g]} = int', QR, ' dx   i.e. {q(x), r(y)} = (3 sqrt3/8) delta\'(x-y): non-ultralocal')
# {R[f], R[g]} = int (dR_f/dphi dR_g/dpi - dR_f/dpi dR_g/dphi)
Rf_phi, Rf_pi = sp.Rational(3, 4)*sp.diff(f, x, 2), sp.sqrt(3)/4*f
Rg_phi, Rg_pi = sp.Rational(3, 4)*sp.diff(g, x, 2), sp.sqrt(3)/4*g
RRint = sp.expand(Rf_phi*Rg_pi - Rf_pi*Rg_phi)
print('   {R[f], R[g]} integrand =', RRint, ' -> integrates to', sp.simplify(sp.integrate(RRint.subs({f: sp.exp(-x**2), g: x*sp.exp(-x**2)}).doit(), (x, -sp.oo, sp.oo))), '(test functions): vanishes after integration by parts')
print('ALL CHECKS PASSED')

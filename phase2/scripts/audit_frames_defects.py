"""Independent checks of defect-spectrum branches and edge-reference subtraction.

All derivations are project claims (c), conditional on the existing CZ sewing
conditions. No quantum frame or regularization is selected. Uses exact SymPy
and 70-digit mpmath; writes only audit_frames_defects_results.json.
"""
from pathlib import Path
import hashlib
import json
import itertools
import mpmath as mp
import oneloop_defect_scattering_and_stability as existing
mp.mp.dps = 70
existing.r3 = mp.sqrt(3)  # reset cached precision after original-module import
ROOT = Path(__file__).resolve().parents[2]
INPUTS = ['phase2/charge_frame_master.tex', 'phase2/oneloop_defect_phase.tex',
          'phase2/defect_semiclassics.tex', 'phase2/toda_edge.tex',
          'phase2/scripts/oneloop_defect_scattering_and_stability.py',
          'phase2/scripts/toda_edge_ba.py', 'phase2/scripts/toda_edge_twoterm.py']
def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
before = hashes()
import sympy as s
S,d,z=s.symbols('S d z', real=True)
r=s.sqrt(3);g=(S*S-1)/2
k1=(S+d)/2;k2=(S-d)/2
def row(a,ax,axx,axxx,at,axt,b,bx,bxx,bxxx,bt,bxt):
 return [-ax-r*axt/3+axxx-bt*z-2*r*bxt/3+r*z*(z*z-1)*(a-b)/6-(ax+bx)*(z*z-1)/2,
 (1-z*z)*(a-b)/2+r*(at+2*bt)/3-axx+r*z*(2*ax+bx)/3,
 at*z-2*r*axt/3+bx-r*bxt/3-bxxx-r*z*(z*z-1)*(a-b)/6+(ax+bx)*(z*z-1)/2,
 (1-z*z)*(a-b)/2+r*(2*at+bt)/3+bxx+r*z*(ax+2*bx)/3]
cols=[]
for side in [0,1]:
 for k in [k1,k2]:
  k=(-1)**side*k
  jet=[1,k,k*k,k**3,g,g*k]
  cols.append(row(*(jet+[0]*6 if side==0 else [0]*6+jet)))
M=s.Matrix.hstack(*map(s.Matrix,cols))
T=s.Matrix.hstack(M[:,0]+M[:,1], (M[:,0]-M[:,1])/d, M[:,2]+M[:,3], (M[:,2]-M[:,3])/d)
T=T.applyfunc(lambda x:s.rem(s.cancel(x),d*d-(2-S*S),d).expand())
from itertools import permutations
D=0
for perm in permutations(range(4)):
 sign=(-1)**sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))
 D+=sign*s.prod(T[i,perm[i]] for i in range(4))
red=s.expand((2-S*S)*D/4)
pm=S*S-(r-1)*z*S/2+(z*z-2-r)/2
pp=S*S-(r+1)*z*S/2+(z*z-2+r)/2
pred=-s.Rational(2,3)*S*S*(S*S-1)*pm*pp*(2-S*S)
print('Exact off-shell sewing determinant identity: verified')
assert s.expand(red-pred)==0

# Columns were replaced by sum and difference/d in each half-line block.
# det(T)=4 det(M)/d^2, d^2=2-S^2. The claimed -1/24 is too small by 16.
assert s.expand(pred-16*(-S*S*(S*S-1)/24*pm*pp*(2-S*S))) == 0
Mfun = s.lambdify((S, d, z), M, 'mpmath')

def det4(A):
    # Leibniz formula also works at exactly singular matrices. mpmath's LU
    # determinant raised TypeError on one exploratory exactly singular matrix.
    return sum((-1)**sum(p[i] > p[j] for i in range(4) for j in range(i+1,4))
               * mp.fprod(A[i,p[i]] for i in range(4))
               for p in itertools.permutations(range(4)))

def direct_det(growth, sigma):
    ss = mp.sqrt(1+2*growth)
    dd = mp.sqrt(1-2*growth)
    return det4(Mfun(ss, dd, sigma))

worst = mp.mpf(0)
ratios = []
for sigma in map(mp.mpf, ['-1.5', '0.5', '1.9']):
    for growth in map(mp.mpf, ['0.2', '0.7']):
        actual = direct_det(growth, sigma)
        ref, k1, k2 = existing.locmat_4x4_det(mp.j*growth, sigma, 1)
        ss = k1+k2
        claimed = (-ss**2*(ss**2-1)/24 * existing.P_minus_quad(ss,sigma)
                   * existing.P_plus_quad(ss,sigma)*(k1-k2)**2)
        worst = max(worst, abs(actual-ref), abs(actual-16*claimed))
        ratios.append(mp.nstr(mp.re(actual/claimed), 30))
assert worst < mp.mpf('1e-60')
print('Nonzero determinant samples: actual/displayed =', ratios)

# Independent linearization of the original nonlinear sewing equations.
# This checks the row matrix, not only its determinant algebra.
nonlinear_error = mp.mpf(0)
kk = [2/mp.sqrt(5),1/mp.sqrt(5)]
growth = mp.mpf('0.4')
for sigma in map(mp.mpf, ['-1.5','0.5','1.9']):
    columns = []
    for side in [0,1]:
        for kval in kk:
            kval = (-1)**side*kval
            jet = [1,kval,kval**2,kval**3,growth,growth*kval]
            args = jet+[0]*6 if side == 0 else [0]*6+jet
            columns.append([mp.diff(lambda t: existing.cz_sewing_vec(
                *[t*j for j in args], sigma, 1)[i], 0) for i in range(4)])
    jac = mp.matrix([[columns[j][i] for j in range(4)] for i in range(4)])
    ours = Mfun(sum(kk),kk[0]-kk[1],sigma)
    nonlinear_error = max(nonlinear_error,max(abs(x) for x in jac-ours))
assert nonlinear_error < mp.mpf('1e-60')
print('Original nonlinear sewing Jacobian agrees with independent row matrix')

def admissible_roots(sigma):
    """Purely growing real modes for -2<sigma<2: retain only S>1."""
    r3 = mp.sqrt(3)
    disc = mp.sqrt(4-sigma*sigma)
    roots = [((r3-1)*sigma+sign*(r3+1)*disc)/4 for sign in [-1,1]]
    roots += [((r3+1)*sigma+sign*(r3-1)*disc)/4 for sign in [-1,1]]
    return [(ss,(ss*ss-1)/2) for ss in roots if ss > 1+mp.mpf('1e-60')]

spectra = {}
worst_root = mp.mpf(0)
for sigma_text in ['-1.5', '-0.95', '0.5', '1.5', '1.9', '1.99']:
    sigma = mp.mpf(sigma_text)
    roots = admissible_roots(sigma)
    expect = 0 if sigma < -1 else (1 if sigma < 1 else 2)
    assert len(roots) == expect
    for ss,growth in roots:
        worst_root = max(worst_root, abs(direct_det(growth,sigma)))
    spectra[sigma_text] = [mp.nstr(growth,50) for ss,growth in roots]
assert worst_root < mp.mpf('1e-60')
# The displayed pair includes a negative growth rate at sigma=1.9.
sigma = mp.mpf('1.9')
bad = (sigma*mp.sqrt(4-sigma*sigma)+mp.sqrt(3)*(2-sigma*sigma))/8
assert bad < 0
assert min(abs(g+bad) for _,g in admissible_roots(sigma)) < mp.mpf('1e-60')
print('At sigma=1.9: admissible growth rates', spectra['1.9'])
print('Displayed negative rate:', mp.nstr(bad,30))
# At s=2, both admissible roots of P_+ coalesce; P_- has S<1.
assert s.simplify(pp.subs(z,2)-(S-(s.sqrt(3)+1)/2)**2) == 0
assert s.simplify((((s.sqrt(3)+1)/2)**2-1)/2-s.sqrt(3)/4) == 0

# Check number of stable vacua on nondegenerate KP triples, without inferring
# any quantum asymptotic-state bootstrap from this classical calculation.
triples = 0
for n in range(1,100):
    theta = mp.pi*n/301
    qs = [mp.cos(theta+2*j*mp.pi/3)/mp.sqrt(3) for j in range(3)]
    assert abs(sum(qs)) < mp.mpf('1e-65')
    for i,j in [(0,1),(1,2),(2,0)]:
        assert abs(qs[i]**2+qs[i]*qs[j]+qs[j]**2-mp.mpf('0.25')) < mp.mpf('1e-65')
    for eta in [-1,1]:
        sigmas = [-2*mp.sqrt(3)*eta*q for q in qs]
        assert sum(not admissible_roots(sig) for sig in sigmas) == 1
        assert sum(sig < -1 for sig in sigmas) == 1
        triples += 1
print('Exactly one stable vacuum on',triples,'oriented nondegenerate triples')

# An edge reference C(Y), independent of y2=r^2Y, cannot change r dependence.
Y,r = s.symbols('Y r', positive=True)
F = s.Function('F')(Y,r)
C = s.Function('C')(Y)
assert s.diff(F-C,r) == s.diff(F,r)
assert s.diff(4*r/s.pi,r) != 0
# Explicit frame limit in the implementation: k_i=sqrt(y_i*h/b).
t,y1,y2,b,a = s.symbols('t y1 y2 b a', positive=True)
k1=t*s.sqrt(y1/b); k2=t*s.sqrt(y2/b)
pn2=4*b*(k2*s.cosh(k2)-s.sinh(k2))
pi2=2*b*s.sinh(k2)
frame=(-2*k1)*(pn2/a-pi2)/t**2
assert s.simplify(s.limit(frame,t,0)-4*s.sqrt(y1*y2)) == 0
print('Edge subtraction leaves ratio derivative unchanged; classical frame limit verified')

assert hashes() == before, 'Audited inputs changed during this run'
output = dict(precision=mp.mp.dps, input_sha256=before,
              determinant_ratio=ratios, determinant_max_error=str(worst),
              nonlinear_jacobian_max_error=str(nonlinear_error),
              root_max_error=str(worst_root), growth_rates=spectra,
              displayed_negative_rate_at_1_9=str(bad), stable_triples=triples,
              exact_checks=['sewing determinant factorization', 's=2 coalescence',
                            'edge-reference derivative', 'classical frame limit'])
Path(__file__).with_name('audit_frames_defects_results.json').write_text(
    json.dumps(output,indent=2)+'\n')
print('ALL CHECKS PASSED')

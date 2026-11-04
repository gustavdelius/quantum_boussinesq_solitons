#!/usr/bin/env python3
"""Exact and 55-digit checks of the potential domain and finite collective measure.

No field regulator or quantum boundary prescription is selected. The auxiliary
r direction is a diagnostic symplectic completion, not a new soliton solution.
"""
import json
from pathlib import Path
import time
import sympy as s
import mpmath as mp

started = time.perf_counter()
mp.mp.dps = 55
x, k, v, t = s.symbols('x k v t', real=True)
f, g, u = [s.Function(n)(x) for n in ('f', 'g', 'u')]
D = lambda z, n=1: s.diff(z, x, n)
L = lambda z: D(z,4)-D(z,2)+6*D(u*D(z))
K = lambda z: L(z)+v*v*D(z,2)
H = lambda z: -D(z,2)+(1-v*v-6*u)*z
assert s.expand(K(f)+D(H(D(f)))) == 0
B = f*D(g,3)-D(f)*D(g,2)+D(f,2)*D(g)-D(f,3)*g+(6*u+v*v-1)*(f*D(g)-g*D(f))
assert s.expand(D(B)-(f*K(g)-g*K(f))) == 0
# First variation boundary coefficient: A delta(phi)-phi_xx delta(phi_x).
A = -u+3*u*u+D(u,2)
assert s.expand(D(A)-(-D(u)+6*u*D(u)+D(u,3))) == 0
cdot, vdot = s.symbols('cdot vdot', real=True)
kv=-v/k
assert s.expand(((cdot+kv*vdot)**2+(cdot-kv*vdot)**2)/2
                -(cdot**2+v*v*vdot*vdot/(k*k))) == 0
# Local conservation law: pi_t = -d_x A; M_t = [pi].
p = s.Function('p')(x)
assert s.expand(-D(A)-(D(u)-6*u*D(u)-D(u,3))) == 0
et, ex, exx, z, ux = s.symbols('et ex exx z ux')
lag = (-v*u+z*(et-v*ex))**2/2-(u+z*ex)**2/2+(u+z*ex)**3-(ux+z*exx)**2/2
assert s.expand(lag).coeff(z,2) == s.expand((et-v*ex)**2/2-(1-6*u)*ex**2/2-exx**2/2)
# The polynomial substitution y=tanh(kx/2) makes all profile checks exact.
y = s.symbols('y', real=True)
dy = lambda F: s.diff(F,y)*k*(1-y*y)/2
U = k*k*(1-y*y)/2
assert s.expand(dy(dy(U))-k*k*U+3*U*U) == 0
assert s.expand(-dy(dy(dy(U)))+(k*k-6*U)*dy(U)) == 0
I = {n: s.integrate((k*k*(1-y*y)/2)**n*2/(k*(1-y*y)),(y,-1,1)) for n in (1,2,3)}
assert I == {1:2*k, 2:2*k**3/3, 3:4*k**5/15}
w = 3*U/k-15*U*U/(2*k**3)
assert s.integrate(w*2/(k*(1-y*y)),(y,-1,1)) == 1
assert s.integrate(w*U*2/(k*(1-y*y)),(y,-1,1)) == 0
E = (v*v+1)*I[2]/2-I[3]+s.integrate(dy(U)**2/(k*(1-y*y)),(y,-1,1))
assert s.simplify(E.subs(v*v,1-k*k)-(2*k**3/3-8*k**5/15)) == 0
P = 2*v*(1-v*v)**s.Rational(3,2)/3
Pi = -2*v*s.sqrt(1-v*v)
Pv = s.diff(P,v)
Piv = s.diff(Pi,v)
assert s.simplify(Pv-s.Rational(2,3)*s.sqrt(1-v*v)*(1-4*v*v)) == 0
assert s.simplify(Piv-2*(2*v*v-1)/s.sqrt(1-v*v)) == 0
assert s.simplify(s.diff(-s.Rational(2,15)*(1-v*v)**s.Rational(5,2),v)-P)==0
d, b = s.symbols('d b', nonzero=True)
W3 = s.Matrix([[0,d,b],[-d,0,0],[-b,0,0]])
assert W3*s.Matrix([0,b,-d]) == s.zeros(3,1)
assert W3.rank()==2
W = s.Matrix([[0,d,b,0],[-d,0,0,0],[-b,0,0,-1],[0,0,1,0]])
assert W.det() == d*d
# Exact finite-dimensional realization, including a transverse canonical pair.
J = s.diag(*[s.Matrix([[0,-1],[1,0]]) for _ in range(3)])
Z = s.zeros(6,4)
Z[:,0] = s.Matrix([0,d,0,b,0,0])
Z[:,1] = s.Matrix([1,0,0,0,0,0])
Z[:,2] = s.Matrix([0,0,1,0,0,0])
Z[:,3] = s.Matrix([0,0,0,1,0,0])
assert Z.T*J*Z == W
Proj = Z*W.inv()*Z.T*J
Q = s.eye(6)-Proj
assert s.simplify(Proj*Proj-Proj) == s.zeros(6)
assert Z.T*J*Q == s.zeros(4,6)
T = Z.row_join(s.eye(6)[:,4:6])
assert s.simplify(T.det()**2-W.det()) == 0
assert s.simplify((Z.T*J*(-Z)).det()-W.det()) == 0
# A formal Gaussian split retains the Schur complement.
aa,bb,cc = s.symbols('aa bb cc', nonzero=True)
assert s.Matrix([[aa,bb],[bb,cc]]).det() == s.expand(cc*(aa-bb*bb/cc))
# Finite temporal Dirichlet benchmark: spatial Jacobi modes are not null here.
n=s.symbols('n',integer=True,positive=True)
assert s.expand((n+1)-(2*n-(n-1)))==0
for n in range(1,7):
    mat = s.zeros(n)
    for i in range(n):
        mat[i,i]=2
        if i+1<n: mat[i,i+1]=mat[i+1,i]=-1
    assert mat.det() == n+1

rows=[]
for vv in map(mp.mpf, ['0.8','-0.7','0.3']):
    kk=mp.sqrt(1-vv*vv)
    def profile(xx): return kk*mp.tanh(kk*xx/2)
    def pulse(xx): return kk*kk/(2*mp.cosh(kk*xx/2)**2)
    def phiv(xx):
        return -vv/kk*(mp.tanh(kk*xx/2)+kk*xx/(2*mp.cosh(kk*xx/2)**2))
    def uv(xx): return mp.diff(lambda V:(1-V*V)/(2*mp.cosh(mp.sqrt(1-V*V)*xx/2)**2),vv)
    def ww(xx): return 3*pulse(xx)/kk-15*pulse(xx)**2/(2*kk**3)
    # Columns (v,X,C,r), evaluated at r=0 in the centred potential convention.
    fields=[(phiv,lambda xx:-pulse(xx)-vv*uv(xx)),
            (lambda xx:-pulse(xx),lambda xx:vv*mp.diff(pulse,xx)),
            (lambda xx:mp.mpf(1),lambda xx:mp.mpf(0)),
            (lambda xx:mp.mpf(0),ww)]
    expected=mp.matrix([[0,mp.diff(lambda V:2*V*(1-V*V)**mp.mpf('1.5')/3,vv),mp.diff(lambda V:-2*V*mp.sqrt(1-V*V),vv),0],
                        [0,0,0,0],[0,0,0,-1],[0,0,1,0]])
    expected[1,0]=-expected[0,1]; expected[2,0]=-expected[0,2]
    worst=mp.mpf(0)
    for i in range(4):
        for j in range(i+1,4):
            val=mp.quad(lambda xx:fields[i][1](xx)*fields[j][0](xx)-fields[j][1](xx)*fields[i][0](xx),[-mp.inf,0,mp.inf])
            worst=max(worst,abs(val-expected[i,j]))
    assert worst<mp.mpf('1e-45')
    # Large-box norms: shift and width modulation are not L2 directions.
    R=mp.mpf(150)/kk
    norm=mp.quad(lambda xx:phiv(xx)**2,[-R,0,R])
    norm2=mp.quad(lambda xx:phiv(xx)**2,[-2*R,0,2*R])
    slope=(norm2-norm)/(2*R)
    assert abs(slope-vv*vv/(kk*kk))<mp.mpf('1e-43')
    rows.append({'v':str(vv),'Wronskian_error':str(worst),'width_norm_slope':str(slope)})

result={'precision':mp.mp.dps,'checks':rows,'runtime_seconds':time.perf_counter()-started,
        'scope':'Finite collective measure and domain checks only; no functional determinant.',
        'failures_recorded':['The three soliton/shift tangents have rank two, so their inverse-W projector does not exist.',
                             'Shift and velocity tangents have linearly divergent configuration-space norms.']}
Path(__file__).with_name('det_geometry_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('ALL CHECKS PASSED')

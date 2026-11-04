#!/usr/bin/env python3
"""Source-equation checks only: classical BC flow is not a quantum regulator.
Source equations read in hep-th/9303166 sections 4-5 and nlin/0102021 section 1.
"""
import json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps=80
x,t=s.symbols('x t',real=True)
u=s.Function('U')(x,t); w=s.Function('W')(x,t)
ut=-s.diff(u,x,2)+2*s.diff(w,x)
wt=s.diff(w,x,2)-s.Rational(2,3)*s.diff(u,x,3)-s.Rational(2,3)*u*s.diff(u,x)
utt=s.expand(-s.diff(ut,x,2)+2*s.diff(wt,x))
assert s.simplify(utt+s.diff(u,x,4)/3+2*s.diff(u*u,x,2)/3)==0
# Source X=x and T=sqrt(3)*t; U=-1/4+(3/2)*u_good.
g=s.Function('g')(x,t)
source_rhs=utt.subs(u,-s.Rational(1,4)+s.Rational(3,2)*g).doit()
assert s.simplify(2*source_rhs-(s.diff(g,x,2)-s.diff(g,x,4)-3*s.diff(g*g,x,2)))==0
# Source pi_g=(4 sqrt(3)/3)*(W-U_X/2) for BC time orientation.
p=s.Function('p')(x,t)
Wmap=s.diff(-s.Rational(1,4)+s.Rational(3,2)*g,x)/2+s.sqrt(3)*p/4
mapped_ut=ut.subs({u:-s.Rational(1,4)+s.Rational(3,2)*g,w:Wmap},simultaneous=True).doit()
assert s.simplify(mapped_ut-s.sqrt(3)*s.diff(p,x)/2)==0
# The printed base lattice flow, linearized independently by differentiating RHS.
z,L,W,r=s.symbols('z L W r',nonzero=True)
e=s.symbols('e')
ll,ww=s.symbols('ll ww')
Ln=lambda j:L+e*ll*z**j
Wn=lambda j:W+e*ww*z**j
rhs1=Wn(-1)-Wn(0)+Ln(0)*(Ln(1)-Ln(-1))
rhs2=Wn(0)*(Ln(2)-Ln(-1))
lin=s.Matrix([s.diff(rhs1,e).subs(e,0),s.diff(rhs2,e).subs(e,0)]).jacobian([ll,ww])
expected=s.Matrix([[L*(z-1/z),1/z-1],[W*(z*z-1/z),0]])
assert s.simplify(lin-expected)==s.zeros(2)
char=s.factor((r*s.eye(2)-lin).det())
assert s.simplify(char-(r*r-L*(z-1/z)*r-W*(1/z-1)*(z*z-1/z)))==0
K=s.symbols('K',real=True)
D=-4*L*L*s.sin(K)**2+8*W*(s.cos(K)-s.cos(2*K))
Dcrit=s.trigsimp(D.subs({L:s.Rational(1,3),W:s.Rational(1,27)}))
assert s.trigsimp(Dcrit+4*(1-s.cos(K))**2/27)==0
rows=[]
for ks in ['0.1','0.7','1.8','3.0']:
 k=m.mpf(ks); zc=m.exp(1j*k)
 mat=m.matrix([[(zc-1/zc)/3,1/zc-1],[(zc*zc-1/zc)/27,0]])
 eig=m.eig(mat,left=False,right=False)
 target=[1j*m.sin(k)/3+sign*1j*(1-m.cos(k))/(3*m.sqrt(3)) for sign in [-1,1]]
 err=max(min(abs(a-b) for b in target) for a in eig)
 assert err<m.mpf('1e-70')
 rows.append(dict(K=ks,eigenvalue_error=str(err)))
# Diagnose why simply using the base-flow phonons with printed U background fails.
d,Q,U=s.symbols('d Q U',real=True)
Db=s.series(D.subs({L:(1-d*d*U/3)/3,W:(1-d*d*U)/27,K:d*Q}),d,0,6).removeO()
assert s.expand(Db+d**4*(Q**4+4*U*Q**2)/27)==0
# Good q^4 branch requires the specified higher Hamiltonian and its real form.
out=dict(precision_dps=80,source_continuum='U_TT=-U_XXXX/3-2*(U^2)_XX/3',
 dictionary='X=x, T=sqrt(3)*t, U=-1/4+3*u/2; BC pi=(4*sqrt(3)/3)*(W-U_X/2); Hikami reverses T',
 lattice_critical_discriminant='-4*(1-cos(K))^2/27',linear_modes=rows,
 base_flow_background_discriminant='-d^4*(Q^4+4*U*Q^2)/27+O(d^6)',
 lambda_lattice=None,
 source_reading=[dict(arxiv='hep-th/9303166',scope='abstract and HTML full-text sections 1-6; focus on 4-5'),
 dict(arxiv='nlin/0102021',scope='abstract and PDF excerpts: section 1, section 2.1 commutators and boundary condition, section 5 opening and section 7 opening; not whole paper')],
 missing=['self-adjoint real form and a Hamiltonian combination generating the desired higher flow',
 'stable uniform background for that Hamiltonian and its physical positive-frequency phonons',
 'canonical symplectic/action and hbar dictionary, not merely a PDE dictionary',
 'strain-dependent full Brillouin-zone mode sum with energy and pressure subtractions'],
 failures_recorded=['Naive use of the base-flow discriminant with the printed continuum U-background expansion has the opposite acoustic U sign to the displayed higher-flow continuum equation. This fails the proposed shortcut, not the higher-flow construction.'])
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Classical continuum good sign confirmed; no quantum lattice lambda extracted')
print('ALL CHECKS PASSED')

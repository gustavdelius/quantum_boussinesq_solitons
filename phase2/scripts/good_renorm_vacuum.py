#!/usr/bin/env python3
"""Good-vacuum static renormalisation and nonuniform on-shell diagnostic.
Independent SymPy/mpmath checks; no project imports. Writes only own JSON.
"""
import json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps = 80
U,q,L,h,lam = s.symbols('U q L h lam', real=True)
V=U**2/2-U**3
assert s.diff(V,U,2)==1-6*U
# Hamilton equations: u_t=pi_x; pi_t=(u-3u^2-u_xx)_x.
xx,tt=s.symbols('xx tt')
uf=s.Function('u')(xx,tt)
pit=s.diff(uf-3*uf**2-s.diff(uf,xx,2),xx)
assert s.simplify(s.diff(pit,xx)-(s.diff(uf,xx,2)-s.diff(uf,xx,4)-3*s.diff(uf**2,xx,2)))==0
pp,boost=s.symbols('pp boost')
assert s.expand((pp+boost)**2/2-pp**2/2)==boost*pp+boost**2/2
eta=s.symbols('eta')
assert s.expand(V.subs(U,U+eta)).coeff(eta,2)==(1-6*U)/2
bare=((L**2+1-6*U)**s.Rational(3,2)-(1-6*U)**s.Rational(3,2))/(6*s.pi)
assert s.simplify(s.diff(bare,L)-L*s.sqrt(L*L+1-6*U)/(2*s.pi))==0
W=-((1-6*U)**s.Rational(3,2)-1+9*U)/(6*s.pi)
assert s.simplify(s.limit(bare-bare.subs(U,0)-U*s.diff(bare,U).subs(U,0),L,s.oo)-W)==0
assert s.simplify(s.diff(W,U,2)+9/(2*s.pi*s.sqrt(1-6*U)))==0
assert s.simplify(s.diff(bare,U).subs(U,0)+3*(s.sqrt(L*L+1)-1)/(2*s.pi))==0
curvature=s.diff(V+h*(W+lam*U*U),U,2)
sound=s.series(s.sqrt(curvature.subs(U,0)),h,0,2).removeO()
assert s.simplify(sound-1-h*(lam-9/(4*s.pi)))==0
compliance=s.series(1/curvature,h,0,2).removeO()
assert s.simplify(compliance-(1/(1-6*U)-h*(2*lam-9/(2*s.pi*s.sqrt(1-6*U)))/(1-6*U)**2))==0
# All-background classical elasticity requires a local function, not constant lambda.
G=-W
assert s.simplify(s.diff(W+G,U,2))==0
assert s.series(G,U,0,5).removeO()==9*U**2/(4*s.pi)+9*U**3/(4*s.pi)+81*U**4/(16*s.pi)
# Differentiate the actual determinant twice, then check a separately integrated bubble.
rows=[]
for us in ['-0.08','0','0.02','0.10']:
 u=m.mpf(us); a=1-6*u
 cur=-9/(2*m.pi)*m.quad(lambda p:p/(p*p+a)**m.mpf('1.5'),[0,1,m.inf])
 exact=-9/(2*m.pi*m.sqrt(a))
 assert abs(cur-exact)<m.mpf('1e-70')
 val=m.quad(lambda p:-18*u*u*p/(m.sqrt(p*p+1)*(m.sqrt(p*p+a)+m.sqrt(p*p+1))**2),[0,1,m.inf])/(2*m.pi)
 target=-((1-6*u)**m.mpf('1.5')-1+9*u)/(6*m.pi)
 assert abs(val-target)<m.mpf('1e-70')
 rows.append(dict(U=us,curvature=str(cur),curvature_error=str(abs(cur-exact)),W_error=str(abs(val-target))))
# Smooth regulator: independent quadrature and analytic erfc formula.
reg=[]
for ds in ['0.1','0.001','0.00001']:
 d=m.mpf(ds)
 value=m.quad(lambda p:p*m.exp(-d*p*p)/(1+p*p)**m.mpf('1.5'),[0,1,1/m.sqrt(d),m.inf])
 exact=1-m.sqrt(m.pi*d)*m.exp(d)*m.erfc(m.sqrt(d))
 assert abs(value-exact)<m.mpf('1e-70')
 reg.append(dict(delta=ds,integral=str(value),error=str(abs(value-exact))))
# Differentiate the determinant logarithm to check the bubble combinatorial factor.
DD,bb,pp=s.symbols('DD bb pp', positive=True)
assert s.diff(s.log(DD-6*bb*pp**2)/2,bb,2).subs(bb,0)==-18*pp**4/DD**2
# Independent contraction primitive and soft dispersion splitting.
assert s.simplify(s.diff((s.sqrt(1+L*L)-1)/(2*s.pi),L)-L/(2*s.pi*s.sqrt(1+L*L)))==0
zz,xx=s.symbols('zz xx',positive=True)
omega=lambda v:v*s.sqrt(1+v*v)
split=s.series(omega(zz*xx)+omega(zz*(1-xx))-omega(zz),zz,0,5).removeO()
assert s.expand(split+3*zz**3*xx*(1-xx)/2)==0
# Frequency convolution, checked without substituting a claimed antiderivative.
conv=[]
for aa,bb,oo in [('1.2','.7','.3'),('.2','2.1','1.3')]:
 a,b,o=map(m.mpf,(aa,bb,oo))
 val=m.quad(lambda v:1/((v*v+a*a)*((v+o)**2+b*b)),[-m.inf,-o,0,m.inf])/(2*m.pi)
 exact=(a+b)/(2*a*b*((a+b)**2+o*o))
 assert abs(val-exact)<m.mpf('1e-70')
 conv.append(str(abs(val-exact)))
# Nonuniform on-shell bubble. q>0, r=q-p, A and B rationalize cancellation.
p,r,x,z=s.symbols('p r x z',real=True)
A=1+2*p*p+3*p*r+2*r*r
B2=(1+p*p)*(1+r*r)
assert s.factor(B2-A*A)==-(p+r)**2*(4*p*p+4*p*r+4*r*r+3)
# Rescale 0<p<q by p=q*x; its kernel times dp has limit -1/6.
pp=z*x; rr=z*(1-x)
AA=A.subs({p:pp,r:rr}); BB=s.sqrt(B2.subs({p:pp,r:rr}))
soft_scaled=-(pp*s.sqrt(1+pp*pp)+rr*s.sqrt(1+rr*rr))*(BB+AA)/(4*BB*z*(4*pp*pp+4*pp*rr+4*rr*rr+3))
assert s.simplify(s.limit(soft_scaled,z,0,dir='+')+s.Rational(1,6))==0
# At p<0 fixed, q->0 the kernel is |p|/[4(1+p^2)^(3/2)].
bubble_rows=[]
for qs in ['0.01','0.0001','0.000001','0.000000000001']:
 Q=m.mpf(qs)
 def kernel(P,inside):
  R=Q-P; B=m.sqrt((1+P*P)*(1+R*R)); A=1+2*P*P+3*P*R+2*R*R
  S=abs(P)*m.sqrt(1+P*P)+abs(R)*m.sqrt(1+R*R)
  if inside:
   return -S*(B+A)/(4*B*Q*Q*(4*P*P+4*P*R+4*R*R+3))/(2*m.pi)
  return S/(4*B*(B+A))/(2*m.pi)
 soft=m.quad(lambda xx:Q*kernel(Q*xx,True),[0,m.mpf('.5'),1])
 hard=2*m.quad(lambda rr:kernel(-rr,False),[0,Q,1,m.inf])
 bubble_rows.append(dict(q=qs,soft=str(soft),hard=str(hard),sequential_s1=str(-9*(soft+hard))))
 if Q==m.mpf('0.000000000001'):
  assert abs(soft+1/(12*m.pi))<m.mpf('1e-22')
  assert abs(hard-1/(4*m.pi))<m.mpf('1e-22')
  assert abs(-9*(soft+hard)+3/(2*m.pi))<m.mpf('1e-21')
# Cubic Hamiltonian is unbounded even at fixed mean: a zero-mean test profile.
x=s.symbols('x',real=True)
f=s.cos(x)+s.cos(2*x)
assert s.integrate(f,(x,0,2*s.pi))==0
assert s.integrate(f**3,(x,0,2*s.pi))==3*s.pi/2
out=dict(precision_dps=m.mp.dps,static_rows=rows,smooth_regulator_rows=reg,frequency_convolution_errors=conv,
 W_exact=str(W),lambda_acoustic='9/(4*pi)',s1_normal_ordered='-9/(4*pi)',
 all_background_counterterm=str(G),on_shell_rows=bubble_rows,
 sequential_on_shell_s1='-3/(2*pi)',
 qualifications=['s1 without qualifier is thermodynamic, not the nonuniform on-shell expansion.',
 'No global ground state exists for the cubic Hamiltonian; vacuum means perturbative local branch.',
 'Exact low-q spectral slope needs a specified quantum state and control of infrared resummation.'],
 failures_recorded=['Exploratory direct on-shell denominator subtraction caused ZeroDivisionError at q=1e-4. Algebraic rationalization removes the cancellation; no tolerance weakened.'])
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('Static s1 = -9/(4*pi); acoustic lambda = 9/(4*pi)')
print('Sequential bare-on-shell s1 = -3/(2*pi), not the thermodynamic limit')
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""Conditional good trace, local counterterms, and inverse continuation.
Uses fresh convergent phase-trace quadratures; imports no historical checker.
"""
import json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps=80
k,lam,h,y,U,mu=s.symbols('k lam h y U mu',real=True)
v=s.sqrt(1-k*k)
profile=k*k*(1-y*y)/2
dxdy=2/(k*(1-y*y))
I2=s.integrate(profile**2*dxdy,(y,-1,1))
I3=s.integrate(profile**3*dxdy,(y,-1,1))
Ix=s.integrate((s.diff(profile,y)/dxdy)**2*dxdy,(y,-1,1))
assert I2==2*k**3/3 and I3==4*k**5/15 and Ix==2*k**5/15
P=v*I2; E=(v*v+1)*I2/2-I3+Ix/2
assert s.simplify(E-2*k**3*(5-4*k*k)/15)==0
assert s.simplify(s.diff(E,k)-v*s.diff(P,k))==0
F=s.sqrt(3)/s.pi*(s.Rational(3,4)-k*k)*s.atanh(2*k/s.sqrt(3))-3*k/(2*s.pi)
assert s.series(F,k,0,6).removeO()==-4*k**3/(3*s.pi)-16*k**5/(45*s.pi)
Cg=s.limit((F+lam*I2)/P,k,0,dir='+')
assert s.simplify(Cg-lam+2/s.pi)==0
sg=lam-9/(4*s.pi)
assert s.simplify(Cg-sg-1/(4*s.pi))==0
# Independent fixed-P classical deformation with stiffness a=1+2 h lambda.
a=s.symbols('a',positive=True)
va=s.sqrt(a-k*k); Pa=va*I2
Ea=(va*va+a)*I2/2-I3+Ix/2
# dk/dh at fixed P, followed by total dE/dh.
dkdh=-2*lam*s.diff(Pa,a)/s.diff(Pa,k)
dEdh=(2*lam*s.diff(Ea,a)+s.diff(Ea,k)*dkdh).subs(a,1)
assert s.simplify(dEdh-lam*I2)==0
assert s.simplify(s.limit(k*k*(2*mu*k/P),k,0,dir='+')-3*mu)==0
# Classical dictionary and consistent one-loop density continuation.
b=s.symbols('b',real=True)
assert s.simplify(s.sin(s.I*b)-s.I*s.sinh(b))==0
assert s.simplify(s.cos(s.I*b)-s.cosh(b))==0
assert s.simplify(-s.cos(3*s.I*b)+s.cosh(3*b))==0
Fb=-s.I*F.subs(k,s.I*k)
expected_Fb=s.sqrt(3)/s.pi*(s.Rational(3,4)+k*k)*s.atan(2*k/s.sqrt(3))-3*k/(2*s.pi)
assert s.simplify(Fb-expected_Fb)==0
Wg=-((1-6*U)**s.Rational(3,2)-1+9*U)/(6*s.pi)
Wb=-Wg.subs(U,-U)
assert s.diff(Wb,U,2).subs(U,0)/2==9/(4*s.pi)
# Inverse of W_b(U)=-W_g(-U): lambda_g=-lambda_b.
lambda_b=-3/(2*s.pi); lambda_inverse=-lambda_b
assert s.simplify(-((lambda_b*U**2).subs(U,-U))-lambda_inverse*U**2)==0
WT=(s.exp(3*U)-1-3*U)/(6*s.pi)
WTg=-WT.subs(U,-U)
assert s.diff(WTg,U,2).subs(U,0)/2==-3/(4*s.pi)
assert s.simplify(s.diff(WTg-Wg,U,2).subs(U,0)/2-lambda_inverse)==0
# Good group velocity cannot meet a subsonic right-mover at any q>0.
q=s.symbols('q',positive=True)
group=s.diff(q*s.sqrt(1+q*q),q)
assert s.simplify(group**2-1-q*q*(3+4*q*q)/(1+q*q))==0
# Fresh phase-density quadrature, with endpoint and Wick subtraction combined.
trace_rows=[]
for ks in ['0.12','0.31','0.6','0.8']:
 K=m.mpf(ks); alpha=m.asin(K); V=m.sqrt(1-K*K)
 roots=[(alpha+2*m.pi/3,1),(alpha-2*m.pi/3,1),(alpha,-2)]
 def ene(q):
  R=m.sqrt(1+q*q)
  return (q/R*sum(c*m.sin(f)*m.cos(f)**2/(q*q+m.sin(f)**2) for f,c in roots)-3*K/(R*(q+R)))/(2*m.pi)
 def mom(q):
  return q/m.sqrt(1+q*q)*sum(c*m.sin(f)*m.cos(f)/(q*q+m.sin(f)**2) for f,c in roots)/(2*m.pi)
 eq=m.quad(ene,[0,K,1,m.inf]); pq=m.quad(mom,[0,K,1,m.inf]); fq=eq-V*pq
 target=m.sqrt(3)/m.pi*(m.mpf('.75')-K*K)*m.atanh(2*K/m.sqrt(3))-3*K/(2*m.pi)
 assert abs(fq-target)<m.mpf('1e-65')
 trace_rows.append(dict(k=ks,e=str(eq),p=str(pq),F=str(fq),error=str(abs(fq-target))))
# Small-k extrapolation uses exact interpolation weights, with no target fitting.
fnum=s.lambdify(k,F/P,'mpmath')
xs=[(m.mpf('0.00001')/2**j)**2 for j in range(4)]
ys=[fnum(m.sqrt(x)) for x in xs]
intercept=sum(ys[j]*m.fprod(-xs[l]/(xs[j]-xs[l]) for l in range(4) if l!=j) for j in range(4))
assert abs(intercept+2/m.pi)<m.mpf('1e-40')
conditions=[]
for name,lv,status in [
 ('free_vacuum_normal_ordering',s.Integer(0),'well_defined_perturbative'),
 ('classical_curvature_at_U0',9/(4*s.pi),'recommended_not_adopted'),
 ('classical_hydrodynamic_sound_at_U0',9/(4*s.pi),'conditional_on_hydrodynamic_identification'),
 ('classical_curvature_all_U',9/(4*s.pi),'requires_full_local_function_G'),
 ('inverse_Toda_matching',3/(2*s.pi),'formal_continuation_not_lattice_matching_for_good'),
 ('literal_bad_coefficient',-3/(2*s.pi),'arbitrary_finite_good_scheme_not_dictionary_continuation'),
 ('sequential_bare_on_shell_subtraction',3/(2*s.pi),'not_exact_q_to_zero_phonon_condition')]:
 c=s.simplify(Cg.subs(lam,lv)); ss=s.simplify(sg.subs(lam,lv))
 conditions.append(dict(id=name,lambda_exact=str(lv),lambda_decimal=str(s.N(lv,70)),
 C_good_exact=str(c),C_good_decimal=str(s.N(c,70)),s1_thermodynamic_exact=str(ss),
 C_good_minus_s1_exact=str(s.simplify(c-ss)),status=status,
 claim_label='c, conditional trace for C_good'))
out=dict(schema_version=1,convention='H_g=pi^2/2+u^2/2-u^3+u_x^2/2; Delta H=hbar*lambda*integral(u^2 dx)',
 lambda_reference='finite addition to free-good-vacuum normal ordering at U=0; fixed reference x,t',
 s1_definition='thermodynamic: c_hyd=1+hbar*s1; do not substitute the sequential on-shell coefficient',
 tadpole_reference='E_vac(0)=0 and E_vac_prime(0)=0; no extra finite mu*M',
 adopted_condition=None,recommendation=dict(id='classical_curvature_at_U0',label='d',lambda_exact='9/(4*pi)'),
 conditions=conditions,functions=dict(F_good='sqrt(3)/pi*(3/4-k^2)*atanh(2*k/sqrt(3))-3*k/(2*pi)+2*lambda*k^3/3',
 C_good='lambda-2/pi',s1='lambda-9/(4*pi)',invariant='1/(4*pi)',
 all_background_G='((1-6*u)^1.5-1+9*u)/(6*pi)'),
 domain='0<k<sqrt(3)/2; C is lim k->0 of one-loop coefficient after hbar expansion',
 precision_dps=80,trace_rows=trace_rows,small_k_extrapolation_error=str(abs(intercept+2/m.pi)),
 physical_determinant_completed=False,lattice_lambda=None,
 qualifications=['C and its invariant use the audited conditional equal-mode-count trace.',
 'The invariant is only within the specified quadratic bulk-counterterm class with fixed tadpole, units and boundary prescription.',
 'No global ground-state completion or uniform finite-hbar small-amplitude limit is claimed.'])
Path(__file__).with_name('good_renorm_conditions.json').write_text(json.dumps(out,indent=2)+'\n')
print('C_good(lambda)=lambda-2/pi; C_good-s1=1/(4*pi)')
print('Trace max quadrature error:',max(m.mpf(row['error']) for row in trace_rows))
print('ALL CHECKS PASSED')

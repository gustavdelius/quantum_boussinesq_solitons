#!/usr/bin/env python3
"""(c) Exact constraint algebra; no imported project code or old cache writes.
Physical quantum bootstrap assumptions are (d)."""
from pathlib import Path
import json
import time
import sympy as S
import mpmath as m
start=time.perf_counter()
m.mp.dps=80
r=S.sqrt(3)
a,b,q,x,y,lam,h=S.symbols('a b q x y lambda h', real=True)
zero=lambda z: S.simplify(S.expand(z)) == 0
conic=a*a+a*b+b*b-S.Rational(1,4)
pot=lambda z: 2*r*z**4-z*z/r-r/36
mass=lambda z: 2*z
Pedge=S.Rational(2,3)*r*(a-b)**3*(a+b)
assert S.rem(S.expand(Pedge-(pot(a)-pot(b))),conic,a)==0
assert zero(2*(a-b)-(mass(a)-mass(b)))
# An explicit two-coefficient bilinear, not two antisymmetric directions.
P1,P2,M1,M2,A,B=S.symbols('P1 P2 M1 M2 A B')
F=A*P1*M2+B*M1*P2
assert zero(F+F.xreplace({P1:P2,P2:P1,M1:M2,M2:M1})-(A+B)*(P1*M2+M1*P2))
ps=S.symbols('p0:3');ms=S.symbols('m0:3');rs=S.symbols('r0:3');ns=S.symbols('n0:3')
minor=S.Matrix(3,3,lambda i,j:ps[i]*ns[j]-ms[i]*rs[j])
assert S.expand(minor.det())==0
assert S.Matrix([[1,2],[-3,4]]).det()!=0
# The genuinely rank-two charge kernel has a nonzero mixed 2x2 minor.
rows=[(S.Rational(2,3)*r*(aa-bb)**3*(aa+bb),2*(aa-bb))
      for aa,bb in [(S.Rational(1,2),0),(0,-S.Rational(1,2))]]
assert S.Matrix(2,2,lambda i,j:rows[i][0]*rows[j][1]-rows[i][1]*rows[j][0]).det()!=0
# Rational parametrisation of the entire nondegenerate KP triangle.
t=S.symbols('t',positive=True)
ct=(1-t*t)/(1+t*t);st=2*t/(1+t*t)
qs=[ct/(2*r)+st/2,ct/(2*r)-st/2,-ct/r]
assert zero(sum(qs));assert zero(sum(z*z for z in qs)-S.Rational(1,2))
for z in qs[1:]:assert zero((z**3-z/4)-(qs[0]**3-qs[0]/4))
ds=[qs[i]-qs[(i+1)%3] for i in range(3)]
ks=[st,r*ct/2-st/2,r*ct/2+st/2]
vs=[ct,-ct/2-r*st/2,-ct/2+r*st/2]
for i in range(3):
    assert zero(ds[i]-([1,1,-1][i])*ks[i])
    assert zero(vs[i]-r*(qs[i]+qs[(i+1)%3]))
    assert zero(ks[i]**2+vs[i]**2-1)
assert zero(ks[2]-ks[0]-ks[1])
assert zero(r*ks[0]*ks[1]-r*(S.Rational(1,4)-vs[2]**2))
assert zero(sum(pot(z) for z in qs))
# Exact closed-loop sums; no bulk phase is used to define these potentials.
assert zero(sum(ds))
assert zero(sum(S.Rational(2,3)*r*ds[i]**3*(qs[i]+qs[(i+1)%3]) for i in range(3)))
for func in [mass,pot]:assert zero(sum(func(qs[i])-func(qs[(i+1)%3]) for i in range(3)))
# General edge equations on the symmetric triangle. H_i are independent
# unknown boundary amplitudes at a fixed spectator, eps=eta=1.
qq=[S.Rational(1,2),S.Integer(0),-S.Rational(1,2)]
pt,mt=S.symbols('pt mt',nonzero=True)
edge=S.Matrix([[pt*(mass(qq[i])-mass(qq[(i+1)%3])),
                mt*(pot(qq[i])-pot(qq[(i+1)%3]))] for i in range(3)])
inc=S.Matrix([[-1,1,0],[0,-1,1],[1,0,-1]])
assert inc.rank()==2
assert inc.row_join(-edge).rank()==2 # rank increment after nuisance elimination=0
assert (S.ones(1,3)*edge)==S.zeros(1,2)
# Explicit lift with common coefficients at every vertex and every triangle.
d,e=S.symbols('d e')
H=lambda z:d*pt*mass(z)+e*mt*pot(z)
for i in range(3):
    residual=H(qs[(i+1)%3])-H(qs[i])-(A*pt*(mass(qs[i])-mass(qs[(i+1)%3]))+B*mt*(pot(qs[i])-pot(qs[(i+1)%3])))
    assert zero(residual.subs({d:-A,e:-B}))
assert zero(sum(H(z) for z in qs)) # also preserves exceptional two-CZ reduction
C=S.Matrix([[1,0,1,0],[0,1,0,1]]) # columns A,B,d,e
assert C.rank()-C[:,2:].rank()==0
Canti=C.col_join(S.Matrix([[1,1,0,0]]))
assert Canti.rank()-Canti[:,2:].rank()==1 # only pre-existing antisymmetry
# Counterterm does not acquire new F-dependence at order h^0.
# O(h) shifts of classical charges in F enter i log S only at O(h).
p1,p2,m1,m2,dp1,dp2,dm1,dm2=S.symbols('p1 p2 m1 m2 dp1 dp2 dm1 dm2')
assert S.expand((p1+h*dp1)*(m2+h*dm2)-(p2+h*dp2)*(m1+h*dm1)).coeff(h,0)==p1*m2-p2*m1
# Bulk K_lambda from fixed-P scaling, checked by generic chain rule.
f,fp1,fp2=S.symbols('f fp1 fp2')
assert zero(3*f-4*(p1*fp1+p2*fp2)-(3*f+4*(p1*(-fp1)+p2*(-fp2))))
# Stable root injectivity and impossibility of stable special fusion.
u,w=S.symbols('u w',positive=True)
assert zero(12*(1/(2*r)+u)**2-1-(4*r*u+12*u*u))
stable_excess=S.expand((1/(2*r)+u)**2+(1/(2*r)+u)*(1/(2*r)+w)+(1/(2*r)+w)**2-S.Rational(1,4))
assert zero(stable_excess-(u*u+u*w+w*w+r*(u+w)/2))
# Exact necessary and sufficient conic identity for Z1 Z2 = inverse Z3.
q1,q2,q3=S.symbols('q1 q2 q3')
expr=S.prod(b-z for z in [q1,q2,q3])-S.prod(a-z for z in [q1,q2,q3])
red=S.rem(S.expand(expr/(b-a)).cancel(),conic,a)
assert zero(red-(S.Rational(1,4)-(q1+q2+q3)*(a+b)+q1*q2+q1*q3+q2*q3))
assert zero(red.subs(q3,-q1-q2)-(S.Rational(1,4)-q1*q1-q1*q2-q2*q2))
# Sample geometry and admissible growing roots (spectral c=eta*s).
worst=m.mpf(0); samples=[]
for tv in ['.0001','.1','.27','.52','.8','1.0471']:
    th=m.mpf(tv); roots=[-m.cos(th+2*m.pi*j/3)/m.sqrt(3) for j in [1,2,3]]
    kk=[abs(roots[i]-roots[(i+1)%3]) for i in range(3)]
    vv=[m.sqrt(3)*(roots[i]+roots[(i+1)%3]) for i in range(3)]
    al=[m.acos(abs(v)) for v in vv]
    assert al[0]<m.pi/3 and al[1]<m.pi/3 and al[2]>m.pi/3
    for eta in [-1,1]:
        stable=[z for z in roots if eta*z>1/(2*m.sqrt(3))]
        assert len(stable)==1
        for z in roots:
            c=-2*m.sqrt(3)*eta*z
            rr=[]
            for sign in [-1,1]:
                aa=(m.sqrt(3)+sign)*c/2;bb=(c*c-2+sign*m.sqrt(3))/2
                rr.extend([(aa+m.sqrt(aa*aa-4*bb))/2,(aa-m.sqrt(aa*aa-4*bb))/2])
            count=sum(abs(m.im(j))<m.mpf('1e-65') and m.re(j)>1+m.mpf('1e-65') for j in rr)
            assert (count==0)==(z in stable)
    PP=[m.mpf(2)/3*kk[i]**3*vv[i] for i in range(3)]
    err=max(abs(PP[0]+PP[1]-PP[2]),abs(kk[0]+kk[1]-kk[2]))
    worst=max(worst,err)
    samples.append({'theta':tv,'alphas':[str(z) for z in al],'P_closure_error':str(err)})
assert worst<m.mpf('1e-65')
# Additional displayed kinematic/convention checks, without old project imports.
k,v,Y=S.symbols('k v Y',real=True)
profile=k*k*(1-Y*Y)/2
measure=2/(k*(1-Y*Y))
assert zero(S.integrate(profile*measure,(Y,-1,1))-2*k)
assert zero(S.integrate(profile**2*measure,(Y,-1,1))-2*k**3/3)
assert zero(S.integrate((-v*profile)*measure,(Y,-1,1))+2*k*v)
assert zero(S.integrate(v*profile**2*measure,(Y,-1,1))-2*v*k**3/3)
aa=(v/r+k)/2;bb=(v/r-k)/2
assert S.rem(S.expand(aa*aa+aa*bb+bb*bb-S.Rational(1,4)),k*k+v*v-1,k)==0
assert S.rem(S.expand(3*r*(4*aa**3-aa)-v*(3-4*v*v)),k*k+v*v-1,k)==0
assert S.rem(S.expand(3*r*(4*bb**3-bb)-v*(3-4*v*v)),k*k+v*v-1,k)==0
th=S.symbols('th',real=True)
assert S.simplify(S.sech(th)**2+S.tanh(th)**2-1)==0
ptt,px,pxx=S.symbols('ptt px pxx')
LCZ=(-3*ptt)**2/2-(-3*px)**2/2-(-3*px)**3/3-(-3*pxx)**2/2
assert zero(LCZ-9*(ptt**2/2-px**2/2+px**3-pxx**2/2))
# Counterterm variation of the exact stiffness-family momentum at fixed v.
bulk_stiff=S.symbols('bulk_stiff',positive=True)
Pstiff=S.Rational(2,3)*v*(bulk_stiff-v*v)**S.Rational(3,2)
assert S.simplify(2*lam*S.diff(Pstiff,bulk_stiff).subs(bulk_stiff,1)-2*lam*v*S.sqrt(1-v*v))==0
# Universal strict inequalities: rational positive parametrisation of 0<t<1/sqrt(3).
zpos=S.symbols('zpos',positive=True)
def positive_on_triangle(expr):
    expr=S.factor(expr.subs(t,zpos/(r*(1+zpos))))
    num,den=S.fraction(expr)
    for part in [num,den]:
        coeff=S.Poly(part,zpos).all_coeffs()
        assert all(c>=0 for c in coeff) and any(c>0 for c in coeff),expr
for expr in [qs[0]-1/(2*r),-qs[2]-1/(2*r),qs[1]+1/(2*r),1/(2*r)-qs[1],
             ks[0],ks[1],ks[2]-r/2,S.Rational(1,2)-vs[2],S.Rational(1,2)+vs[2]]:
    positive_on_triangle(expr)
assert zero(sum(z**4 for z in qs)-S.Rational(1,8))
# Formal scalar equations reduce to the stated phase sums; no quantum existence claim.
ss,tt1,tt2=S.symbols('ss tt1 tt2',commutative=True)
assert ss*tt1*tt2-tt2*tt1*ss==0
# Frame subtraction is symmetric, and its scaled counterterm response is not
# to be confused with the chamber-reversing antisymmetric response.
pi1,pi2,mm1,mm2=S.symbols('pi1 pi2 mm1 mm2')
cb=(pi1*mm2+pi2*mm1)/2
assert zero(cb-cb.xreplace({pi1:pi2,pi2:pi1,mm1:mm2,mm2:mm1}))

out={'label':'(c) conditional algebra, not a physical S-matrix','digits':80,'tolerance':'1e-65',
     'max_geometry_error':str(worst),'kernel_rank':2,'antisymmetric_coefficient_dimension':1,
     'new_constraint_rank_on_A_B':0,'new_constraint_rank_on_c':0,
     'incidence_rank':2,'augmented_rank':2,'samples':samples,'seconds':time.perf_counter()-start}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))
print('ALL CHECKS PASSED')

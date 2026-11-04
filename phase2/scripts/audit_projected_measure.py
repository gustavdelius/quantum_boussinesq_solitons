#!/usr/bin/env python3
"""Projected collective symplectic measure audit; all results are (c).

No field determinant regularisation or asymptotic quantum convention is chosen.
Exact SymPy identities and 55-digit quadrature of the exact two-soliton family.
"""
import json
from functools import lru_cache
from pathlib import Path
import mpmath as mp
import sympy as sp
mp.mp.dps=55
RESULT={'precision':mp.mp.dps,'failures_recorded':['PDF rebuild failed after equation-tag cleanup left a blank paragraph inside a display; removed it.', 'Initial auxiliary 6D embedding used the opposite canonical sign, so Z^T J Z=-W instead of W. Corrected the embedding signs; no tolerances changed.']}

def text(x): return mp.nstr(x,35)

def pair(a,e):
    k,v=mp.sin(a),e*mp.cos(a)
    return (k+v/mp.sqrt(3))/2,(v/mp.sqrt(3)-k)/2

def scattering_A(a,e,b,f):
    pa,pb=pair(a,e);qa,qb=pair(b,f)
    return (pa-qa)*(pb-qb)/((pa-qb)*(pb-qa))

def matrix_algebra():
    d1,d2,c=sp.symbols('d1 d2 c',nonzero=True,real=True)
    D=sp.diag(d1,d2);C=sp.Matrix([[0,c],[-c,0]])
    W=C.row_join(D).col_join((-D).row_join(sp.zeros(2)))
    Winv=sp.zeros(2).row_join(-D.inv()).col_join(D.inv().row_join(D.inv()*C*D.inv()))
    assert W*Winv==sp.eye(4)
    assert sp.factor(W.det())==d1*d1*d2*d2
    # Shear X'_2=X_2-c/d2 * alpha_1 at this tangent point.
    F=sp.Matrix([[0,0],[-c/d2,0]])
    A=sp.eye(2).row_join(sp.zeros(2)).col_join(F.row_join(sp.eye(2)))
    J=sp.zeros(2).row_join(D).col_join((-D).row_join(sp.zeros(2)))
    assert A.T*J*A==W and A.det()==1
    # Embed this collective subspace in six-dimensional canonical phase space.
    J6=sp.zeros(3).row_join(sp.eye(3)).col_join((-sp.eye(3)).row_join(sp.zeros(3)))
    Z=sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,c,d1,0],[0,0,0,d2],[0,0,0,0]])
    assert Z.T*J6*Z==W
    Project=sp.simplify(Z*W.inv()*Z.T*J6)
    assert sp.simplify(Project*Project-Project)==sp.zeros(6)
    assert sp.simplify(Project*Z-Z)==sp.zeros(6,4)
    assert sp.simplify(Z.T*J6*(sp.eye(6)-Project))==sp.zeros(4,6)
    Za=Z*A.inv()
    other=sp.simplify(Za*(Za.T*J6*Za).inv()*Za.T*J6)
    assert other==Project
    # Fixed total-M tangent: da2=-m1/m2 da1. Its pulled-back form has rank two.
    m1,m2=sp.symbols('m1 m2',nonzero=True,real=True)
    Q=sp.Matrix([[1,0,0],[-m1/m2,0,0],[0,1,0],[0,0,1]])
    fixed=Q.T*W*Q
    null=sp.Matrix([0,m1/d1,m2/d2])
    assert fixed*null==sp.zeros(3,1) and fixed.rank()==2
    a,b,epsilon=sp.symbols('a b epsilon',real=True)
    assert sp.trigsimp(sp.diff(sp.Rational(2,3)*epsilon*sp.sin(a)**3*sp.cos(a),a)-sp.Rational(2,3)*epsilon*sp.sin(a)*sp.sin(3*a))==0
    assert sp.simplify(sp.expand_trig(sp.diff(sp.Rational(2,3)*epsilon*sp.sinh(b)**3*sp.cosh(b),b)-sp.Rational(2,3)*epsilon*sp.sinh(b)*sp.sinh(3*b)))==0
    RESULT['signed_momentum_derivatives']='Pgood_prime=(2/3)epsilon sin(alpha)sin(3alpha); Pbad_prime=(2/3)epsilon sinh(beta)sinh(3beta)'
    RESULT['exact_algebra']=['det W=(d1*d2)^2','Darboux shear determinant=1',
        'basis-independent symplectic projector','fixed-M restriction rank 2']
    print('1. Wronskian determinant, Darboux measure, projector, constrained rank: exact.',flush=True)

def flux_identity():
    x=sp.symbols('x');f=sp.Function('f')(x);g=sp.Function('g')(x);u=sp.Function('u')(x)
    L=lambda h:sp.diff(h,x,4)-sp.diff(h,x,2)+6*sp.diff(u*sp.diff(h,x),x)
    B=f*sp.diff(g,x,3)-sp.diff(f,x)*sp.diff(g,x,2)+sp.diff(f,x,2)*sp.diff(g,x)-sp.diff(f,x,3)*g+(6*u-1)*(f*sp.diff(g,x)-g*sp.diff(f,x))
    assert sp.expand(sp.diff(B,x)-(f*L(g)-g*L(f)))==0
    RESULT['flux_identity']='d_x B = f Lg - g Lf'
    print('2. Full fourth-order spatial symplectic boundary flux: exact.',flush=True)

def exact_tau_wronskians(a,b,e,f,t):
    k=[mp.sin(a),mp.sin(b)];v=[e*mp.cos(a),f*mp.cos(b)]
    kp=[mp.cos(a),mp.cos(b)];vp=[-e*mp.sin(a),-f*mp.sin(b)]
    w=[k[j]*v[j] for j in range(2)];wp=[kp[j]*v[j]+k[j]*vp[j] for j in range(2)]
    A=scattering_A(a,e,b,f);assert A>0 and v[0]>v[1]
    L=mp.log(A)
    La=[mp.diff(lambda aa:mp.log(scattering_A(aa,e,b,f)),a),mp.diff(lambda bb:mp.log(scattering_A(a,e,bb,f)),b)]
    bits=[(0,0),(1,0),(0,1),(1,1)]
    K=[sum(z[j]*k[j] for j in range(2)) for z in bits]
    W=[sum(z[j]*w[j] for j in range(2)) for z in bits]
    indices=[(i,j) for i in range(4) for j in range(i+1,4)]
    @lru_cache(maxsize=None)
    def modes(x):
        logs=[sum(z[j]*(k[j]*x-w[j]*t) for j in range(2))+z[0]*z[1]*L for z in bits]
        shift=max(logs);p=[mp.exp(z-shift) for z in logs];norm=sum(p);p=[z/norm for z in p]
        out=[]
        for h in range(4):
            j=h%2
            if h<2:
                dl=[z[j]*(kp[j]*x-wp[j]*t)+z[0]*z[1]*La[j] for z in bits]
                dK=[z[j]*kp[j] for z in bits];dW=[z[j]*wp[j] for z in bits]
            else:
                dl=[-z[j]*k[j] for z in bits];dK=[0]*4;dW=[0]*4
            ml=sum(p[i]*dl[i] for i in range(4))
            df=2*(sum(p[i]*dK[i] for i in range(4))+sum(p[i]*p[jj]*(K[i]-K[jj])*(dl[i]-dl[jj]) for i,jj in indices))
            dpi=-2*sum(p[i]*p[jj]*((dK[i]-dK[jj])*(W[i]-W[jj])+(K[i]-K[jj])*(dW[i]-dW[jj])+(K[i]-K[jj])*(W[i]-W[jj])*(dl[i]+dl[jj]-2*ml)) for i,jj in indices)
            out.append((df,dpi))
        return out
    result=mp.matrix(4)
    for i,j in indices:
        def integrand(x):
            z=modes(x);return z[i][1]*z[j][0]-z[j][1]*z[i][0]
        value=mp.quad(integrand,[-mp.inf,-20,0,20,mp.inf])
        result[i,j]=value;result[j,i]=-value
    ds=[mp.mpf(2)/3*e*mp.sin(a)*mp.sin(3*a),mp.mpf(2)/3*f*mp.sin(b)*mp.sin(3*b)]
    Mp=[2*kp[0],2*kp[1]];Pip=[-2*wp[0],-2*wp[1]]
    cin=ds[1]*La[0]/k[1]-Mp[0]*Pip[1]
    cout=-ds[0]*La[1]/k[0]+Pip[0]*Mp[1]
    expected=mp.matrix([[0,cin,ds[0],0],[-cin,0,0,ds[1]],[-ds[0],0,0,0],[0,-ds[1],0,0]])
    err=max(abs(result[i,j]-expected[i,j]) for i in range(4) for j in range(4))
    assert err<mp.mpf('1e-43') and abs(cin-cout)<mp.mpf('1e-48')
    ratio=mp.det(result)/(ds[0]*ds[1])**2
    assert abs(ratio-1)<mp.mpf('1e-42')
    return dict(a=text(a),b=text(b),eps1=e,eps2=f,time=text(t),A=text(A),
                mixed_pairing=text(cin),matrix_error=text(err),determinant_ratio=text(ratio),
                matrix=[[text(result[i,j]) for j in range(4)] for i in range(4)])

def numerical_collision_checks():
    rows=[]
    for sa,sb,e,f,st in [('.3','.7',1,1,'-8'),('.3','.7',1,1,'0'),('.3','.7',1,1,'8'),('.25','.45',1,-1,'0')]:
        row=exact_tau_wronskians(mp.mpf(sa),mp.mpf(sb),e,f,mp.mpf(st));rows.append(row)
        print('3. Exact tau collective Wronskian',sa,sb,e,f,st,'error',row['matrix_error'],flush=True)
    RESULT['collision_wronskians']=rows

def boundary_and_phase_controls():
    # Finite matrix benchmark only: no field-theory regulator is selected.
    m,h=sp.symbols('m h',nonzero=True,real=True)
    Nsym=sp.symbols('N',integer=True,positive=True)
    # Laplace expansion: D_N=2 D_(N-1)-D_(N-2), D_0=1,D_1=2.
    assert sp.expand((Nsym+1)-2*Nsym+(Nsym-1))==0
    assert (Nsym+1).subs(Nsym,0)==1 and (Nsym+1).subs(Nsym,1)==2
    for N in range(1,7):
        K=sp.diag(*([2]*N))
        for j in range(N-1):K[j,j+1]=K[j+1,j]=-1
        assert (m/h*K).det()==(N+1)*(m/h)**N
    x,t=sp.symbols('x t',real=True)
    ground=sp.exp(-x*x/2)
    assert sp.simplify((-sp.diff(ground,x,2)+x*x*ground)/2-ground/2)==0
    assert sp.exp(-sp.I*2*sp.pi/2)==-1
    R=sp.Matrix([[sp.cos(t),sp.sin(t)],[-sp.sin(t),sp.cos(t)]])
    J=sp.Matrix([[0,1],[-1,0]])
    assert sp.simplify(R.T*J*R)==J and R.subs(t,2*sp.pi)==sp.eye(2)
    RESULT['boundary_phase_controls']=['Dirichlet free-particle Hessian invertible',
        'classical oscillator 2pi flow identity but quantum ground-state phase -1',
        'General-N Dirichlet determinant proved by exact recurrence and base cases']
    print('4. Boundary-condition and phase-versus-volume controls: exact.',flush=True)


def fixed_boundary_projector_control():
    z,k,c,d=sp.symbols('z k c d',positive=True)
    norm=sp.integrate(k**3*(1-z*z)/2,(z,-1,1))
    assert sp.simplify(norm-sp.Rational(2,3)*k**3)==0
    # eta=(0,u), so b_X=integral u^2 and the alpha coefficient is -b_X/P'.
    W=sp.Matrix([[0,d],[-d,0]])
    ba=sp.symbols('ba',real=True)
    coeff=W.inv()*sp.Matrix([ba,norm])
    assert sp.simplify(coeff[0]+norm/d)==0
    tail=sp.simplify(2*c*coeff[0])
    assert tail==-sp.Rational(4,3)*c*k**3/d
    a=mp.mpf('.4')
    dp=mp.mpf(2)/3*mp.sin(a)*mp.sin(3*a)
    value=-4*mp.cos(a)*mp.sin(a)**3/(3*dp)
    assert abs(value)>mp.mpf('.1')
    RESULT['fixed_boundary_projection']=dict(test='eta_phi=0, eta_pi=u',
        alpha='.4',projected_phi_right_tail=text(value),formula='-4 cos(alpha) sin(alpha)^3/(3 Pprime)')
    print('5. Localized perturbation projected to a nonzero potential tail: exact.',flush=True)

if __name__=='__main__':
    matrix_algebra();flux_identity();numerical_collision_checks();boundary_and_phase_controls();fixed_boundary_projector_control()
    Path(__file__).with_name('audit_projected_measure_results.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print('ALL CHECKS PASSED')

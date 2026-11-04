#!/usr/bin/env python3
"""Independent half-line factor, leading-log subtraction, and dispersion checks.
No earlier Wiener--Hopf formula is imported. Default validates saved results.
--recompute runs two inverse-Laplace algorithms at 50 and 70 digits.
"""
from pathlib import Path
import json,time,sys,hashlib
import mpmath as m
import sympy as s
HERE=Path(__file__).resolve().parent;CACHE=HERE/'onebody_prec_edge.json'
START=time.monotonic();m.mp.dps=70
# Exact differential/thermodynamic algebra.
a,q,h,y,A,B,d,e,ss=s.symbols('a q h y A B d e ss',positive=True)
n=-1/(2*s.log(q/2));amp=n*s.sqrt(2*h/q);vs=q/(2*n)
assert s.simplify(1-vs*s.diff(amp,q)/amp-1/(4*n))==0
bb=s.exp(-a/2);e0=bb**2; e1=2*bb/s.pi
assert s.simplify(a*s.diff(e1,a,2)/(2*bb)-a/(4*s.pi))==0
k=s.symbols('k',positive=True)
assert s.integrate(2*bb*s.sin(k/2),(k,0,2*s.pi))/(4*s.pi)==2*bb/s.pi
P=4*bb*(k*s.cosh(k)-s.sinh(k))/a;E=2*bb**2*(s.sinh(k)*s.cosh(k)-k)
assert s.simplify(s.diff(E,k)/s.diff(P,k)-a*bb*s.sinh(k)/k)==0
assert s.limit((E-a*bb*P)/(h*P)*h/(bb*k*k),k,0)==a/s.Integer(10)
# Leading log and power tails: integration uses exact dressed momentum.
u=y-A*s.log(y)+B+d/y+e/y**2
F=4*y**s.Rational(3,2)/3
J=8*y**s.Rational(5,2)/15-8*A*y**s.Rational(3,2)/9-8*d*s.sqrt(y)/3+16*e/(3*s.sqrt(y))
assert s.simplify(s.diff(J,y)-F*s.diff(u,y))==0
C=s.simplify(3*ss+3*J/(4*F)-3*y/10)
assert s.simplify(C-(3*ss-A/2-3*d/(2*y)+3*e/y**2))==0
Cv=s.simplify(3*ss+y*(s.diff(u,y)-1)/2)
assert s.simplify(Cv-(3*ss-A/2-d/(2*y)-e/y**2))==0
# Derive rather than assume the first two momentum inverse-series coefficients.
z,L,alpha=s.symbols('z L alpha',positive=True)
AA=1/(2*s.pi);BB=1+s.log(2*s.pi)-s.EulerGamma
# Inverse Laplace of s^-alpha (log s)^j, evaluated after removing u power.
def ilog(al,j):
    expr=(-1)**j*s.diff(s.exp((alpha-1)*L)/s.gamma(alpha),alpha,j)
    return s.simplify(s.expand_func(expr.subs(alpha,al))/s.exp((al-1)*L))
f1=s.simplify(3*s.sqrt(s.pi)/4*AA*(BB*ilog(s.Rational(3,2),0)-ilog(s.Rational(3,2),1)))
f2=s.simplify(3*s.sqrt(s.pi)/4*((s.Rational(1,48)+AA**2*BB**2/2)*ilog(s.Rational(1,2),0)-AA**2*BB*ilog(s.Rational(1,2),1)+AA**2/2*ilog(s.Rational(1,2),2)))
a1=s.simplify(s.Rational(2,3)*f1);a2=s.simplify(s.Rational(2,3)*f2-s.Rational(1,9)*f1*f1)
cedge=(1-s.log(8*s.pi))/(2*s.pi);d0=s.Rational(1,48)-1/(4*s.pi**2)
assert s.simplify(a1-(AA*L-cedge))==0
# y=u+a1(log u)+a2(log u)/u; inverse has d0=A*a1-a2.
assert s.simplify(AA*a1-a2-d0)==0
assert s.simplify((3*ss-AA/2).subs(ss,1/(4*s.pi))-1/(2*s.pi))==0

def g(z):return m.exp(m.loggamma(1+z/(2*m.pi))-z/(2*m.pi)*m.log(z/(2*m.pi*m.e)))/m.sqrt(z)
# Factorisation at nonzero frequency follows from the digamma integral.
# Khat(w)=-2*pi/(exp(|w|)-1); sigma=2*pi-Khat.
factors=[]
for w in map(m.mpf,['.03','.2','1','3','10']):
    product=2*m.pi*g(1j*w)*g(-1j*w)
    sigma=2*m.pi+2*m.pi/m.expm1(w)
    err=abs(product-sigma);assert err<m.mpf('1e-64')
    factors.append(dict(w=str(w),error=str(err)))
# Classical normalization is essential: L[(4/3)u^(3/2)]=sqrt(pi)s^(-5/2).
assert abs(4*m.gamma(m.mpf('2.5'))/3-m.sqrt(m.pi))<m.mpf('1e-68')

def inverse(u,n,method):return m.invertlaplace(lambda z:m.sqrt(m.pi)*g(z)/z**n,u,method=method)
def point(y,dps,method):
    with m.workdps(dps):
        yy=m.mpf(str(y));target=4*yy**m.mpf('1.5')/3
        guess=yy-m.log(yy)/(2*m.pi)+(1-m.log(8*m.pi))/(2*m.pi)
        uu=m.findroot(lambda x:inverse(x,2,method)-target,(guess,guess+m.mpf('.01')),solver='secant',tol=m.mpf(10)**(-dps+8))
        FF=inverse(uu,2,method);ff=inverse(uu,1,method);JJ=inverse(uu,3,method)
        ce=3/(4*m.pi)+3*JJ/(4*FF)-3*yy/10
        cv=3/(4*m.pi)+3*FF/(4*ff)-yy/2
        up=2*m.sqrt(yy)/ff
        return dict(y=str(yy),u=str(uu),energy=str(ce),velocity=str(cv),du=str(up),F=str(FF),J=str(JJ),dps=dps,method=method)
if '--recompute' in sys.argv or not CACHE.exists():
    rows=[]
    for yy in [4,8,16,32,64,128,256,512,1024]:
        st=time.monotonic();row=point(yy,50,'dehoog')
        # Independent contour and precision, not just reevaluation at same precision.
        other=point(yy,70,'talbot')
        err=max(abs(m.mpf(row[key])-m.mpf(other[key])) for key in ['u','energy','velocity','du'])
        assert err<m.mpf('1e-35'),(yy,err)
        row['cross_method_error']=str(err);row['seconds']=time.monotonic()-st
        rows.append(row)
        CACHE.write_text(json.dumps(dict(rows=rows,factor_checks=factors,d0=str(m.mpf(1)/48-1/(4*m.pi**2)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
        print('y',yy,'C',row['energy'][:24],'cross check',m.nstr(err,3),'seconds',round(row['seconds'],1),flush=True)
else:rows=json.loads(CACHE.read_text())['rows']
assert len(rows)==9
for row in rows:
    assert m.mpf(row['cross_method_error'])<m.mpf('1e-35')
    yy=m.mpf(row['y']);uu=m.mpf(row['u']);dd=m.mpf(1)/48-1/(4*m.pi**2)
    lead=yy-m.log(yy)/(2*m.pi)+(1-m.log(8*m.pi))/(2*m.pi)+dd/yy
    assert abs(uu-lead)*yy**2<m.mpf('.0006')
print('ALL CHECKS PASSED')

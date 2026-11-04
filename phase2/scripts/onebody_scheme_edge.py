#!/usr/bin/env python3
"""Diagnose the failed audit bound with fresh Laplace and finite-sea solves.

No old script is imported. Saved rows are read only and clearly distinguished
from fresh calculations. All output filenames have the task prefix.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import hashlib
import time
import numpy as np
from scipy.special import roots_legendre, digamma, loggamma
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root_scalar,brentq
import mpmath as m
import sympy as s

start=time.monotonic(); m.mp.dps=60
here=Path(__file__).resolve().parent
inputs=['onebody_toda_results.json','frame_analysis.json','onebody_prec_scan.json','onebody_prec_analysis.json']
hashes={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in inputs}
old=json.loads((here/inputs[0]).read_text(),parse_float=m.mpf)
frame=json.loads((here/inputs[1]).read_text(),parse_float=m.mpf)
prec=json.loads((here/inputs[2]).read_text(),parse_float=m.mpf)['rows']
analysis=json.loads((here/inputs[3]).read_text())
diagnosis=[]
for r in old['rows']:
    yy=r['y'];f=next(t for t in frame['one_body'] if t['y']==yy)
    residual=r['velocity']['value']-old['sound']['value']-yy*f['jacobian_minus_one']['value']/2
    spread=r['velocity']['edge_spread']+old['sound']['edge_spread']+yy*f['jacobian_minus_one']['edge_spread']/2
    assert abs(residual)<spread
    diagnosis.append({'y':str(yy),'residual':str(residual),'spread':str(spread)})
maximum=max(abs(m.mpf(r['residual'])) for r in diagnosis)
# Reproduce the failure as a positive diagnostic. The original assertion and
# tolerance are not changed or misrepresented as passing.
assert maximum>m.mpf('3e-6')
assert abs(maximum-m.mpf('0.00022546734881565'))<m.mpf('1e-15')
print('Original maximum residual:',m.nstr(maximum,16),flush=True)

# Independent high-precision fits of the finer, previously saved SEA outputs.
# A fit of the residual is linear-equivalent to fitting its three constituents.
base=[r for r in prec if r['factor']==1]
def fit_residual(yy,last,degree):
    rr=base[-last:];hs=[r['h'] for r in rr];ys=[]
    for r in rr:
        p=next(t for t in r['points'] if t['y']==yy)
        ys.append(p['velocity']-r['sound']-yy*p['du_remainder']/2)
    A=m.matrix([[(h/hs[0])**j for j in range(degree+1)] for h in hs])
    c=m.lu_solve(A.T*A,A.T*m.matrix(ys))
    return c[0]
refined=[]
for yy in (16,32,64):
    q4=fit_residual(yy,7,4);q5=fit_residual(yy,8,5);q3=fit_residual(yy,6,3)
    assert abs(q4)<m.mpf('3e-6')
    refined.append({'y':yy,'quartic7':str(q4),'quintic8':str(q5),'cubic6':str(q3)})
print('Refined residuals:',refined,flush=True)

# Derive the sound coefficient and edge logarithm, with correct Fourier scale.
z,y,al,LL=s.symbols('z y al LL',positive=True)
AA=1/(2*s.pi);BB=1+s.log(2*s.pi)-s.EulerGamma
# First correction to F=(4/3)u^(3/2), from inverse Laplace of g(s)/s^2.
inv0=s.exp((al-1)*LL)/s.gamma(al)
correction=s.simplify(s.sqrt(s.pi)*AA*(BB*inv0+s.diff(inv0,al)).subs(al,s.Rational(3,2)))
relative=s.simplify(s.expand_func(correction)/s.exp(LL/2))
# y(u)=u + (1/2) correction/u^(1/2) + ...;
# hence the inverse u(y) has coefficient -1/(2pi) on log y.
assert s.simplify(s.diff(relative/2,LL)-AA)==0
cedge=s.simplify(-(relative/2).subs(LL,0))
assert s.simplify(cedge-(1-s.log(8*s.pi))/(2*s.pi))==0
u_asym=y-AA*s.log(y)+cedge
Cv=s.Rational(3,4)/s.pi+y*(s.diff(u_asym,y)-1)/2
assert s.simplify(Cv)==1/(2*s.pi)
# Integrating C_v dP with P proportional to y^(3/2) gives the SAME constant.
F=4*y**s.Rational(3,2)/3
J=s.integrate(F*s.diff(u_asym,y),y)
assert s.simplify(3/(4*s.pi)+3*J/(4*F)-3*y/10)==1/(2*s.pi)
# Audit's written 3 e0/(2y^2) energy tail is incorrect: the result is 3 e0/y^2.
e=s.symbols('e');J2=s.integrate(F*s.diff(e/y**2,y),y)
assert s.simplify(3*J2/(4*F))==3*e/y**2

def g(ss):
    return m.exp(m.loggamma(1+ss/(2*m.pi))-ss*m.log(ss/(2*m.pi*m.e))/(2*m.pi))/m.sqrt(ss)
for freq in map(m.mpf,['0.07','0.8','4']):
    assert abs(2*m.pi*g(1j*freq)*g(-1j*freq)-2*m.pi/(1-m.exp(-freq)))<m.mpf('1e-54')
def il(u,power,method):
    return m.invertlaplace(lambda ss:m.sqrt(m.pi)*g(ss)/ss**power,u,method=method)
halfline=[]
for yy in (16,64):
    with m.workdps(45):
        yy=m.mpf(yy);target=4*yy**m.mpf('1.5')/3
        uu=yy-m.log(yy)/(2*m.pi)+(1-m.log(8*m.pi))/(2*m.pi)
        for iteration in range(6):
            ff=il(uu,1,'talbot');FF=il(uu,2,'talbot')
            step=(FF-target)/ff;uu-=step
            if abs(step)<m.mpf('1e-38'):break
        assert abs(il(uu,2,'talbot')-target)<m.mpf('1e-37')
        ff=il(uu,1,'talbot');JJ=il(uu,3,'talbot')
        # Cross-contour check at the independently solved root, all three transforms.
        cross=max(abs(il(uu,j,'dehoog')-il(uu,j,'talbot')) for j in (1,2,3))
        assert cross<m.mpf('1e-35')
        du=2*m.sqrt(yy)/ff
        cv=3/(4*m.pi)+3*target/(4*ff)-yy/2
        ce=3/(4*m.pi)+3*JJ/(4*target)-3*yy/10
        assert abs(cv-(3/(4*m.pi)+yy*(du-1)/2))<m.mpf('1e-40')
        oldr=next(r for r in old['rows'] if r['y']==yy)
        newr=next(r for r in analysis['rows'] if m.mpf(r['y'])==yy)
        error_old=oldr['velocity']['value']-cv
        error_new=m.mpf(newr['velocity']['value'])-cv
        assert abs(error_new)<m.mpf('3e-6')
        halfline.append({'y':str(yy),'u':str(uu),'du':str(du),'Cv':str(cv),'CE':str(ce),
                         'cross_contour_error':str(cross),'old_Cv_error':str(error_old),'refined_Cv_error':str(error_new)})
        print('Fresh half-line y',yy,'Cv',m.nstr(cv,18),'old error',m.nstr(error_old,6),'new error',m.nstr(error_new,6),flush=True)

# Independent composite Gauss sea solver, rather than the historical single
# interval mesh. The count equation, p inversion, and endpoint identity are
# checked without target C_T entering the numerical solver.
class Sea:
    def __init__(self,hh,order):
        self.h=hh;self.b=np.exp(-.5)
        nseg=int(np.ceil(1/(2*hh)));edges=np.linspace(-1,1,nseg+1)
        zz,ww=roots_legendre(order)
        z=np.concatenate([(lo+hi)/2+(hi-lo)*zz/2 for lo,hi in zip(edges[:-1],edges[1:])])
        w=np.concatenate([(hi-lo)*ww/2 for lo,hi in zip(edges[:-1],edges[1:])])
        def count(q,keep=False):
            x=q*z;weights=q*w
            mat=2*np.pi*hh*np.eye(len(x))-self.K(x[:,None]-x[None,:])*weights[None,:]
            lu=lu_factor(mat);rho=lu_solve(lu,np.ones(len(x)))
            if keep:self.x,self.w,self.rho,self.lu=x,weights,rho,lu
            return weights@rho-1
        sol=root_scalar(count,x0=2*self.b+hh,x1=2*self.b+2*hh,xtol=1e-13)
        assert sol.converged
        self.q=sol.root;assert abs(count(self.q,True))<2e-11
        self.eta=lu_solve(self.lu,self.x)
        self.vs=self.V(self.q)/self.D(self.q)
        endpoint_error=abs(self.D(self.q)*self.V(self.q)-np.pi*hh)
        assert endpoint_error<2e-11,(hh,order,endpoint_error)
        self.N=len(z)
    def K(self,d):return 2*np.real(digamma(1+1j*d/self.h))+2*np.log(self.h)
    def T(self,d):return 2*self.h*np.imag(loggamma(1+1j*d/self.h))+2*d*np.log(self.h)
    def D(self,p):return 1+np.dot(self.w*self.rho,self.K(p-self.x))
    def V(self,p):return p+np.dot(self.w*self.eta,self.K(p-self.x))
    def P(self,p):return p-self.q+np.dot(self.w*self.rho,self.T(p-self.x)-self.T(self.q-self.x))
    def point(self,yy):
        kk=np.sqrt(self.h*yy/self.b)
        target=4*self.b*(kk*np.cosh(kk)-np.sinh(kk))
        pp=brentq(lambda p:self.P(p)-target,self.q,self.q+4*self.h*yy+1,xtol=5e-15)
        assert abs(self.P(pp)-target)<2e-11
        cv=3*(self.V(pp)/self.D(pp)-self.b*np.sinh(kk)/kk)/self.h
        sound=3*(self.vs-self.b)/self.h
        jac=2*np.sinh(kk)/self.D(pp)-np.sinh(kk)/kk
        return dict(y=yy,Cv=float(cv),sound=float(sound),jac=float(jac),residual=float(cv-sound-yy*jac/2))
fresh=[]
for hh,order in ((.02,32),(.01,32),(.005,32),(.005,40)):
    sea=Sea(hh,order);row={'h':hh,'N':sea.N,'points':[sea.point(y) for y in (16,64)]}
    fresh.append(row)
    print('Fresh sea h',hh,'N',sea.N,'residual at 64',row['points'][-1]['residual'],flush=True)
# Mesh errors must be small compared with the failed extrapolation assertion.
mesh=max(abs(p[key]-q[key]) for p,q in zip(fresh[-2]['points'],fresh[-1]['points']) for key in ('Cv','jac','residual'))
assert mesh<1e-7
for i in (0,1):
    assert abs(fresh[2]['points'][i]['residual'])<abs(fresh[1]['points'][i]['residual'])<abs(fresh[0]['points'][i]['residual'])
for row in fresh:
    cached=next(r for r in base if abs(float(r['h'])-row['h'])<1e-12)
    for pp in row['points']:
        cp=next(t for t in cached['points'] if t['y']==pp['y'])
        assert abs(m.mpf(pp['Cv'])-cp['velocity'])<m.mpf('1e-7')

out={'original_bound':'3e-6 (failed, unchanged)','maximum_original_residual':str(maximum),
     'original_rows':diagnosis,'refined_residual_fits':refined,'fresh_halfline':halfline,
     'fresh_sea':fresh,'fresh_mesh_difference':mesh,'input_sha256':hashes,
     'seconds':time.monotonic()-start,
     'failures_recorded':['Original audit residual 2.254673488e-4 exceeds 3e-6.',
       'Fresh composite order 16 at h=.02 failed endpoint tolerance 2e-11 with 3.3117711365e-8.',
       'Order 24 also failed with 4.4067346994e-11; increased to orders 32 and 40 without changing tolerance.'],
     'limitations':['The identity is an edge limit, not an exact finite-h identity.',
       'Fits diagnose convergence; they are not rigorous uniform error bounds.',
       'The fixed-nonzero-P interchange of limits is not proved.']}
for name,digest in hashes.items():assert hashlib.sha256((here/name).read_bytes()).hexdigest()==digest
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL CHECKS PASSED')

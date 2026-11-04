#!/usr/bin/env python3
"""Independent finite-sea two-body solve; writes only twobody_toda_scan.json.
Float64 Nystrom solves, mpmath classical subtraction. No old script imports.
Default resumes complete rows; --refresh recomputes. Assertions are unchanged
between fresh and cached runs. Total grid is deliberately below 30 minutes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,sys,time
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import roots_legendre,digamma,loggamma
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root_scalar,brentq
mp.mp.dps=60
HERE=Path(__file__).resolve().parent
CACHE=HERE/'twobody_toda_scan.json'
HS=[.08,.06,.04,.03,.02,.015,.01,.0075,.005,.00375,.0025]
PAIRS=[(Y,r) for Y in [4.,8.,16.,32.,64.] for r in [.4,.5,.6]]

class Sea:
    def __init__(self,h,N,qseed=None):
        self.h=h;self.N=N;self.t=2*np.pi*h;self.b=np.exp(-.5)
        z,w=roots_legendre(N)
        self.weight_correction=0.;self.quadrature='scipy roots_legendre'
        if N>=5000:
            # Recompute high-order weights from a long-double Legendre recurrence.
            # At thousands of nodes scipy's weights can lose enough endpoint
            # accuracy to fail the unchanged integration-by-parts reciprocity test.
            zz=np.asarray(z,dtype=np.longdouble)
            def recurrence(xx):
                p0=np.ones_like(xx);p1=xx.copy()
                for k in range(2,N+1):p0,p1=p1,((2*k-1)*xx*p1-(k-1)*p0)/k
                return p1,N*(xx*p1-p0)/(xx*xx-1)
            for unused in range(2):
                pp,dp=recurrence(zz);zz-=pp/dp
            pp,dp=recurrence(zz)
            ww=2/((1-zz*zz)*dp*dp)
            self.weight_correction=float(np.max(abs(ww-w)))
            z=np.asarray(zz,dtype=float);w=np.asarray(ww,dtype=float)
            for k in [0,2,4,16,32]:assert abs(np.dot(w,z**k)-2/(k+1))<3e-15
            self.quadrature='long-double Newton/Legendre weights, float64 sea'
        def count(q,save=False):
            x=q*z;ww=q*w
            lu=lu_factor(self.t*np.eye(N)-self.K(x[:,None]-x[None,:])*ww[None,:])
            rho=lu_solve(lu,np.ones(N))
            if save:self.x,self.w,self.lu,self.rho=x,ww,lu,rho
            return ww@rho-1
        sol=root_scalar(count,x0=2*self.b+h if qseed is None else qseed,
                        x1=2*self.b+2*h if qseed is None else qseed+1e-8,xtol=2e-14)
        assert sol.converged
        self.q=sol.root;self.count_error=abs(count(self.q,True))
        self.eta=lu_solve(self.lu,self.x)
        self.Dq=self.D(self.q);self.Vq=self.V(self.q)
        self.edge_error=abs(self.q+self.dot(self.Th(self.q-self.x),self.rho)-np.pi*h)
    def K(self,d):return 2*np.real(digamma(1+1j*d/self.h))+2*np.log(self.h)
    def Th(self,d):return 2*self.h*np.imag(loggamma(1+1j*d/self.h))+2*d*np.log(self.h)
    def dot(self,f,g):return np.dot(self.w*f,g)
    def D(self,p):return 1+self.dot(self.K(p-self.x),self.rho)
    def V(self,p):return p+self.dot(self.K(p-self.x),self.eta)
    def relP(self,p):return p-self.q+self.dot(self.Th(p-self.x)-self.Th(self.q-self.x),self.rho)
    def point(self,y,sign):
        hm=mp.mpf(str(self.h));ym=mp.mpf(str(y));bm=mp.exp(-mp.mpf('.5'))
        km=mp.sqrt(hm*ym/bm);Pm=4*bm*(km*mp.cosh(km)-mp.sinh(km))
        pp=brentq(lambda p:self.relP(p)-float(Pm),self.q,self.q+max(2.,4*self.h*y),xtol=5e-15)
        p=sign*pp;e=sign*self.q
        df=lu_solve(self.lu,self.Th(p-self.x)-self.Th(e-self.x))
        ends=np.array([(self.Th(p-z)-self.Th(e-z)+self.dot(self.K(z-self.x),df))/self.t for z in [self.q,-self.q]])
        M=self.Dq*(ends[0]-ends[1]);Q=np.pi*self.h*(ends[0]+ends[1])/self.Dq
        d=self.D(p);v=self.V(p)
        return dict(y=y,sign=sign,k=float(km),P=sign*float(Pm),p=p,e=e,df=df,ends=ends,M=M,Q=Q,
                    u=(pp-self.q)/self.h,du=2*np.sinh(float(km))/d,
                    momentum_error=abs(self.relP(pp)-float(Pm)),
                    endpoint_error=max(abs(M+(d-self.Dq)),abs(Q-(sign*float(Pm)-v+sign*self.Vq)/(2*self.Dq*self.Vq/self.t))))
    def directional(self,p,z):
        return self.Th(p['p']-z['p'])-self.Th(p['p']-z['e'])-self.dot(self.K(p['p']-self.x),z['df'])
    def U(self,p,z):
        return self.Th(p['e']-z['p'])-self.Th(p['e']-z['e'])-self.dot(self.K(p['e']-self.x),z['df'])
    def pair(self,p,z):
        n12=self.directional(p,z);n21=self.directional(z,p);u12=self.U(p,z);u21=self.U(z,p)
        f12=n12-u12;f21=n21-u21
        sym=self.t*(p['ends'][0]*z['ends'][0]-p['ends'][1]*z['ends'][1])
        # Full classical Toda phase and directional frame term, evaluated at 60 digits.
        b=mp.exp(-mp.mpf('.5'));h=mp.mpf(str(self.h))
        k1=mp.sqrt(mp.mpf(str(p['y']))*h/b);k2=mp.sqrt(mp.mpf(str(z['y']))*h/b)
        s1=p['sign'];s2=z['sign']
        if s1==s2:la=2*mp.log(abs(mp.sinh((k1-k2)/2)/mp.sinh((k1+k2)/2)))
        else:la=2*mp.log(mp.cosh((k1-k2)/2)/mp.cosh((k1+k2)/2))
        delta=(s1*2*b*mp.cosh(k1)-s2*2*b*mp.cosh(k2))*la+4*k1*k2*b*(s1*mp.sinh(k1)/k1-s2*mp.sinh(k2)/k2)
        c12=-2*k1*s2*(4*b*(k2*mp.cosh(k2)-mp.sinh(k2))-2*b*mp.sinh(k2))
        c21=-2*k2*s1*(4*b*(k1*mp.cosh(k1)-mp.sinh(k1))-2*b*mp.sinh(k1))
        r12=n12/self.h-float((delta+c12)/h)
        r21=n21/self.h-float((-delta+c21)/h)
        return dict(y1=p['y'],y2=z['y'],s1=s1,s2=s2,N12=n12/self.h,N21=n21/self.h,U12=u12/self.h,U21=u21/self.h,
                    four12=f12/self.h,four21=f21/self.h,endpoint_symmetric=sym/self.h,
                    endpoint_B12=(f12-p['M']*z['Q'])/self.h,
                    reciprocity_error=abs(f12+f21-sym)/self.h,
                    R12=r12,R21=r21,R_symmetric=(r12+r21)/2,R_antisymmetric=(r12-r21)/2,
                    u1=p['u'],u2=z['u'],du1=p['du'],du2=z['du'])

def main():
    rows=[] if '--refresh' in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text())['rows']
    jobs=[(h,1.) for h in HS]+[(h,1.25) for h in [.01,.005,.0025]]
    for h,f in jobs:
        if any(r['h']==h and r['factor']==f for r in rows):continue
        start=time.monotonic();seed=next((r['q'] for r in rows if r['h']==h),None)
        m=Sea(h,int(np.ceil(f*(10/h+200))),seed)
        ys=sorted(set([Y for Y,r in PAIRS]+[Y*r*r for Y,r in PAIRS]))
        pts={(y,s):m.point(y,s) for y in ys for s in [1,-1]}
        pairs=[m.pair(pts[Y,1],pts[Y*r*r,s]) for Y,r in PAIRS for s in [1,-1]]
        row=dict(h=h,N=m.N,factor=f,q=m.q,quadrature=m.quadrature,weight_correction=m.weight_correction,sound=3*(m.Vq/m.Dq-m.b)/h,count_error=m.count_error,edge_error=m.edge_error,
                 endpoint_error=max(p['endpoint_error'] for p in pts.values()),momentum_error=max(p['momentum_error'] for p in pts.values()),pairs=pairs,seconds=time.monotonic()-start)
        assert row['count_error']<2e-11 and row['edge_error']<2e-10
        assert row['endpoint_error']<2e-9 and row['momentum_error']<2e-11
        assert max(p['reciprocity_error'] for p in pairs)<2e-8,(h,m.N,max(p['reciprocity_error'] for p in pairs))
        rows.append(row);CACHE.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
        p=next(p for p in pairs if p['y1']==16 and p['y2']==4 and p['s2']==1)
        print('h',h,'N',m.N,'R12',p['R12'],'R21',p['R21'],'U12',p['U12'],'seconds',round(row['seconds'],1),flush=True)
    assert len(rows)==len(jobs)
    for row in rows:
        assert row['count_error']<2e-11 and row['edge_error']<2e-10
        assert row['endpoint_error']<2e-9 and row['momentum_error']<2e-11
        assert max(p['reciprocity_error'] for p in row['pairs'])<2e-8
    print('ALL CHECKS PASSED')

if __name__=='__main__':main()

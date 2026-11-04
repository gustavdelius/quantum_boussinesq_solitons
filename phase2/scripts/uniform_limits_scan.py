#!/usr/bin/env python3
"""Finite-sea fixed-momentum and edge tests, adapted from the audited Sea algorithm.
Only uniform_limits_scan.json is written; existing scripts are not imported.
Default validates cache, --refresh rebuilds. BLAS is limited to one thread.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import json,time,sys
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import roots_legendre,digamma,loggamma
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root_scalar,brentq
mp.mp.dps=60
HERE=Path(__file__).resolve().parent;CACHE=HERE/'uniform_limits_scan.json'
class Sea:
    def __init__(self,h,N):
        self.h=h;self.N=N;self.t=2*np.pi*h;self.b=np.exp(-.5)
        z,w=roots_legendre(N)
        def count(q,save=False):
            x=q*z;ww=q*w
            A=self.t*np.eye(N)-self.K(x[:,None]-x[None,:])*ww[None,:]
            lu=lu_factor(A);rho=lu_solve(lu,np.ones(N))
            if save:self.x,self.w,self.lu,self.rho=x,ww,lu,rho
            return ww@rho-1
        sol=root_scalar(count,x0=2*self.b+h,x1=2*self.b+2*h,xtol=2e-14)
        assert sol.converged
        self.q=sol.root;self.count_error=abs(count(self.q,True));assert self.count_error<2e-11
        self.eta=lu_solve(self.lu,self.x)
        self.Dq=self.D(self.q);self.Vq=self.V(self.q);self.vs=self.Vq/self.Dq
        self.edge_error=abs(self.q+self.dot(self.Th(self.q-self.x),self.rho)-np.pi*h)
        assert self.edge_error<2e-10
    def K(self,d):return 2*np.real(digamma(1+1j*d/self.h))+2*np.log(self.h)
    def Th(self,d):return 2*self.h*np.imag(loggamma(1+1j*d/self.h))+2*d*np.log(self.h)
    def dot(self,f,g):return np.dot(self.w*f,g)
    def D(self,p):return 1+self.dot(self.K(p-self.x),self.rho)
    def V(self,p):return p+self.dot(self.K(p-self.x),self.eta)
    def P(self,p):return p-self.q+self.dot(self.Th(p-self.x)-self.Th(self.q-self.x),self.rho)
    def point(self,y):
        hm=mp.mpf(str(self.h));bm=mp.exp(-mp.mpf('.5'));km=mp.sqrt(hm*mp.mpf(str(y))/bm)
        pm=4*bm*(km*mp.cosh(km)-mp.sinh(km));em=2*bm**2*(mp.sinh(km)*mp.cosh(km)-km)
        k=float(km);target=float(pm)
        p=brentq(lambda p:self.P(p)-target,self.q,self.q+max(2.,4*self.h*y),xtol=5e-15)
        th=self.Th(p-self.x)-self.Th(self.q-self.x)
        # Subtract the sound line inside the integral to reduce cancellation.
        # E-bP=(p-q)((p+q)/2-b)+<theta_diff,eta-b rho>.
        nonlinear=(p-self.q)*((p+self.q)/2-self.b)+self.dot(th,self.eta-self.b*self.rho)
        clnon=float(em-bm*pm)
        energy=3*(nonlinear-clnon)/(self.h*target)
        d=self.D(p);v=self.V(p)
        velocity=3*(v/d-float(bm*mp.sinh(km)/km))/self.h
        du=2*np.sinh(k)/d
        urem=(p-self.q)/self.h-float(2*bm*(mp.cosh(km)-1)/hm)
        # Exact finite-sea identity V'=D-v_s partial_q D, tested without q differencing.
        Kp=self.K(p-self.x);end=self.K(self.x-self.q)+self.K(self.x+self.q)
        rhoq=self.Dq/self.t; rq=lu_solve(self.lu,rhoq*end)
        Dqpartial=rhoq*(self.K(p-self.q)+self.K(p+self.q))+self.dot(Kp,rq)
        etaprime=self.rho-self.vs*rq
        Vprime=1-(self.Vq/self.t)*(self.K(p-self.q)+self.K(p+self.q))+self.dot(Kp,etaprime)
        identity=abs(Vprime-(d-self.vs*Dqpartial))
        assert identity<2e-11
        assert abs(self.P(p)-target)<2e-11*max(1,target)
        assert d>0
        direct=(p-self.q)*(p+self.q)/2+self.dot(th,self.eta)
        assert abs((direct-float(em))/(self.h*target)-energy/3)<3e-7
        return dict(y=y,k=k,P=target,energy=energy,velocity=velocity,u_remainder=urem,du_remainder=du-np.sinh(k)/k,D=d,V=v,identity_error=identity)

def classical(h,Y,r):
    h,Y,r=map(lambda x:mp.mpf(str(x)),(h,Y,r));b=mp.exp(-mp.mpf('.5'))
    k=mp.sqrt(Y*h/b);l=r*k
    la=2*mp.log(mp.sinh((k-l)/2)/mp.sinh((k+l)/2))
    return (2*b*(mp.cosh(k)-mp.cosh(l))*la+4*b*(l*mp.sinh(k)-k*mp.sinh(l)))/h

def phase(sea,y1,r):
    k1=np.sqrt(sea.h*y1/sea.b);k2=r*k1
    P1=4*sea.b*(k1*np.cosh(k1)-np.sinh(k1))
    P2=4*sea.b*(k2*np.cosh(k2)-np.sinh(k2))
    def momentum(p):return brentq(lambda x:sea.P(x)-p,sea.q,sea.q+max(2.,4*sea.h*y1),xtol=5e-15)
    p1,p2=momentum(P1),momentum(P2)
    assert abs(sea.P(p1)-P1)<2e-11*max(1,P1)
    assert abs(sea.P(p2)-P2)<2e-11*max(1,P2)
    fd=lu_solve(sea.lu,-sea.Th(sea.x-p2)+sea.Th(sea.x-sea.q))
    val=sea.Th(p1-p2)-sea.Th(p1-sea.q)-sea.dot(sea.K(p1-sea.x),fd)
    frame=(-2*k1)*(P2-2*sea.b*np.sinh(k2))
    value=(val-frame)/sea.h
    return dict(Y=y1,r=r,phase=value,residual=float(mp.mpf(str(value))-classical(sea.h,y1,r)))

def run():
    rows=[];start=time.monotonic()
    jobs=[(h,1) for h in [.02,.01,.005,.0025]]+[(h,1.25) for h in [.01,.005,.0025]]
    for h,f in jobs:
        ts=time.monotonic();N=int(np.ceil(f*(10/h+200)));sea=Sea(h,N)
        fixed=[]
        for k in [.12,.18,.25,.35,.5,.7]:
            y=sea.b*k*k/h;p=sea.point(y);p['kappa_fixed']=k
            p['two_body']=phase(sea,y,.5);fixed.append(p)
        edge=[phase(sea,Y,r) for Y in [4.,8.,16.,32.,64.,128.] for r in [.4,.5,.6]]
        row=dict(h=h,N=N,factor=f,q=sea.q,sound=3*(sea.vs-sea.b)/h,count_error=sea.count_error,edge_error=sea.edge_error,fixed=fixed,edge=edge,seconds=time.monotonic()-ts)
        rows.append(row);CACHE.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
        print('sea',h,N,'seconds',round(row['seconds'],1),flush=True)
    return rows
if __name__=='__main__':
    rows=run() if '--refresh' in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text())['rows']
    assert len(rows)==7
    for row in rows:
        assert row['count_error']<2e-11 and row['edge_error']<2e-10
        assert max(p['identity_error'] for p in row['fixed'])<2e-11
    print('ALL CHECKS PASSED')

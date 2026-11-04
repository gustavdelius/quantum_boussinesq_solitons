#!/usr/bin/env python3
"""Fresh finite-sea scan. Only onebody_prec_*.json is written.
Double precision sea solves; 60-digit stable classical subtractions.
Run --recompute to regenerate, otherwise validate cache.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time,json,sys,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import roots_legendre,digamma,loggamma
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root_scalar,brentq
HERE=Path(__file__).resolve().parent; CACHE=HERE/'onebody_prec_scan.json'
mp.mp.dps=60
YS=[.25,.5,1.,2.,4.,6.,8.,12.,16.,24.,32.,48.,64.,96.,128.]
HS=[.08,.06,.04,.03,.02,.015,.01,.0075,.005,.00375,.0025]
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

def run():
    rows=[]
    jobs=[(h,1) for h in HS]+[(h,1.25) for h in [.01,.005,.0025]]
    for h,f in jobs:
        start=time.monotonic();N=int(np.ceil(f*(10/h+200)))
        sea=Sea(h,N)
        row=dict(h=h,N=N,factor=f,q=sea.q,sound=3*(sea.vs-sea.b)/h,count_error=sea.count_error,edge_error=sea.edge_error,points=[sea.point(y) for y in YS],seconds=time.monotonic()-start)
        rows.append(row);CACHE.write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
        print('h',h,'N',N,'seconds',round(row['seconds'],1),'C(32)',row['points'][10]['energy'],flush=True)
    return rows
if __name__=='__main__':
    rows=run() if '--recompute' in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text())['rows']
    assert len(rows)==14
    for r in rows:
        assert r['count_error']<2e-11 and r['edge_error']<2e-10
        assert len(r['points'])==len(YS)
        assert max(p['identity_error'] for p in r['points'])<2e-11
    print('ALL CHECKS PASSED')

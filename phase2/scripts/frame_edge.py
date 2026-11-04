#!/usr/bin/env python3
"""Independent finite-sea dictionary scan; all output restricted to frame_*.json.
Run OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 frame_edge.py.
Default reads its cache; --recompute regenerates it. No WH result is used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre, digamma, loggamma
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root_scalar, brentq
HERE=Path(__file__).resolve().parent

class Sea:
    def __init__(self,h,N):
        self.h=h; self.a=1.; self.b=np.exp(-.5); self.t=2*np.pi*h; self.N=N
        self.z,self.w=roots_legendre(N)
        def count(q,save=False):
            x=q*self.z; w=q*self.w
            A=self.t*np.eye(N)-self.K(x[:,None]-x[None,:])*w[None,:]
            lu=lu_factor(A); rho=lu_solve(lu,np.ones(N))
            if save:self.x,self.wq,self.lu,self.rho=x,w,lu,rho
            return np.dot(w,rho)-1
        sol=root_scalar(count,x0=2*self.b+h,x1=2*self.b+2*h,xtol=1e-13)
        assert sol.converged
        self.q=sol.root; assert abs(count(self.q,True))<2e-11
        self.eta=lu_solve(self.lu,self.x)
        self.Dq=self.D(self.q); self.Vq=self.V(self.q)
        self.rhoq=self.Dq/self.t; self.etaq=self.Vq/self.t
        self.vs=self.Vq/self.Dq
    def K(self,d):return 2*np.real(digamma(1+1j*d/self.h))+2*np.log(self.h)
    def Th(self,d):return 2*self.h*np.imag(loggamma(1+1j*d/self.h))+2*d*np.log(self.h)
    def dot(self,f,g):return np.dot(self.wq*f,g)
    def P(self,p):return p+self.dot(self.Th(p-self.x),self.rho)
    def E(self,p):return p*p/2+self.dot(self.Th(p-self.x),self.eta)
    def D(self,p):return 1+self.dot(self.K(p-self.x),self.rho)
    def V(self,p):return p+self.dot(self.K(p-self.x),self.eta)
    def relP(self,p):return p-self.q+self.dot(self.Th(p-self.x)-self.Th(self.q-self.x),self.rho)
    def relE(self,p):return (p-self.q)*(p+self.q)/2+self.dot(self.Th(p-self.x)-self.Th(self.q-self.x),self.eta)
    def point(self,y):
        k=np.sqrt(y*self.h/self.b)
        target=4*self.b*(k*np.cosh(k)-np.sinh(k))
        p=brentq(lambda p:self.relP(p)-target,self.q,self.q+max(2.,4*self.h*y),xtol=6e-15)
        d=self.D(p); v=self.V(p)
        df=lu_solve(self.lu,self.Th(p-self.x)-self.Th(self.q-self.x))
        ends=np.array([(self.Th(p-e)-self.Th(self.q-e)+self.dot(self.K(e-self.x),df))/self.t for e in (self.q,-self.q)])
        M=self.t*self.rhoq*(ends[0]-ends[1]); Q=(ends[0]+ends[1])/(2*self.rhoq)
        E=self.relE(p); Ecl=2*self.b**2*(np.sinh(k)*np.cosh(k)-k)
        return dict(y=y,k=k,p=p,u=(p-self.q)/self.h,P=target,E=E,Ecl=Ecl,
                    du_dy=2*np.sinh(k)/d,D=d,V=v,M=M,Q=Q,
                    M_identity_error=abs(M+(d-self.Dq)),
                    Q_identity_error=abs(Q-(target-v+self.Vq)/(2*self.t*self.rhoq*self.etaq)),
                    E_scaled=(E-self.vs*target)*np.sqrt(self.b)/self.h**2.5,
                    dispersion_ratio=(E-Ecl)/(self.h*target),ends=ends.tolist(),df=df)
    def four(self,p,z):
        return (self.Th(p['p']-z['p'])-self.Th(p['p']-self.q)
                -self.Th(self.q-z['p'])+self.Th(0)
                -self.dot(self.K(p['p']-self.x)-self.K(self.q-self.x),z['df']))

def run():
    rows=[]
    for h,factor in [(h,1.) for h in (.08,.04,.02,.01,.005)]+[(.01,1.35),(.005,1.25)]:
        start=time.monotonic();N=int(np.ceil(factor*(10/h+200)))
        m=Sea(h,N); ys=sorted(set([.1,.5,1.,2.,4.,8.,16.,32.,64.]+[Y*r*r for Y in (4.,16.,64.) for r in (.4,.5,.6)]))
        pts={y:m.point(y) for y in ys}
        pairs=[]
        for Y in (4.,16.,64.):
            for r in (.4,.5,.6):
                p,z=pts[Y],pts[Y*r*r]
                sym=m.t*(p['ends'][0]*z['ends'][0]-p['ends'][1]*z['ends'][1])
                C=lambda p,z:-2*p['k']*(z['P']-2*m.b*np.sinh(z['k']))
                rem=(sym-C(p,z)-C(z,p))/h
                four=m.four(p,z);rev=m.four(z,p)
                pairs.append(dict(Y=Y,r=r,symmetric_remainder=rem,
                     reciprocity_error=abs((four+rev-sym)/h),
                     algebraic_antisymmetric=(four-p['M']*z['Q'])/h,
                     phase_four=four/h,
                     jacobian_no_endpoint=8*np.sqrt(Y*(Y*r*r))/(p['du_dy']*z['du_dy'])-8*Y*r))
        # A rectangle removes every one-body reference term, including endpoint terms.
        def rem(y,z):
            p,t=pts[y],pts[z]
            C=lambda p,t:-2*p['k']*(t['P']-2*m.b*np.sinh(t['k']))
            return (p['M']*t['Q']+t['M']*p['Q']-C(p,t)-C(t,p))/h
        rect=rem(16.,4.)-rem(8.,4.)-rem(16.,2.)+rem(8.,2.)
        fixed=[]
        for P in (.15,.4,.6):
            k=brentq(lambda k:4*m.b*(k*np.cosh(k)-np.sinh(k))-P,1e-5,3)
            v=m.point(m.b*k*k/h)
            fixed.append(dict(P=P,E1=(v['E']-v['Ecl'])/h))
        for p in pts.values():del p['df']
        row=dict(h=h,N=N,factor=factor,q=m.q,vs=m.vs,
                 sound_correction=(m.vs-m.b)/h,edge_error=abs(m.P(m.q)-np.pi*h),
                 points=list(pts.values()),pairs=pairs,rectangle=rect,fixed=fixed,seconds=time.monotonic()-start)
        assert row['edge_error']<1e-9
        assert max(x['reciprocity_error'] for x in pairs)<1e-8
        assert max(p['M_identity_error'] for p in pts.values())<1e-9
        assert max(p['Q_identity_error'] for p in pts.values())<1e-9
        rows.append(row)
        (HERE/'frame_edge_data.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('h',h,'N',N,'seconds',round(row['seconds'],2),'rectangle',rect,'sound',row['sound_correction'],flush=True)
    return rows

if __name__=='__main__':
    if '--recompute' in sys.argv or not (HERE/'frame_edge_data.json').exists(): rows=run()
    else: rows=json.loads((HERE/'frame_edge_data.json').read_text())
    assert len(rows)==7
    for row in rows:
        assert row['edge_error']<1e-9
        assert max(p['reciprocity_error'] for p in row['pairs'])<1e-8
        assert all(p['du_dy']>0 for p in row['points'])
    print('ALL CHECKS PASSED')

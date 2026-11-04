#!/usr/bin/env python3
"""Independent 60-digit checks of the exact identity used in the edge reduction.
Fresh quadratures and q finite differences; no earlier module is imported.
"""
from pathlib import Path
import time,json,sys
import mpmath as m
m.mp.dps=60;START=time.monotonic();HERE=Path(__file__).resolve().parent
CACHE=HERE/'onebody_prec_identities.json'
if CACHE.exists() and '--recompute' not in sys.argv:
 saved=json.loads(CACHE.read_text())
 assert saved['precision']==60 and len(saved['rows'])==2
 for row,tol in zip(saved['rows'],[m.mpf('1e-14'),m.mpf('1e-22')]):
  assert max(map(m.mpf,row['errors']))<tol
 print('ALL CHECKS PASSED')
 sys.exit(0)
class Sea:
 def __init__(self,N,q):
  self.h=m.mpf('.7');self.t=2*m.pi*self.h;self.q=q
  z,w=m.gauss_quadrature(N,'legendre');self.x=q*z;self.w=q*w
  M=m.matrix([[self.t*(i==j)-self.K(self.x[i]-self.x[j])*self.w[j] for j in range(N)] for i in range(N)])
  self.rho=m.lu_solve(M,m.ones(N,1));self.eta=m.lu_solve(M,self.x)
  self.vs=self.V(q)/self.D(q)
 def K(self,x):return 2*m.re(m.digamma(1+1j*x/self.h))+2*m.log(self.h)
 def dot(self,f,g):return sum(w*f(x)*r for x,w,r in zip(self.x,self.w,g))
 def D(self,p):return 1+self.dot(lambda x:self.K(p-x),self.rho)
 def V(self,p):return p+self.dot(lambda x:self.K(p-x),self.eta)
 def Vprime(self,p):return 1+self.dot(lambda x:-2*m.im(m.polygamma(1,1+1j*(p-x)/self.h))/self.h,self.eta)
rows=[]
for N,tol in [(32,m.mpf('1e-14')),(48,m.mpf('1e-22'))]:
 q=m.mpf('1');dq=m.mpf('1e-15');base=Sea(N,q);plus=Sea(N,q+dq);minus=Sea(N,q-dq)
 errs=[]
 for p in map(m.mpf,['1','1.2','2','4']):
  deriv=(plus.D(p)-minus.D(p))/(2*dq)
  error=abs(base.Vprime(p)-base.D(p)+base.vs*deriv)
  assert error<tol,(N,p,error)
  errs.append(str(error))
 rows.append(dict(N=N,errors=errs))
 print('N',N,'max error',m.nstr(max(map(m.mpf,errs)),5),flush=True)
out=dict(precision=60,rows=rows,seconds=time.monotonic()-START)
(HERE/'onebody_prec_identities.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL CHECKS PASSED')

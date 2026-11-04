#!/usr/bin/env python3
"""Independent corrected half-line inversion and endpoint-resolved convolution.
Writes only uniform_limits_wh.json; --refresh recomputes cached numerics.
No finite-sea or physical-frame uniformity is assumed by these checks.
"""
from pathlib import Path
import json, sys, time
import mpmath as m
m.mp.dps=40
HERE=Path(__file__).resolve().parent
CACHE=HERE/'uniform_limits_wh.json'
A=1/(2*m.pi); c=(1-m.log(8*m.pi))/(2*m.pi)
d=m.mpf(1)/48-1/(4*m.pi**2)
e=(4+6*m.zeta(3)-m.pi**2)/(96*m.pi**3)
def g(s):
    return m.exp(m.loggamma(1+s/(2*m.pi))-s/(2*m.pi)*m.log(s/(2*m.pi*m.e)))/m.sqrt(s)
def inv(u,p,method='talbot'):
    if p==0:
        return m.invertlaplace(lambda s:m.sqrt(m.pi)*(g(s)-1),u,method=method)
    return m.invertlaplace(lambda s:m.sqrt(m.pi)*g(s)/s**p,u,method=method)
def uof(y):
    target=4*y**m.mpf('1.5')/3
    guess=y-A*m.log(y)+c+d/y+e/y**2 if y>2 else target/m.sqrt(m.pi)
    return m.findroot(lambda u:inv(u,2)-target,(guess,guess+m.mpf('.01')))
def classical(y,r):return 2*y*(1-r*r)*m.log((1-r)/(1+r))+4*r*y
def c0(r):return -2*(m.polylog(2,r)-m.polylog(2,-r)-2*r)/m.pi
def c1(r):return -r**3/(6*(1-r*r))-2*d*((1-r*r)/r**2*m.log((1-r)/(1+r))+2/r)
def convolution(u,v,n):
    # Explicit panels retain the z=O(1) endpoint at every large v.
    edges=[m.mpf(0),min(m.mpf(1),v)]
    while edges[-1]<v:edges.append(min(4*edges[-1],v))
    nodes,weights=m.gauss_quadrature(n,'legendre')
    val=m.mpf(0)
    for left,right in zip(edges[:-1],edges[1:]):
        mid=(left+right)/2; half=(right-left)/2
        for x,w in zip(nodes,weights):
            z=mid+half*x
            val+=2*half*w*inv(u-v+z,0)*inv(z,1)
    return val

def compute():
    rows=[];start=time.monotonic()
    for r0 in ['.4','.5','.6']:
        r=m.mpf(r0)
        for Y in [16,32,64,128,256,512]:
            ts=time.monotonic();y=m.mpf(Y);u=uof(y);v=uof(r*r*y)
            lo=convolution(u,v,12);hi=convolution(u,v,20)
            err=abs(hi-lo)
            assert err<m.mpf('2e-12'),(r,Y,err)
            R=hi-classical(y,r)
            # Independent inversion method at the two arguments and midpoint.
            cross=max(abs(inv(z,p)-inv(z,p,'dehoog')) for z,p in [(u,2),(v,2),(u-v/2,0),(v/2,1)])
            assert cross<m.mpf('1e-30'),cross
            row=dict(r=str(r),Y=Y,u=str(u),v=str(v),phase=str(hi),residual=str(R),quad_error=str(err),inverse_cross_error=str(cross),c1_effective=str(y*(R-c0(r))),c1_formal=str(c1(r)),seconds=time.monotonic()-ts)
            rows.append(row)
            CACHE.write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
            print('WH',r,Y,'Y(R-c0)',m.nstr(y*(R-c0(r)),15),'quad',m.nstr(err,3),'seconds',round(row['seconds'],1),flush=True)
    return rows
if __name__=='__main__':
    rows=compute() if '--refresh' in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text())['rows']
    assert len(rows)==18
    for row in rows:
        assert m.mpf(row['quad_error'])<m.mpf('2e-12')
        assert m.mpf(row['inverse_cross_error'])<m.mpf('1e-30')
    print('ALL CHECKS PASSED')

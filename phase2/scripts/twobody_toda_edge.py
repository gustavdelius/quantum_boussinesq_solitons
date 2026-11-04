#!/usr/bin/env python3
"""Corrected WH convolution at fixed dressed momentum, independent inversions.
All numbers are dimensionless Toda action/h, equal to Boussinesq action/hbar_B.
Conditional edge model, not a claim of uniform finite-sea asymptotics.
Run --refresh to rebuild only this script's own cache.
"""
from pathlib import Path
import json,sys,time
from functools import lru_cache
import mpmath as m
m.mp.dps=32
HERE=Path(__file__).resolve().parent
CACHE=HERE/'twobody_toda_edge.json'

def factor(s):
    x=s/(2*m.pi)
    return m.exp(m.loggamma(1+x)-x*(m.log(x)-1))/m.sqrt(s)
def il(u,o=0,method='talbot'):
    return cached_il(u,o,method,m.mp.dps)
@lru_cache(maxsize=30000)
def cached_il(u,o,method,dps):
    if u==0:
        return [m.mpf(0),m.sqrt(m.pi),m.pi**m.mpf('1.5')/6][o]
    return m.invertlaplace(lambda s:m.sqrt(m.pi)*(factor(s)-(1 if o==2 else 0))*s**(o-2),u,method=method)
@lru_cache(None)
def point(ytext,dps=32,method='talbot'):
    with m.workdps(dps):
        y=m.mpf(ytext);target=4*y**m.mpf('1.5')/3
        guess=y-m.log(y)/(2*m.pi)+(1-m.log(8*m.pi))/(2*m.pi)
        u=m.findroot(lambda u:il(u,0,method)-target,(max(y/10,guess),y),tol=m.mpf(10)**(-dps+5))
        B=il(u,1,method);J=m.invertlaplace(lambda s:m.sqrt(m.pi)*factor(s)/s**3,u,method=method)
        return dict(y=str(y),u=str(u),B=str(B),du=str(2*m.sqrt(y)/B),C_E=str(3/(4*m.pi)+3*J/(4*target)-3*y/10),C_v=str(3/(4*m.pi)+y*(2*m.sqrt(y)/B-1)/2))
@lru_cache(None)
def rule(n,dps):
    with m.workdps(dps):return m.gauss_quadrature(n,'legendre')
def integrate(u,v,n,method='talbot'):
    # u>v: write t=v-w, geometrically subdividing the small-t endpoint layer.
    xs,ws=rule(n,m.mp.dps)
    cuts=[m.mpf(0)];a=m.mpf('.25')
    while a<v:cuts.append(a);a*=4
    cuts.append(v);val=m.mpf(0)
    for a,b in zip(cuts,cuts[1:]):
        for x,w in zip(xs,ws):
            t=a+(b-a)*(x+1)/2
            val+=(b-a)*w*il(u-v+t,2,method)*il(t,1,method)
    return val
def chi(r):return (m.polylog(2,r)-m.polylog(2,-r))/2
def candidate(r):return -4/m.pi*(chi(r)-r)
def anti_candidate(r):return m.pi-4/m.pi*chi(r)+2/m.pi*(r-1/r)
def calc(Ytext,rtext):
    Y=m.mpf(Ytext);r=m.mpf(rtext);z=Y*r*r
    p=point(str(Y));q=point(str(z));u=m.mpf(p['u']);v=m.mpf(q['u'])
    B1=m.mpf(p['B']);B2=m.mpf(q['B'])
    vals=[integrate(u,v,n) for n in [16,24]]
    err=abs(vals[1]-vals[0]);assert err<m.mpf('1e-11'),(Y,r,err)
    H=vals[1];cl=4*m.sqrt(Y*z)+2*(Y-z)*m.log((1-r)/(1+r))
    R12=H-cl;sy=B1*B2-m.pi-4*m.sqrt(Y*z);an=R12-sy
    U1=2*m.sqrt(m.pi)*(B1-m.sqrt(m.pi));U2=2*m.sqrt(m.pi)*(B2-m.sqrt(m.pi))
    four=H-U2;endpoint_B=four-(B1-m.sqrt(m.pi))*(B2-m.sqrt(m.pi))
    return {k:str(v) for k,v in dict(y1=Y,y2=z,r=r,u1=u,u2=v,B1=B1,B2=B2,
            N12=H,N21=2*B1*B2-2*m.pi-H,U12=U2,U21=U1,
            R12=R12,R21=2*sy-R12,R_symmetric=sy,R_antisymmetric=an,
            four12=four,four_antisymmetric=endpoint_B,
            reference_antisymmetric=(U1-U2)/2,
            forward_limit=candidate(r),symmetric_limit=2/m.pi*(r+1/r)-m.pi,
            antisymmetric_limit=anti_candidate(r),quadrature_difference=err,
            C_E_1=m.mpf(p['C_E']),C_v_1=m.mpf(p['C_v'])).items()}

def main():
    rows=[] if '--refresh' in sys.argv or not CACHE.exists() else json.loads(CACHE.read_text())['rows']
    jobs=[(Y,r) for Y in [4,8,16,32,64,128,256,512,1024] for r in ['.4','.5','.6']]
    jobs +=[(Y,r) for Y in [32,128,512] for r in ['.25','.75','.9']]
    for Y,r in jobs:
        if any(m.mpf(x['y1'])==Y and m.mpf(x['r'])==m.mpf(r) for x in rows):continue
        start=time.monotonic();row=calc(str(Y),r);row['seconds']=time.monotonic()-start;rows.append(row)
        CACHE.write_text(json.dumps(dict(rows=rows,status='conditional corrected half-line model; finite-y residuals are not pure one-loop coefficients'),indent=2)+'\n')
        print('Y',Y,'r',r,'forward',row['R12'][:20],'anti',row['R_antisymmetric'][:20],'quad',row['quadrature_difference'][:10],'seconds',round(row['seconds'],1),flush=True)
    # Different inversion algorithm and higher precision, independent of convolution fits.
    cross=[]
    for y in ['4','16','64','256']:
        a=point(y);b=point(y,45,'dehoog')
        err=max(abs(m.mpf(a[k])-m.mpf(b[k])) for k in ['u','B','du','C_E','C_v'])
        assert err<m.mpf('1e-25'),(y,err)
        cross.append(dict(y=y,error=str(err)))
    # Check the derivative used in the convolution by an independent derivative of B.
    with m.workdps(45):
        deriv=[]
        for u in ['.2','2','20']:
            uu=m.mpf(u);eps=m.mpf('1e-12')
            numerical=(il(uu+eps,1,'dehoog')-il(uu-eps,1,'dehoog'))/(2*eps)
            err=abs(numerical-il(uu,2,'talbot'))
            assert err<m.mpf('1e-23');deriv.append(str(err))
    assert len(rows)==len(jobs)
    assert max(m.mpf(r['quadrature_difference']) for r in rows)<m.mpf('1e-11')
    # Independently differentiate the actual convolution along a characteristic.
    u=m.mpf(4);v=m.mpf(1);eps=m.mpf('1e-4')
    def four(delta):
        return integrate(u+delta,v+delta,24)-2*m.sqrt(m.pi)*(il(v+delta,1)-m.sqrt(m.pi))
    difference=(8*(four(eps)-four(-eps))-(four(2*eps)-four(-2*eps)))/(12*eps)
    source=2*il(u,2)*il(v,1)-2*m.sqrt(m.pi)*il(v,2)
    printed=2*il(u,2)*(il(v,1)-m.sqrt(m.pi))
    assert abs(difference-source)<m.mpf('1e-13')
    assert abs(difference-printed)>m.mpf('.1')
    pde=dict(u=str(u),v=str(v),derivative=str(difference),correct_source=str(source),printed_source=str(printed),correct_error=str(abs(difference-source)),printed_error=str(abs(difference-printed)))
    CACHE.write_text(json.dumps(dict(rows=rows,cross_method=cross,derivative_errors=deriv,pde_check=pde,
        status='conditional corrected half-line model; finite-y residuals are not pure one-loop coefficients'),indent=2)+'\n')
    print('ALL CHECKS PASSED')

if __name__=='__main__':main()

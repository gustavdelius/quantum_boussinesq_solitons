"""Connected u^2 insertion versus canonical endpoint response.

Uses the bilinear equation and independent spatial integration of temporal
log-tau derivatives at large positive/negative times, not a phase fit.
"""
import sys
sys.dont_write_bytecode=True
import sympy as sp
import mpmath as mp
from good_twobody_kernels import sol,classical,save

def derivative_logtau(s,t,x,time,logA):
    z=s['k']*(x-s['v']*time)
    w=t['k']*(x-t['v']*time)
    logs=[mp.mpf(0),z,w,z+w+logA]
    shift=max(logs)
    weights=[mp.exp(y-shift) for y in logs]
    logtau=shift+mp.log(sum(weights))
    softplus=lambda z:max(z,mp.mpf(0))+mp.log1p(mp.exp(-abs(z)))
    # Analytically subtract the vacuum plateau BEFORE infinite-interval quadrature.
    return -mp.expm1(logA)*(s['k']*s['v']*mp.exp(z+w-logtau-softplus(z))+
                            t['k']*t['v']*mp.exp(z+w-logtau-softplus(w)))

if __name__=='__main__':
    mp.mp.dps=65
    x,time=sp.symbols('x t')
    f=sp.Function('f')(x,time);tau=sp.exp(f)
    bilinear=2*(tau*sp.diff(tau,time,2)-sp.diff(tau,time)**2-
                tau*sp.diff(tau,x,2)+sp.diff(tau,x)**2+
                tau*sp.diff(tau,x,4)-4*sp.diff(tau,x)*sp.diff(tau,x,3)+3*sp.diff(tau,x,2)**2)
    assert sp.simplify(bilinear/tau**2-2*(sp.diff(f,time,2)-sp.diff(f,x,2)+
                                         sp.diff(f,x,4)+6*sp.diff(f,x,2)**2))==0
    E,F,A,om1,om2=sp.symbols('E F A om1 om2',positive=True)
    original=(-om1*E-om2*F-(om1+om2)*A*E*F)/(1+E+F+A*E*F)+om1*E/(1+E)+om2*F/(1+F)
    stable=-(A-1)*E*F/(1+E+F+A*E*F)*(om1/(1+E)+om2/(1+F))
    assert sp.factor(original-stable)==0
    results=[]
    for e2 in (1,-1):
        s,t=sol('.3',1),sol('.6',e2)
        c=classical(s,t);logA=c['logA']
        slopes=[]
        for tt in (mp.mpf(-3000),mp.mpf(3000)):
            centres=[z['v']*tt+shift/z['k'] for z in (s,t) for shift in (0,-logA)]
            points=sorted(set(y+z for y in centres for z in (mp.mpf(-50),mp.mpf(0),mp.mpf(50))))
            val=mp.quad(lambda xx:derivative_logtau(s,t,xx,tt,logA),[-mp.inf]+points+[mp.inf])
            target=-logA*(t['v'] if tt<0 else s['v'])
            assert abs(val-target)<mp.mpf('1e-35'),(e2,tt,val-target)
            slopes.append(val)
        insertion=-2*(slopes[1]-slopes[0])/3
        target=2*(s['v']-t['v'])*logA/3
        assert abs(insertion-target)<mp.mpf('1e-35')
        results.append(dict(epsilon=[1,e2],connected_u2_spacetime_integral=insertion,
                            target=target,error=abs(insertion-target),
                            canonical_endpoint_response=c['lambda_coefficient']+insertion))
    save('good_twobody_local_insertion.json',dict(precision=65,time_endpoints=[-3000,3000],
          samples=results,scope='connected local insertion only; finite canonical endpoint term also needed'))
    print('ALL CHECKS PASSED')

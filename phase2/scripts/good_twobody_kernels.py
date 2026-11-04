"""Independent continuum integrals; no imports or writes of legacy packages.

These functions are conditional ingredients, NOT a completed determinant.
Run with PYTHONDONTWRITEBYTECODE=1. All cache writes use this task's prefix.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import mpmath as mp

WEIGHTS = (1, 1, -2)

def roots(a):
    return (a + 2*mp.pi/3, a - 2*mp.pi/3, a)

def sol(a, e):
    a = mp.mpf(a)
    k, v = mp.sin(a), e*mp.cos(a)
    return dict(a=a, e=e, k=k, v=v, M=2*k, Pi=-2*k*v,
                P=2*v*k**3/3, E=2*k**3/3-8*k**5/15,
                ell=-e*mp.cos(3*a), pa=(k+v/mp.sqrt(3))/2,
                pb=(v/mp.sqrt(3)-k)/2,
                gamma=-3*mp.cos(2*a)/(mp.sin(a)*mp.sin(3*a)))

def interaction(s, t):
    return ((s['pa']-t['pa'])*(s['pb']-t['pb']) /
            ((s['pa']-t['pb'])*(s['pb']-t['pa'])))

def classical(s, t):
    A = interaction(s, t)
    assert A > 0
    logA = mp.log(A)
    top = 2*s['M']*t['M']*(s['v']-t['v'])/3
    d0 = 2*(s['ell']-t['ell'])*logA/9 + top
    x1 = -logA/s['k'] + s['gamma']*t['M']
    x2 = logA/t['k'] - t['gamma']*s['M']
    D = s['P']*x1 + t['P']*x2
    return dict(A=A, logA=logA, delta0=d0, top=top, X1=x1, X2=x2, D=D,
                lambda_coefficient=3*d0+4*D,
                eta_coefficient=2*d0+2*D,
                zeta_coefficient=2*d0+D,
                kappa_coefficient=-d0-D)

def theta(q, a, e):
    z = abs(q)
    return -sum(c*(mp.sign(q)*mp.atan(mp.sin(f)/z) +
                  e*mp.atan(mp.sin(f)*mp.cos(f)/(z*mp.sqrt(1+z*z)+z*z+mp.sin(f)**2)))
                for c, f in zip(WEIGHTS, roots(a)))

def dtheta(q, a, e):
    return sum(c*mp.sin(f)/(q*q+mp.sin(f)**2)*
               (1+e*mp.sign(q)*mp.cos(f)/mp.sqrt(1+q*q))
               for c, f in zip(WEIGHTS, roots(a)))

def L(z):
    if z == 0:
        return mp.mpf(0)
    w = abs(z)
    if w == 1:
        return -mp.sign(z)*mp.pi**2/4
    # Invert large arguments to avoid complex polylog branch bookkeeping.
    if w > 1:
        return -mp.sign(z)*mp.pi**2/2 - L(1/z)
    return -mp.sign(z)*(mp.polylog(2,w)-mp.polylog(2,-w) +
                        mp.log(w)*mp.log((1-w)/(1+w)))

def integrals(s, t):
    a, b, e, f = s['a'], t['a'], s['e'], t['e']
    same = mp.mpf(0)
    head = mp.mpf(0)
    for c, r in zip(WEIGHTS, roots(a)):
        for d, z in zip(WEIGHTS, roots(b)):
            x, y = mp.tan(r/2), mp.tan(z/2)
            same += c*d*L(x/y)/(2*mp.pi)
            head += c*d*L(x*y)/(2*mp.pi)
    raw = (e+f)*same/2 + (e-f)*head/2
    symmetric = -mp.pi*(e+f)/4
    return raw, raw-symmetric, symmetric

def momentum_mode(s):
    return -s['e']*sum(c*mp.sin(f)*mp.log(abs(mp.tan(f/2)))
                       for c, f in zip(WEIGHTS,roots(s['a'])))/(2*mp.pi)

def energy_mode(s):
    return (-6*s['k']-2*sum(c*mp.sin(f)*mp.cos(f)*mp.log(abs(mp.tan(f/2)))
                            for c,f in zip(WEIGHTS,roots(s['a']))))/(4*mp.pi)

def lagrangian_trace(a,e):
    s=sol(a,e)
    return s['v']*momentum_mode(s)-energy_mode(s)

def momentum_canonical(s):
    return mp.diff(lambda a: lagrangian_trace(a,s['e']),s['a'])/(-s['e']*s['k'])

def quadrature(s,t):
    widths=sorted(set([mp.mpf('0.03'),mp.mpf('0.1'),mp.mpf(1),mp.mpf(5)] +
                      [abs(mp.sin(f)) for a in (s['a'],t['a']) for f in roots(a)]))
    fun=lambda q: (theta(q,s['a'],s['e'])*dtheta(q,t['a'],t['e'])+
                   theta(-q,s['a'],s['e'])*dtheta(-q,t['a'],t['e']))/(4*mp.pi)
    return mp.quad(fun,[0]+widths+[mp.inf])

def serialize(x):
    if isinstance(x,dict): return {k:serialize(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serialize(v) for v in x]
    if isinstance(x,(mp.mpf,mp.mpc)): return mp.nstr(x,55)
    return x

def save(name,data):
    assert name.startswith('good_twobody_') and name.endswith('.json')
    Path(__file__).with_name(name).write_text(json.dumps(serialize(data),indent=2)+'\n')

if __name__ == '__main__':
    mp.mp.dps=65
    worst=mp.mpf(0)
    for e,f in ((1,1),(1,-1),(-1,1),(-1,-1)):
        s,t=sol('0.3',e),sol('0.6',f)
        i,j,sym=integrals(s,t)
        iq=quadrature(s,t)
        err=abs(i-iq)
        worst=max(worst,err)
        assert err<mp.mpf('1e-55'),(e,f,err)
        irev,jrev,_=integrals(t,s)
        assert abs(i+irev-2*sym)<mp.mpf('1e-60')
        assert abs(j+jrev)<mp.mpf('1e-60')
        for q in (mp.mpf('-.7'),mp.mpf('.23')):
            assert abs(mp.diff(lambda z:theta(z,s['a'],e),q)-dtheta(q,s['a'],e))<mp.mpf('1e-60')
            ph=sol(s['a'],e)
            pa=mp.sign(q)*mp.sqrt(1+q*q)/(2*mp.sqrt(3))+1j*q/2
            pb=mp.conj(pa)
            trans=(ph['pa']-pa)*(ph['pb']-pb)/((ph['pa']-pb)*(ph['pb']-pa))
            assert abs(trans-mp.exp(1j*theta(q,s['a'],e)))<mp.mpf('1e-60')
        pquad=mp.quad(lambda q:q*(dtheta(q,s['a'],e)-dtheta(-q,s['a'],e))/(4*mp.pi),
                      [0,mp.mpf('.1'),1,5,mp.inf])
        assert abs(pquad-momentum_mode(s))<mp.mpf('1e-55')
        equad=-3*s['k']/(2*mp.pi)+mp.quad(
            lambda q:sum(c*mp.sin(z)*mp.cos(z)**2*q/
                         (mp.sqrt(1+q*q)*(q*q+mp.sin(z)**2))
                         for c,z in zip(WEIGHTS,roots(s['a'])))/(2*mp.pi),
            [0,mp.mpf('.1'),1,5,mp.inf])
        assert abs(equad-energy_mode(s))<mp.mpf('1e-55')
        for sign in (-1,1):
            assert abs(theta(sign*mp.mpf('1e-50'),s['a'],e)-mp.pi*(sign+e))<mp.mpf('1e-45')
    for z in (mp.mpf('-.7'),mp.mpf('.23'),mp.mpf('2.3')):
        assert abs(mp.diff(L,z)-2*mp.log(abs(z))/(1-z*z))<mp.mpf('1e-60')
    save('good_twobody_kernels.json',dict(precision=65,quadrature_max_error=worst,
          status='continuum integrals only; physical determinant unestablished'))
    print('max independent quadrature error',mp.nstr(worst,8))
    print('ALL CHECKS PASSED')

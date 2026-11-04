#!/usr/bin/env python3
"""Exact canonical higher charge and its unrestricted elastic deformation.

All identities concern local continuum functionals, modulo total derivatives,
on a periodic line or with decaying variations and fixed potential endpoints.
"""
import sympy as sp
import mpmath as mp

u = sp.symbols('u0:12')
p = sp.symbols('p0:12')
s = sp.Symbol('s', real=True)

def dx(f):
    return sp.expand(sum(sp.diff(f,u[i])*u[i+1]+sp.diff(f,p[i])*p[i+1]
                         for i in range(11)))

def dn(f,n):
    for _ in range(n):
        f = dx(f)
    return f

def euler(f,jets):
    return sp.expand(sum((-1)**i*dn(sp.diff(f,jets[i]),i) for i in range(8)))

def bracket(f,g):
    return sp.expand(euler(f,u)*dx(euler(g,p))+euler(f,p)*dx(euler(g,u)))

def zero_functional(f):
    assert euler(f,u)==0
    assert euler(f,p)==0
    assert f.subs({v:0 for v in u+p})==0

h = (p[0]**2+s*u[0]**2+u[1]**2)/2-u[0]**3
q = p[1]**2+u[2]**2-3*u[0]*p[0]**2+3*u[0]**4-9*u[0]*u[1]**2+s*(u[1]**2-u[0]**3)
j = u[1]**2-u[0]**3

def checks():
    assert euler(h,u)==s*u[0]-3*u[0]**2-u[2]
    assert euler(h,p)==p[0]
    qu = -3*p[0]**2+12*u[0]**3+9*u[1]**2+18*u[0]*u[2]+2*u[4]-s*(3*u[0]**2+2*u[2])
    qp = -6*u[0]*p[0]-2*p[2]
    assert sp.expand(euler(q,u)-qu)==0 and sp.expand(euler(q,p)-qp)==0
    zero_functional(bracket(q,h))
    for lower in (u[0],p[0],-u[0]*p[0]):
        zero_functional(bracket(lower,h))
    # Verify the canonical variational derivative delta/dphi=-D delta/du.
    canonical_rate=-dx(qu)*p[0]+qp*dx(euler(h,u))
    assert sp.expand(bracket(q,h)-canonical_rate-dx(qu*p[0]))==0
    la,eps,U=sp.symbols('lambda eps U')
    assert sp.expand(h.subs(s,1+2*eps*la)-h.subs(s,1)-eps*la*u[0]**2)==0
    assert sp.expand(q.subs(s,1+2*eps*la)-q.subs(s,1)-2*eps*la*j)==0
    zero_functional(bracket(q,la*u[0]**2)+bracket(2*la*j,h))
    zero_functional(bracket(j,u[0]**2))
    # The uncured I2 term is not conserved; this is a retained negative check.
    assert euler(bracket(u[0]**2,h),p)==-2*u[1]
    assert euler(bracket(u[0]**2,h),u)==2*p[1]
    shifted_h=h.subs(s,1).subs(u[0],u[0]+U)
    assert sp.expand(shifted_h-h.subs(s,1-6*U)-(U-3*U**2)*u[0]-(U**2/2-U**3))==0
    shifted_q=q.subs(s,1).subs(u[0],u[0]+U)
    assert sp.expand(shifted_q-q.subs(s,1-6*U)+6*U*h.subs(s,1-6*U)
                     -(12*U**3-3*U**2)*u[0]-(3*U**4-U**3))==0
    # Check both Lax equations, rather than only the scalar PDE.
    laxq=sp.Rational(3,2)*u[0]-s/4
    laxr=sp.Rational(3,4)*u[1]+sp.sqrt(3)*p[0]/4
    ut=p[1]; pt=dx(euler(h,u))
    assert sp.expand(sp.Rational(3,2)*ut/sp.sqrt(3)-(2*dx(laxr)-dn(laxq,2)))==0
    assert sp.expand((sp.Rational(3,4)*dx(ut)+sp.sqrt(3)*pt/4)/sp.sqrt(3)
                     -(dn(laxr,2)-sp.Rational(2,3)*dn(laxq,3)-sp.Rational(2,3)*laxq*dx(laxq)))==0
    # Independent Riccati construction. Integrated H_n=-n int f_n.
    # f''+3 f f'+f^3+laxq*f+laxr=z^3, f=z+sum f_n z^-n.
    fs={-1:sp.Integer(1)}
    for n in range(1,8):
        power=2-n
        out=sum(3*a*dx(b) for i,a in fs.items() for k,b in fs.items() if -i-k==power)
        out+=sum(a*b*c for i,a in fs.items() for k,b in fs.items()
                 for l,c in fs.items() if -i-k-l==power)
        out+=sum(dn(a,2)+laxq*a for i,a in fs.items() if -i==power)
        if power==0: out+=laxr
        fs[n]=sp.expand(-out/3)
    def vacuum_subtract(f):
        return sp.expand(f-f.subs({v:0 for v in u+p}))
    # Direct pseudo-differential multiplication independently verifies Riccati's
    # integrated residue dictionary. D^i a=sum binomial(i,m) a^(m) D^(i-m).
    def multiply(left,right,minimum):
        out={}
        for i,a in left.items():
            for k,b in right.items():
                for m in range(max(0,i+k-minimum+1)):
                    coefficient=sp.binomial(i,m)
                    if coefficient==0: continue
                    power=i+k-m
                    out[power]=out.get(power,0)+coefficient*a*dn(b,m)
        return {power:sp.expand(value) for power,value in out.items() if value!=0}
    root={1:sp.Integer(1)}; target={3:sp.Integer(1),1:laxq,0:laxr}
    for n in range(1,8):
        unknown=sp.Symbol('root_coefficient')
        trial=dict(root); trial[-n]=unknown
        cube=multiply(multiply(trial,trial,1-n),trial,2-n)
        root[-n]=sp.expand(-(cube.get(2-n,0)-target.get(2-n,0)).subs(unknown,0)/3)
    power={0:sp.Integer(1)}
    for n in range(1,8):
        power=multiply(power,root,-1-(7-n))
        zero_functional(vacuum_subtract(power.get(-1,0)+n*fs[n]))
    H1=vacuum_subtract(-fs[1]); H5=vacuum_subtract(-5*fs[5]); H7=vacuum_subtract(-7*fs[7])
    zero_functional(H1-u[0]/2)
    zero_functional(vacuum_subtract(-2*fs[2])-sp.sqrt(3)*p[0]/6)
    zero_functional(vacuum_subtract(-4*fs[4])-sp.sqrt(3)*u[0]*p[0]/6+sp.sqrt(3)*s*p[0]/36)
    zero_functional(H5-sp.Rational(5,24)*h+sp.Rational(5,288)*s**2*u[0])
    zero_functional(H7+sp.Rational(7,144)*(q+s*h)-sp.Rational(7,2592)*s**3*u[0])
    for n in (3,6): zero_functional(vacuum_subtract(-n*fs[n]))
    # Independent exact integration and 60-digit spatial quadratures.
    k,v,y=sp.symbols('k v y', real=True)
    pulse=k**2*(1-y**2)/2
    def pd(f): return sp.expand(k*(1-y**2)*sp.diff(f,y)/2)
    sub={u[0]:pulse,p[0]:-v*pulse}
    for i in range(1,5): sub[u[i]]=pd(sub[u[i-1]]); sub[p[i]]=pd(sub[p[i-1]])
    assert sp.expand((v**2*pd(pd(pulse))-s*pd(pd(pulse))+pd(pd(pd(pd(pulse))))+3*pd(pd(pulse**2))).subs(v**2,s-k**2))==0
    measure=sp.cancel(2*q.subs(sub)/(k*(1-y**2)))
    integral=sp.expand(sp.integrate(measure,(y,-1,1))).subs(v**2,s-k**2)
    expected=4*k**5*(20*k**2-21*s)/105
    assert sp.expand(integral-expected)==0
    mp.mp.dps=60
    f=sp.lambdify((y,k,v,s),measure,'mpmath')
    worst=mp.mpf(0)
    for ss,kk in [('1','.31'),('1.4','.72'),('.7','.43')]:
        ss,kk=mp.mpf(ss),mp.mpf(kk)
        ans=mp.quad(lambda yy:f(yy,kk,mp.sqrt(ss-kk**2),ss),[-1,0,1])
        err=abs(ans-4*kk**5*(20*kk**2-21*ss)/105)
        worst=max(worst,err); assert err<mp.mpf('1e-55')
    print('Exact Euler, Lax, Riccati, background-shift and arbitrary-lambda identities passed.')
    print('60-digit soliton quadrature maximum error:',mp.nstr(worst,8))
    print('Retained negative check: int u^2 alone is not conserved.')

if __name__=='__main__':
    checks()
    print('ALL CHECKS PASSED')

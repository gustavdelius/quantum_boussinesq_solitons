#!/usr/bin/env python3
"""Independent action/charge audit; no existing project formulas are imported.

Checks algebra and conditional mode-sum consequences, not quantum defect
renormalizability or completeness of the fluctuation determinant.
"""
import mpmath as mp
import sympy as s

mp.mp.dps = 65
a, v, u, ux, et, ex, exx, z = s.symbols('a v u ux et ex exx z', real=True)
# Direct Taylor expansion of the fixed project Lagrangian in moving coordinates.
lag = ( -v*u + z*(et-v*ex))**2/2 - (u+z*ex)**2/2 + (u+z*ex)**3 - (ux+z*exx)**2/2
lag2 = s.expand(lag).coeff(z, 2)
assert s.expand(lag2-((et-v*ex)**2/2-(1-6*u)*ex**2/2-exx**2/2)) == 0
qq,Om=s.symbols('qq Om', real=True)
vacuum_pencil=-(Om+v*qq)**2+qq**2+qq**4
assert s.simplify(vacuum_pencil.subs(Om,s.Abs(qq)*s.sqrt(1+qq**2)-v*qq))==0
phis = [a+2*s.pi/3, a-2*s.pi/3, a]
weights = [1, 1, -2]
assert s.trigsimp(sum(c*s.sin(p) for p,c in zip(phis,weights))+3*s.sin(a)) == 0
assert s.trigsimp(sum(c*s.cos(p) for p,c in zip(phis,weights))+3*s.cos(a)) == 0
print('PASS: quadratic action; weighted sine/cosine sums (printed zero cosine claim fails).')
x,k=s.symbols('x k', positive=True)
profile=k**2*s.sech(k*x/2)**2/2
assert s.simplify(s.expand_trig(-k**2*profile+s.diff(profile,x,2)+3*profile**2)) == 0
bad=-k**2*s.sech(k*x/2)**2
assert s.simplify((-k**2*bad+s.diff(bad,x,2)+3*bad**2).subs(x,0)) == 9*k**4/2

def roots(alpha):
    return [(alpha+2*mp.pi/3, 1), (alpha-2*mp.pi/3, 1), (alpha,-2)]

def theta(q, data):
    aq=abs(q)
    return sum(c*(-mp.sign(q)*mp.atan(mp.sin(p)/aq)-mp.atan(mp.sin(p)*mp.cos(p)/(aq*mp.sqrt(1+q*q)+q*q+mp.sin(p)**2))) for p,c in data)

def pm(alpha):
    return -sum(c*mp.sin(p)*mp.log(abs(mp.tan(p/2))) for p,c in roots(alpha))/(2*mp.pi)

def em(alpha):
    return sum(c*mp.sin(p)*(1-mp.cos(p)*mp.log(abs(mp.tan(p/2)))) for p,c in roots(alpha))/(2*mp.pi)

def ll(alpha):
    return mp.cos(alpha)*pm(alpha)-em(alpha)

def pc(alpha):
    return -mp.diff(ll,alpha)/mp.sin(alpha)

def ec(alpha):
    return mp.cos(alpha)*pc(alpha)-ll(alpha)

def pcl(alpha):
    return mp.mpf(2)/3*mp.sin(alpha)**3*mp.cos(alpha)

def ecl(alpha):
    return mp.mpf(2)/15*mp.sin(alpha)**3*(1+4*mp.cos(alpha)**2)

for alpha in map(mp.mpf, ['0.15','0.5','0.85']):
    data=roots(alpha)
    cutoff=mp.mpf('1e20'); T=mp.sqrt(1+cutoff**2)
    # Exact raw phase-density energy integrated from -cutoff to cutoff.
    raw=sum(c*mp.sin(p)*(T-1+mp.cos(p)/2*(mp.log((T-mp.cos(p))/(T+mp.cos(p)))-mp.log((1-mp.cos(p))/(1+mp.cos(p))))) for p,c in data)/(2*mp.pi)
    endpoint=cutoff*T*(theta(cutoff,data)-theta(-cutoff,data))/(4*mp.pi)
    pendpoint=cutoff*(theta(cutoff,data)+theta(-cutoff,data))/(4*mp.pi)
    assert abs(pendpoint)<mp.mpf('1e-19')
    counter=3*mp.sin(alpha)*(T-1)/mp.pi  # Existing normal ordering, unchanged.
    assert abs(raw-endpoint+counter-em(alpha)) < mp.mpf('1e-19')
    assert abs(endpoint/cutoff-3*mp.sin(alpha)/(2*mp.pi)) < mp.mpf('1e-35')
    pclosed=-mp.sqrt(3)*mp.cos(alpha)/mp.pi*mp.log(abs(mp.tan((alpha+2*mp.pi/3)/2)/mp.tan((alpha-2*mp.pi/3)/2)))
    assert abs(pc(alpha)-pclosed) < mp.mpf('1e-60')
    printed_weight=sum(c*(mp.cos(2*p)-mp.cos(alpha+p))*mp.log(abs(mp.tan(p/2))) for p,c in data)/(2*mp.pi)
    assert abs(mp.diff(ll,alpha)-printed_weight)<mp.mpf('1e-60')
    assert abs(mp.diff(ll,alpha))>mp.mpf('1e-4')
    assert abs(mp.diff(ec,alpha)-mp.cos(alpha)*mp.diff(pc,alpha)) < mp.mpf('1e-59')
    assert abs((ec(alpha)-em(alpha))-mp.cos(alpha)*(pc(alpha)-pm(alpha))) < mp.mpf('1e-60')
    assert abs(mp.diff(ecl,alpha)-mp.cos(alpha)*mp.diff(pcl,alpha)) < mp.mpf('1e-60')
    shift=(pm(alpha)-pc(alpha))/mp.diff(pcl,alpha)
    assert abs(pc(alpha)+mp.diff(pcl,alpha)*shift-pm(alpha)) < mp.mpf('1e-60')
    assert abs(ec(alpha)+mp.diff(ecl,alpha)*shift-em(alpha)) < mp.mpf('1e-60')
    print('PASS: alpha',alpha,'Pcan-Pmode=',mp.nstr(pc(alpha)-pm(alpha),12), 'conditional absorption shift=',mp.nstr(shift,12))

# Finite-dimensional form of the stationary-saddle derivative identity.
H, B, C, D = s.symbols('H B C D', nonzero=True)
y0prime=-B/H; y1=-C/H
assert s.simplify((D+B*y1)-(D+C*y0prime)) == 0
print('PASS: stationary-saddle chain rule, including first-order profile correction.')

# Independently expand the CZ boundary action and conserved momentum.
w, sig0, ax, bx, da, db = s.symbols('w sig0 ax bx da db', real=True)
al=s.sqrt(3)/3; be=2*al; mu=1/s.sqrt(3)
sig=sig0+mu*w
rho0=-(sig0-sig0**3/3)/(2*mu)
rho=rho0+(sig-sig**3/3)/(2*mu)
tau=s.integrate(-be*sig*rho/2,w)
h=s.integrate(-al*rho,w)
Hdef=sig*(be*ax**2+2*al*ax*bx+be*bx**2)/2+rho*(ax+bx)+tau
Pdef=-al*(ax**2+bx**2)/2-be*ax*bx+h
assert s.degree(tau,w)==5 and s.degree(h,w)==4
sub={w:z*(da-db),ax:z*ax,bx:z*bx}
P2=s.expand(Pdef.subs(sub,simultaneous=True)).coeff(z,2)
H2=s.expand(Hdef.subs(sub,simultaneous=True)).coeff(z,2)
P2expected=-al*(ax**2+bx**2)/2-be*ax*bx-al*(1-sig0**2)*(da-db)**2/4
H2expected=sig0*(be*(ax**2+bx**2)/2+al*ax*bx)+(1-sig0**2)*(da-db)*(ax+bx)/2-be*sig0*(1-sig0**2)*(da-db)**2/8
assert s.simplify(P2-P2expected)==0
assert s.simplify(H2-H2expected)==0
assert P2.subs({ax:1,bx:0,da:0,db:0}) == -al/2
gamma, eta = s.symbols('gamma eta', real=True)
SD=-eta*s.sin(2*s.pi/3-gamma)+eta*s.sin(-2*s.pi/3-gamma)
assert s.trigsimp(SD+eta*s.sqrt(3)*s.cos(gamma))==0
q=mp.mpf('1e20'); sigma=mp.mpf('-1.5'); gam=mp.acos(sigma/2)
data=[(2*mp.pi/3-gam,-1),(-2*mp.pi/3-gam,1)]
S=sum(c*mp.sin(p) for p,c in data)
assert abs(-((1+2*q*q)/mp.sqrt(1+q*q))*(theta(q,data)-theta(-q,data))/(4*mp.pi)-S/mp.pi)<mp.mpf('1e-35')
assert abs(S+mp.sqrt(3)*sigma/2)<mp.mpf('1e-60')
print('PASS: nonlinear defect potential, nonzero localized quadratic momentum, defect UV slope=',mp.nstr(S/mp.pi,12))
print('Failures recorded: published profile, zero weighted cosine sum, sign of -dL1/dalpha, and linear defect-potential assertions fail; no implementation assertion failed in this script.')
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""65-digit reanalysis of saved finite-sea data, without executing old scripts.

The arithmetic is high precision; the input solves are double precision.
Error envelopes are sensitivity diagnostics, not confidence intervals.
"""
from pathlib import Path
import json, hashlib, time
import mpmath as m
import sympy as s

start=time.monotonic();m.mp.dps=65
HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'frame_edge_data.json').read_text(),parse_float=m.mpf)
saved=json.loads((HERE/'frame_analysis.json').read_text(),parse_float=m.mpf)
assert hashlib.sha256((HERE/'frame_edge_data.json').read_bytes()).hexdigest()==saved['input_sha256']['frame_edge_data.json']
base=[r for r in raw if r['factor']==1];fine=[r for r in raw if r['factor']!=1]
assert len(base)==5 and len(fine)==2
hs=[r['h'] for r in base];b=m.exp(-m.mpf('.5'))

# Exact unit and chain-rule checks, including the factor THREE.
a,bb,h,hB,PT,ET,cE,y=s.symbols('a b h hB PT ET cE y',positive=True)
PB=a*PT/(6*s.sqrt(3)*bb)
assert s.simplify((h*PT*cE/(6*s.sqrt(3)*bb**2)).subs(h,3*bb*hB)/(hB*PB)-3*cE/a)==0
kap=s.symbols('kappa',positive=True)
P0=4*bb/a*(kap*s.cosh(kap)-s.sinh(kap))
E0=2*bb**2*(s.sinh(kap)*s.cosh(kap)-kap)
assert s.simplify(s.diff(E0,kap)/s.diff(P0,kap)-a*bb*s.sinh(kap)/kap)==0
assert s.simplify(s.diff(P0.subs(kap,s.sqrt(y*h/bb)),y)-2*h*s.sinh(s.sqrt(y*h/bb))/a)==0
cfn=s.Function('c')(y)
assert s.simplify(s.diff(y**s.Rational(3,2)*cfn,y)/s.diff(y**s.Rational(3,2),y)-cfn-s.Rational(2,3)*y*s.diff(cfn,y))==0
beta=s.symbols('beta',positive=True)
assert s.limit(P0.subs(kap,s.sqrt(3)*beta)*a/(6*s.sqrt(3)*bb*beta**3),beta,0)==s.Rational(2,3)
assert s.limit(E0.subs(kap,s.sqrt(3)*beta)/(6*s.sqrt(3)*bb**2*beta**3),beta,0)==s.Rational(2,3)
assert s.limit((E0-a*bb*P0).subs(kap,s.sqrt(3)*beta)/(6*s.sqrt(3)*bb**2*beta**5),beta,0)==s.Rational(1,5)

def linear_fit(xs,fs,funcs):
    A=m.matrix([[f(x) for f in funcs] for x in xs])
    # Small, well-conditioned-enough systems evaluated at 65 digits.
    W=(A.T*A)**-1*A.T
    coef=W*m.matrix(fs)
    return coef,list(W[0,:]),max(abs((A*coef)[i]-fs[i]) for i in range(len(xs)))

def edge(values):
    opts={}
    for name,n,funcs in [('quadratic',3,[lambda h:1,lambda h:h,lambda h:h*h]),
                         ('linear',2,[lambda h:1,lambda h:h]),
                         ('cubic',4,[lambda h:1,lambda h:h,lambda h:h*h,lambda h:h**3]),
                         ('hlog',4,[lambda h:1,lambda h:h,lambda h:h*h,lambda h:h*m.log(h)])]:
        coeff,w,res=linear_fit(hs[-n:],values[-n:],funcs)
        opts[name]={'value':coeff[0],'weight_norm':sum(abs(x) for x in w)}
    center=opts['quadratic']['value']
    return dict(value=center,edge_spread=max(abs(o['value']-center) for o in opts.values()),models=opts)

def point(row,y):return next(p for p in row['points'] if p['y']==y)
def report(fun):
    vals=[fun(row) for row in base];out=edge(vals)
    mesh=max(abs(fun(r)-fun(next(x for x in base if x['h']==r['h']))) for r in fine)
    out.update(raw=vals,mesh=mesh,mesh_propagated=mesh*max(o['weight_norm'] for o in out['models'].values()))
    return out

rows=[]
for yy in map(m.mpf,['0.1','0.5','1','2','4','8','16','32','64']):
    def energy(r):
        p=point(r,yy)
        # Reconstruct from the actual saved energies, rather than fitting a target.
        ratio=(p['E']-p['Ecl'])/(r['h']*p['P'])
        assert abs(ratio-p['dispersion_ratio'])<m.mpf('2e-10')
        return 3*p['dispersion_ratio']
    def velocity(r):
        p=point(r,yy)
        return 3*(p['V']/p['D']-b*m.sinh(p['k'])/p['k'])/r['h']
    ee=report(energy);vv=report(velocity)
    old=next(p for p in saved['one_body'] if p['y']==yy)['energy_coefficient']
    assert abs(ee['value']-3*old['value'])<m.mpf('1e-12')
    assert abs(ee['edge_spread']-3*old['edge_spread'])<m.mpf('1e-12')
    rows.append(dict(y=yy,energy=ee,velocity=vv))

families=[('inverse',[lambda y:1,lambda y:1/y]),
          ('inverse_squared',[lambda y:1,lambda y:1/y,lambda y:1/y**2]),
          ('inverse_sqrt_and_inverse',[lambda y:1,lambda y:1/m.sqrt(y),lambda y:1/y]),
          ('log_over_y',[lambda y:1,lambda y:1/y,lambda y:m.log(y)/y])]
def asymptotic(quantity):
    fits=[]
    for lower in (4,8,16):
        rr=[r for r in rows if r['y']>=lower]
        for name,funcs in families:
            coeff,weights,res=linear_fit([r['y'] for r in rr],[r[quantity]['value'] for r in rr],funcs)
            fits.append(dict(lower_y=lower,model=name,value=coeff[0],
                edge_propagated=sum(abs(w)*r[quantity]['edge_spread'] for w,r in zip(weights,rr)),
                mesh_propagated=sum(abs(w)*r[quantity]['mesh_propagated'] for w,r in zip(weights,rr)),
                max_residual=res,coefficients=list(coeff)))
    central=fits[0]
    model_spread=max(abs(f['value']-central['value']) for f in fits)
    envelope=max(abs(f['value']-central['value'])+f['edge_propagated']+f['mesh_propagated'] for f in fits)
    return dict(central=central,model_spread=model_spread,envelope=envelope,fits=fits)

energy_fit=asymptotic('energy');velocity_fit=asymptotic('velocity')
sound=report(lambda r:3*r['sound_correction'])
assert abs(sound['value']-3*saved['sound']['value'])<m.mpf('1e-12')
fixed=[]
for ii,pp in enumerate(map(m.mpf,['.15','.4','.6'])):
    rr=report(lambda row:row['fixed'][ii]['E1'])
    assert abs(rr['value']-saved['fixed_P_energy'][ii]['coefficient']['value'])<m.mpf('1e-12')
    fixed.append(dict(P_T=pp,coefficient_T=rr))

C=energy_fit['central']['value'];V=velocity_fit['central']['value'];formal=2/m.pi
assert abs(C-V)<energy_fit['envelope']+velocity_fit['envelope']
assert formal-C>50*energy_fit['envelope']
out=dict(precision=m.mp.dps,input_precision='Saved finite-sea solves use double precision.',
    input_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('frame_edge_data.json','frame_analysis.json','../frame_dictionary.tex','../det_projected_oneloop.tex','../toda_route.tex','../oneloop.tex')},
    rows=rows,energy_asymptotic=energy_fit,velocity_asymptotic=velocity_fit,sound=sound,fixed_P=fixed,
    continuum_formal_coefficient=formal,formal_minus_Toda=formal-C,
    naive_good_minus_Toda=-2/m.pi-C,Toda_minus_sound=C-sound['value'],
    formal_to_Toda_ratio=formal/C,runtime_seconds=time.monotonic()-start,
    failures_recorded=['The formal continued continuum coefficient is outside every fitted Toda diagnostic interval.',
      'Finite-y edge extrapolations do not prove the fixed-P semiclassical limit or commutation of limits; that unbounded systematic is not included in the fit envelope.'],
    limitations=['No two-body phase or Phase 2i formula is used.',
      'The 12 fit models/windows are diagnostics, not a statistically calibrated error distribution.',
      'No microscopic finite-sea solver or historical checker is rerun or newly certified.'])

def encode(x):
    if isinstance(x,m.mpf):return float(x)
    raise TypeError(type(x).__name__)
Path(__file__).with_name('onebody_toda_results.json').write_text(json.dumps(out,default=encode,indent=2)+'\n')
for r in rows:
    print('y',float(r['y']),'C_E',m.nstr(r['energy']['value'],14),'edge',m.nstr(r['energy']['edge_spread'],4),'C_v',m.nstr(r['velocity']['value'],14))
for name,fit in [('energy',energy_fit),('velocity',velocity_fit)]:
    print(name,'central',m.nstr(fit['central']['value'],15),'model spread',m.nstr(fit['model_spread'],6),'envelope',m.nstr(fit['envelope'],6))
print('sound',m.nstr(sound['value'],15),'formal minus Toda',m.nstr(formal-C,15))
print('ALL CHECKS PASSED')

#!/usr/bin/env python3
"""60-digit unconstrained extrapolations and separately propagated error budgets.
No constant hypothesis is used in any fitted design matrix or intercept.
"""
from pathlib import Path
import json,hashlib,time
import mpmath as m
import sympy as sp
m.mp.dps=60;START=time.monotonic();HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'onebody_prec_scan.json').read_text(),parse_float=m.mpf)['rows']
assert len(raw)==14
base=sorted([r for r in raw if r['factor']==1],key=lambda r:-r['h']);fine=[r for r in raw if r['factor']!=1]
wh=json.loads((HERE/'onebody_prec_edge.json').read_text())['rows']
wh={m.mpf(r['y']):{k:m.mpf(r[k]) for k in ['energy','velocity','du','u']} for r in wh}
d0=m.mpf(1)/48-1/(4*m.pi**2)
# Exact inherited unit conversion, including the factor three.
bb,hh,hB,pp,cc=sp.symbols('b h hB P c',positive=True)
PB=pp/(6*sp.sqrt(3)*bb)
assert sp.simplify((hh*pp*cc/(6*sp.sqrt(3)*bb**2)).subs(hh,3*bb*hB)/(hB*PB)-3*cc)==0
def fit(xs,vs,fs):
 A=m.matrix([[f(x) for f in fs] for x in xs]);W=(A.T*A)**-1*A.T;c=W*m.matrix(vs)
 return c,list(W[0,:]),max(abs(v) for v in A*c-m.matrix(vs))
def hfits(vals):
 opts=[]
 specs=[('quartic7',7,4,None),('cubic6',6,3,None),('quintic8',8,5,None),('quadratic4',4,2,None),('h2log7',7,3,2),('hlog7',7,3,1)]
 for name,n,deg,logpow in specs:
  rr=base[-n:];scale=rr[0]['h'];xs=[r['h']/scale for r in rr]
  fs=[lambda x,j=j:x**j for j in range(deg+1)]
  if logpow:fs.append(lambda x,j=logpow:x**j*m.log(x))
  c,w,res=fit(xs,vals[-n:],fs)
  opts.append(dict(name=name,value=c[0],weights=w,weight_norm=sum(abs(t) for t in w),residual=res))
 return opts

def point(r,y):return next(p for p in r['points'] if p['y']==y)
rows=[]
for yy in [m.mpf(p['y']) for p in base[0]['points']]:
 row=dict(y=yy)
 for qty in ['energy','velocity','u_remainder','du_remainder']:
  fs=hfits([point(r,yy)[qty] for r in base]);center=fs[0]['value']
  delta=max(abs(point(r,yy)[qty]-point(next(x for x in base if x['h']==r['h']),yy)[qty]) for r in fine)
  # Add a double-arithmetic floor rather than calling high-precision fitting high-precision sea data.
  floor=m.mpf('2e-10');mesh=(delta+floor)*max(f['weight_norm'] for f in fs)
  row[qty]=dict(value=center,h_error=max(abs(f['value']-center) for f in fs),mesh_error=mesh,mesh_difference=delta,fits=fs)
  if qty in ['energy','velocity'] and yy in wh:
   row[qty]['halfline_difference']=abs(center-wh[yy][qty])
 row['u_log_subtracted']=row['u_remainder']['value']+m.log(yy)/(2*m.pi)-(1-m.log(8*m.pi))/(2*m.pi)
 rows.append(row)

def amplitude(qty):
 fits=[]
 # Known first power tail is subtracted; the constant remains wholly free.
 tail=lambda y:-3*d0/(2*y) if qty=='energy' else -d0/(2*y)
 for lo,hi in [(4,32),(8,32),(8,64),(16,64)]:
  rr=[r for r in rows if lo<=r['y']<=hi]
  for name,powers,log in [('inv2',[0,2],False),('inv2_inv3',[0,2,3],False),('fractional',[0,1.5,2],False),('log_inv2',[0,2],True)]:
   fs=[lambda y,p=m.mpf(str(p)):y**(-p) for p in powers]
   if log:fs.append(lambda y:m.log(y)/y**2)
   c,w,res=fit([r['y'] for r in rr],[r[qty]['value']-tail(r['y']) for r in rr],fs)
   fits.append(dict(lower=lo,upper=hi,model=name,value=c[0],h_error=sum(abs(t)*r[qty]['h_error'] for t,r in zip(w,rr)),mesh_error=sum(abs(t)*r[qty]['mesh_error'] for t,r in zip(w,rr)),weights=w,residual=res))
 central=next(f for f in fits if f['lower']==8 and f['upper']==32 and f['model']=='inv2_inv3')
 center=central['value'];model=max(abs(f['value']-center) for f in fits)
 he=max(f['h_error'] for f in fits);me=max(f['mesh_error'] for f in fits)
 return dict(value=center,mesh_error=me,h_error=he,amplitude_error=model,total_error=me+he+model,fits=fits)
energy=amplitude('energy');velocity=amplitude('velocity')
sound=hfits([r['sound'] for r in base]);soundv=sound[0]['value'];sounderr=max(abs(f['value']-soundv) for f in sound)
# Independent high-precision half-line extrapolation; still a free intercept.
wr=sorted(wh.items());whfits=[]
for lo in [16,32,64,128]:
 rr=[(y,r) for y,r in wr if y>=lo]
 for powers in [[0,2,3],[0,2,3,4],[0,1.5,2,3]]:
  if len(rr)<len(powers):continue
  c,w,res=fit([y for y,r in rr],[r['energy']+3*d0/(2*y) for y,r in rr],[lambda y,p=m.mpf(str(p)):y**(-p) for p in powers])
  whfits.append(dict(lower=lo,powers=powers,value=c[0],residual=res))
C=energy['value'];err=energy['total_error']
assert err<m.mpf('.00334')/10
assert abs(C-velocity['value'])<err+velocity['total_error']
for row in rows:
 if row['y'] in wh and row['y']<=64:
  for qty in ['energy','velocity']:
   assert row[qty]['halfline_difference']<row[qty]['h_error']+row[qty]['mesh_error'],(row['y'],qty,row[qty])
# Independently test the momentum logarithm and the edge velocity identity.
for row in rows:
 yy=row['y']
 if yy in wh and yy<=64:
  for qty,expected in [('u_remainder',wh[yy]['u']-yy),('du_remainder',wh[yy]['du']-1)]:
   assert abs(row[qty]['value']-expected)<row[qty]['h_error']+row[qty]['mesh_error'],(yy,qty)
  diff=abs(row['velocity']['value']-soundv-yy*row['du_remainder']['value']/2)
  budget=row['velocity']['h_error']+row['velocity']['mesh_error']+sounderr+yy*(row['du_remainder']['h_error']+row['du_remainder']['mesh_error'])/2
  assert diff<budget,(yy,diff,budget)
# Only now compare the hypotheses.
targets={'1/(2pi)':1/(2*m.pi),'1/6':m.mpf(1)/6,'sqrt(3)/(2pi)^2':m.sqrt(3)/(2*m.pi)**2,
 '1/(2pi)*(1+1e-3)':(1+m.mpf('.001'))/(2*m.pi),'1/(2pi)*(1+1e-4)':(1+m.mpf('.0001'))/(2*m.pi),'1/(2pi)*(1+1e-5)':(1+m.mpf('.00001'))/(2*m.pi)}
tests={name:dict(value=v,difference=C-v,envelopes=abs(C-v)/err,compatible=abs(C-v)<=err) for name,v in targets.items()}
assert tests['1/(2pi)']['compatible']
assert not tests['1/6']['compatible'] and not tests['sqrt(3)/(2pi)^2']['compatible']
prov={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'onebody_prec_scan.json',HERE/'onebody_prec_edge.json',HERE/'../frame_dictionary.tex',HERE/'../onebody_dispersion_test.tex']}
out=dict(precision=60,sea_precision='IEEE double; high-precision fits do not increase input precision',rows=rows,energy=energy,velocity=velocity,sound=dict(value=soundv,h_error=sounderr),halfline_fits=whfits,hypothesis_tests=tests,relative_small_interval=[2*m.pi*(C-err)-1,2*m.pi*(C+err)-1],input_sha256=prov,seconds=time.monotonic()-START)
def encode(x):
 if isinstance(x,m.mpf):return str(x)
 raise TypeError(type(x).__name__)
(HERE/'onebody_prec_analysis.json').write_text(json.dumps(out,default=encode,indent=2)+'\n')
for qty,result in [('energy',energy),('velocity',velocity)]:
 print(qty,{k:m.nstr(v,13) for k,v in result.items() if k!='fits'})
print('sound',m.nstr(soundv,15),'h error',m.nstr(sounderr,4))
print('relative small interval',[m.nstr(x,6) for x in out['relative_small_interval']])
print('halfline intercept range',m.nstr(min(f['value'] for f in whfits),20),m.nstr(max(f['value'] for f in whfits),20))
print('ALL CHECKS PASSED')

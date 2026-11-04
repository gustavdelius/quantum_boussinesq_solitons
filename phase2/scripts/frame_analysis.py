#!/usr/bin/env python3
"""Reproducible diagnostics, not certified asymptotic error bounds.
Reads frame_edge_data.json; outputs frame_analysis.json. All fits are unconstrained.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
START=time.monotonic(); HERE=Path(__file__).resolve().parent; b=np.exp(-.5)
allrows=json.loads((HERE/'frame_edge_data.json').read_text())
rows=[r for r in allrows if r['factor']==1.]
assert len(rows)==5 and len(allrows)==7
hs=np.array([r['h'] for r in rows])
def points(row):return {p['y']:p for p in row['points']}
def estimates(v):
    v=np.asarray(v); h=hs
    quad=np.polyfit(h[-3:],v[-3:],2)[-1]
    linear=np.polyfit(h[-2:],v[-2:],1)[-1]
    cubic=np.polyfit(h[-4:],v[-4:],3)[-1]
    hlog=np.linalg.solve(np.array([np.ones(4),h[-4:],h[-4:]**2,h[-4:]*np.log(h[-4:])]).T,v[-4:])[0]
    return dict(value=float(quad),linear=float(linear),cubic=float(cubic),hlog=float(hlog),
                edge_spread=float(max(abs(quad-linear),abs(quad-cubic),abs(quad-hlog))))
def mesh(fun):
    return max(abs(fun(row)-fun(next(r for r in rows if r['h']==row['h']))) for row in allrows if row['factor']!=1.)
# A conservative *diagnostic* propagation: worst measured fine-mesh discrepancy
# times the largest l1 norm of the four edge extrapolation weight vectors.
w=[np.polyfit(hs[-3:],np.eye(3),2)[-1],np.polyfit(hs[-2:],np.eye(2),1)[-1],
   np.polyfit(hs[-4:],np.eye(4),3)[-1],
   np.linalg.inv(np.array([np.ones(4),hs[-4:],hs[-4:]**2,hs[-4:]*np.log(hs[-4:])]).T)[0]]
amp=max(np.sum(abs(v)) for v in w)
def report(fun):
    v=[fun(row) for row in rows];e=estimates(v)
    e.update(raw=v,mesh=mesh(fun),mesh_propagated=amp*mesh(fun));return e
one=[]
for y in (.1,.5,1.,2.,4.,8.,16.,32.,64.):
    def du(row):
        p=points(row)[y];return p['u']-2*b*(np.cosh(p['k'])-1)/row['h']
    def jac(row):
        p=points(row)[y];return p['du_dy']-np.sinh(p['k'])/p['k']
    one.append(dict(y=y,u_minus_y=report(du),jacobian_minus_one=report(jac),
                    energy_coefficient=report(lambda row:points(row)[y]['dispersion_ratio'])))
def remainder(row,y,z):
    p,t=points(row)[y],points(row)[z]
    def C(p,t):return -2*p['k']*(t['P']-2*b*np.sinh(t['k']))
    return (p['M']*t['Q']+t['M']*p['Q']-C(p,t)-C(t,p))/row['h']
rect=[]
for Y in (4.,8.,16.,32.):
    def rectfun(row):return remainder(row,2*Y,Y/2)-remainder(row,Y,Y/2)-remainder(row,2*Y,Y/4)+remainder(row,Y,Y/4)
    out=report(rectfun);out['Y']=Y;rect.append(out)
assert all(abs(e['value'])>10*(e['edge_spread']+e['mesh_propagated']) for e in rect[:3])
# Do not assert large-Y convergence: fit dependence is explicitly retained.
yy=np.array([e['Y'] for e in rect]);vv=np.array([e['value'] for e in rect])
fitrect={name:float(np.linalg.lstsq(A,vv,rcond=None)[0][0]) for name,A in {
    'constant_plus_inverse_sqrt':np.array([np.ones(4),1/np.sqrt(yy)]).T,
    'constant_plus_inverse':np.array([np.ones(4),1/yy]).T,
    'constant_plus_inverse_sqrt_and_inverse':np.array([np.ones(4),1/np.sqrt(yy),1/yy]).T}.items()}
fixed=[]
for i,P in enumerate((.15,.4,.6)):
    fixed.append(dict(P=P,coefficient=report(lambda row:row['fixed'][i]['E1'])))
# A one-body-reference-free rectangle of the proposed phase ambiguity.
def ambiguity(y,z):return y*z*(y-z)/(y+z)**3
ambrect=ambiguity(16,4)-ambiguity(8,4)-ambiguity(16,2)+ambiguity(8,2)
assert abs(ambrect)>1e-3
# Preserve source provenance without importing the old scripts (which can write).
inputs=['frame_edge.py','frame_edge_data.json','frame_identities.py','frame_identities.json',
        'toda_edge_ba.py','toda_bethe_quantum.py','matching_toda.py','matching_toda_data.json',
        'matching_toda_fine.json','matching_analysis.json','audit_toda_dictionary_results.json']
prov={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in inputs}
out=dict(one_body=one,rectangles=rect,rectangle_large_Y_fit_intercepts=fitrect,
         fixed_P_energy=fixed,sound=report(lambda row:row['sound_correction']),
         mesh_amplification=amp,ambiguity_rectangle=ambrect,input_sha256=prov,
         runtime_seconds=time.monotonic()-START,
         limitations=['Fit spreads are deterministic sensitivity diagnostics, not rigorous bounds.',
         'No physical scalar frame is selected; no unconditional one-loop phase prediction is delivered.',
         'Large-Y rectangle intercept depends on fitting model; no commuting-limit theorem follows.'])
(HERE/'frame_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for e in one:print('y',e['y'],'u-y',e['u_minus_y']['value'],'cE',e['energy_coefficient']['value'],'edge',e['energy_coefficient']['edge_spread'])
print('Rectangles',[(e['Y'],e['value'],e['edge_spread'],e['mesh']) for e in rect])
print('Large Y models',fitrect)
print('Fixed P',[(e['P'],e['coefficient']['value'],e['coefficient']['edge_spread']) for e in fixed])
print('ALL CHECKS PASSED')

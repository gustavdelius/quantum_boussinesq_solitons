#!/usr/bin/env python3
"""Read-only 60-digit regressions; empirical error components stay separate.
All input hashes are saved and checked. Only uniform_limits_refit.json is written.
"""
from pathlib import Path
import hashlib,json
import mpmath as m
m.mp.dps=60
HERE=Path(__file__).resolve().parent
HASH={}
def read(name):
    p=HERE/name; raw=p.read_bytes();HASH[name]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw,parse_float=m.mpf)
def fit(xs,ys,fs):
    X=m.matrix([[f(x) for f in fs] for x in xs]);W=(X.T*X)**-1*X.T;c=W*m.matrix(ys)
    return dict(coef=list(c),weights=[list(W[j,:]) for j in range(W.rows)],residual=max(abs(t) for t in X*c-m.matrix(ys)),weight_norm=sum(abs(t) for t in W[0,:]))
def hf(hs,ys):
    out=[]
    for n in [3,4,5]:
        if n>len(hs):continue
        xx=[v/hs[-n] for v in hs[-n:]];vv=ys[-n:]
        for name,powr,log in [('quadratic',2,None),('cubic',3,None),('hlog',1,1),('h2log',2,2),('quadratic_hlog',2,1)]:
            fs=[lambda x,j=j:x**j for j in range(powr+1)]
            if log:fs.append(lambda x,j=log:x**j*m.log(x))
            if len(fs)>n:continue
            row=fit(xx,vv,fs);row.update(name=name,n=n)
            out.append(row)
    return out

def classical(h,Y,r):
    b=m.exp(-m.mpf('.5'));k=m.sqrt(Y*h/b);l=r*k
    la=2*m.log(m.sinh((k-l)/2)/m.sinh((k+l)/2))
    return (2*b*(m.cosh(k)-m.cosh(l))*la+4*b*(l*m.sinh(k)-k*m.sinh(l)))/h
out={}
old=read('toda_edge_twoterm.json');failed=[]
for key,rows in old.items():
    for a,b in zip(rows[:-1],rows[1:]):
        change=abs(b[1]-a[1])
        if change>=m.mpf('.004'):failed.append(dict(point=key,h_from=a[0],h_to=b[0],change=change))
assert len(failed)==3
out['phase2d_failed_bound']=failed
out['phase2d_step_ratios']={key:abs(rows[2][1]-rows[1][1])/abs(rows[1][1]-rows[0][1]) for key,rows in old.items()}
saved=read('matching_toda_data.json')+read('matching_toda_fine.json');saved.sort(key=lambda x:-x['h'])
mesh=read('matching_toda_nodes.json')[0];base=next(x for x in saved if x['h']==mesh['h'])
mesherr=max(abs(a['phase']-b['phase']) for a,b in zip(mesh['same'],base['same']))
# Sensitivity of uncorrected published subleading signs is reproduced explicitly.
oldfit=read('matching_analysis.json');signs=[]
for r in map(m.mpf,['.4','.5','.6']):
    pts=[next(x for x in oldfit['same'] if x['Y']==Y and x['r']==r) for Y in [16,32]]
    vals={key:32*(pts[0][key]-pts[1][key]) for key in ['residual','cubic','h_log_h_intercept']}
    assert vals['residual']<0<vals['h_log_h_intercept'];signs.append(dict(r=r,slopes=vals))
out['published_sign_reversal']=signs
# Correct classical curvature, then ALL h bases and several Y bases, free c0.
savedfits=[]
for r in map(m.mpf,['.4','.5','.6']):
    edge=[]
    for Y in [4,8,16,32]:
        vals=[]
        for row in saved:
            p=next(p for p in row['same'] if p['r']==r and p['Y']==Y)
            vals.append(p['phase']-classical(row['h'],Y,r))
        edge.append(hf([v['h'] for v in saved],vals))
    for j,hh in enumerate(edge[0]):
        values=[v[j]['coef'][0] for v in edge]
        for indices in [[1,2,3],[0,1,2,3]]:
            for name,fs in [('power',[lambda x:1,lambda x:1/x,lambda x:1/x**2]),('log',[lambda x:1,lambda x:1/x,lambda x:m.log(x)/x]),('log2',[lambda x:1,lambda x:1/x,lambda x:m.log(x)**2/x])]:
                yy=[m.mpf([4,8,16,32][i]) for i in indices]
                f=fit(yy,[values[i] for i in indices],fs)
                f.update(r=r,h_basis=hh['name'],h_points=hh['n'],Y_indices=indices,Y_basis=name,edge=values)
                savedfits.append(f)
out['saved_classical_subtracted_fits']=savedfits
out['saved_mesh_at_h01']=mesherr
# One-body original independent data, renewed nested free-intercept fit.
raw=read('onebody_prec_scan.json')['rows'];base=sorted([r for r in raw if r['factor']==1],key=lambda x:-x['h']);fine=[r for r in raw if r['factor']!=1]
d=m.mpf(1)/48-1/(4*m.pi**2)
points=[]
def point(row,y):return next(p for p in row['points'] if p['y']==y)
for y in [4,6,8,12,16,24,32,48,64]:
    fs=[]
    for name,n,deg,lp in [('quartic7',7,4,0),('cubic6',6,3,0),('quintic8',8,5,0),('quad4',4,2,0),('hlog7',7,3,1),('h2log7',7,3,2)]:
        rr=base[-n:];basis=[lambda x,j=j:x**j for j in range(deg+1)]
        if lp:basis.append(lambda x,lp=lp:x**lp*m.log(x))
        f=fit([r['h']/rr[0]['h'] for r in rr],[point(r,y)['energy'] for r in rr],basis);f['name']=name;fs.append(f)
    central=fs[0]['coef'][0]
    delta=max(abs(point(r,y)['energy']-point(next(b for b in base if b['h']==r['h']),y)['energy']) for r in fine)
    points.append(dict(y=y,value=central,h_error=max(abs(f['coef'][0]-central) for f in fs),mesh_error=(delta+m.mpf('2e-10'))*max(f['weight_norm'] for f in fs),fits=fs))
amp=[]
for lo,hi in [(4,32),(8,32),(8,64),(16,64)]:
    rr=[r for r in points if lo<=r['y']<=hi]
    for name,powers,lg in [('power2',[0,2],False),('power23',[0,2,3],False),('fractional',[0,m.mpf('1.5'),2],False),('log2',[0,2],True)]:
        fs=[lambda y,p=p:y**(-p) for p in powers]
        if lg:fs.append(lambda y:m.log(y)/y**2)
        f=fit([m.mpf(r['y']) for r in rr],[r['value']+3*d/(2*r['y']) for r in rr],fs)
        f.update(lo=lo,hi=hi,name=name,h_error=sum(abs(w)*r['h_error'] for w,r in zip(f['weights'][0],rr)),mesh_error=sum(abs(w)*r['mesh_error'] for w,r in zip(f['weights'][0],rr)))
        amp.append(f)
center=next(f['coef'][0] for f in amp if f['lo']==8 and f['hi']==32 and f['name']=='power23')
he=max(f['h_error'] for f in amp);me=max(f['mesh_error'] for f in amp);ae=max(abs(f['coef'][0]-center) for f in amp)
assert abs(center-1/(2*m.pi))<he+me+ae
out['onebody']=dict(value=center,h_error=he,mesh_error=me,amplitude_error=ae,total=he+me+ae,amplitude_fits=amp,points=points)
# Exact half-line convolution fits, with no target c0 or c1 constrained.
wh=read('uniform_limits_wh.json')['rows'];whfits=[];whsummary=[]
for r in map(m.mpf,['.4','.5','.6']):
    rr=[p for p in wh if m.mpf(p['r'])==r];variants=[]
    for lo in [16,32]:
        pts=[p for p in rr if p['Y']>=lo];xs=[m.mpf(p['Y']) for p in pts]
        specs=[('powers',[lambda x:1,lambda x:1/x,lambda x:1/x**2,lambda x:1/x**3]),('fractional',[lambda x:1,lambda x:1/x,lambda x:x**(-m.mpf('1.5')),lambda x:1/x**2]),('logs',[lambda x:1,lambda x:1/x,lambda x:m.log(x)/x,lambda x:m.log(x)**2/x,lambda x:1/x**2])]
        for name,fs in specs:
            f=fit(xs,[m.mpf(p['residual']) for p in pts],fs)
            f.update(r=r,lower=lo,name=name,quad_c1=sum(abs(w)*(m.mpf(p['quad_error'])+m.mpf('1e-28')) for w,p in zip(f['weights'][1],pts)))
            variants.append(f);whfits.append(f)
    center=next(f for f in variants if f['name']=='powers' and f['lower']==32)
    spread=max(abs(f['coef'][1]-center['coef'][1]) for f in variants)
    quad=max(f['quad_c1'] for f in variants)
    target=-r**3/(6*(1-r*r))-2*d*((1-r*r)/r**2*m.log((1-r)/(1+r))+2/r)
    assert all(f['coef'][1]<0 for f in variants)
    assert abs(center['coef'][1]-target)<spread+quad
    whsummary.append(dict(r=r,c0=center['coef'][0],c1=center['coef'][1],c1_spread=spread,c1_quad=quad,formal_c1=target,range=[min(f['coef'][1] for f in variants),max(f['coef'][1] for f in variants)]))
out['wh_fits']=whfits;out['wh_summary']=whsummary
fresh=read('uniform_limits_scan.json')['rows'];base=sorted([r for r in fresh if r['factor']==1],key=lambda r:-r['h']);fine=[r for r in fresh if r['factor']!=1]
fixed=[]
for j in range(6):
    row=dict(k=base[0]['fixed'][j]['kappa_fixed'])
    for quantity in ['energy','phase']:
        get=lambda r:r['fixed'][j]['energy'] if quantity=='energy' else r['fixed'][j]['two_body']['residual']
        fs=hf([r['h'] for r in base],[get(r) for r in base]);central=next(f for f in fs if f['name']=='quadratic' and f['n']==3)
        de=max(abs(get(r)-get(next(b for b in base if b['h']==r['h']))) for r in fine)
        row[quantity]=dict(value=central['coef'][0],spread=max(abs(f['coef'][0]-central['coef'][0]) for f in fs),mesh=(de+m.mpf('2e-10'))*max(f['weight_norm'] for f in fs),fits=fs)
    fixed.append(row)
out['fixed_kappa']=fixed
# Fresh edge comparison separates mesh, h-basis spread and WH discrepancy.
edge=[]
for j in range(18):
    p=base[0]['edge'][j];fs=hf([r['h'] for r in base],[r['edge'][j]['residual'] for r in base])
    central=next(f for f in fs if f['name']=='cubic' and f['n']==4)['coef'][0]
    de=max(abs(r['edge'][j]['residual']-next(b for b in base if b['h']==r['h'])['edge'][j]['residual']) for r in fine)
    row=dict(Y=p['Y'],r=p['r'],value=central,h_spread=max(abs(f['coef'][0]-central) for f in fs),mesh=(de+m.mpf('2e-10'))*max(f['weight_norm'] for f in fs))
    candidates=[v for v in wh if v['Y']==p['Y'] and m.mpf(v['r'])==p['r']]
    if candidates:row['wh_difference']=abs(central-m.mpf(candidates[0]['residual']))
    edge.append(row)
out['fresh_edge']=edge
out['input_sha256']=HASH
for name,hash_ in HASH.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==hash_
def encode(x):
    if isinstance(x,m.mpf):return str(x)
    raise TypeError(type(x).__name__)
(HERE/'uniform_limits_refit.json').write_text(json.dumps(out,default=encode,indent=2)+'\n')
print('onebody',m.nstr(out['onebody']['value'],14),'total',m.nstr(out['onebody']['total'],5))
for row in whsummary:print('WH summary',{k:m.nstr(v,12) if isinstance(v,m.mpf) else v for k,v in row.items() if k!='range'})
for row in fixed:print('fixed k',row['k'],'energy',m.nstr(row['energy']['value'],12),'+/-',m.nstr(row['energy']['spread'],3),'phase',m.nstr(row['phase']['value'],10),'+/-',m.nstr(row['phase']['spread'],3))
print('ALL CHECKS PASSED')

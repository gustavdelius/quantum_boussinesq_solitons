"""Cache conditional ingredients and unknown physical entries explicitly as null.

Angles are radians. Exchange swaps BOTH alpha and epsilon. No legacy imports.
"""
import sys
sys.dont_write_bytecode=True
import time
import mpmath as mp
from good_twobody_kernels import (sol,classical,integrals,momentum_mode,momentum_canonical,
                                 interaction,save,quadrature)

def row(a,b,e,f):
    s,t=sol(a,e),sol(b,f)
    c=classical(s,t)
    i,j,sym=integrals(s,t)
    p1,p2=momentum_mode(s),momentum_mode(t)
    pc1,pc2=momentum_canonical(s),momentum_canonical(t)
    rm=c['X1']*p1+c['X2']*p2
    rc=c['X1']*pc1+c['X2']*pc2
    return dict(alpha=[s['a'],t['a']],epsilon=[e,f],P0=[s['P'],t['P']],
                channel='overtaking' if e==f else 'head_on',
                ordering='forward_v1_gt_v2' if s['v']>t['v'] else 'inverse_oriented_generator',
                A=c['A'],logA=c['logA'],I=i,I_symmetric=sym,J_antisymmetric=j,
                R_mode=rm,R_canonical=rc,conditional_mode_representative=j+rm,
                conditional_canonical_representative=j+rc,
                lambda_coefficient=c['lambda_coefficient'],
                eta_coefficient=c['eta_coefficient'],zeta_coefficient=c['zeta_coefficient'],
                kappa_coefficient=c['kappa_coefficient'],tadpole_coefficient=mp.mpf(0),
                physical_symmetric=None,physical_antisymmetric=None,
                physical_status='uncomputed boundary determinant and state map',
                unitary_representative_symmetric=mp.mpf(0),
                unknown_additive_antisymmetric='B12 (mode convention); not assumed zero')

if __name__=='__main__':
    start=time.monotonic()
    rows=[];worst=mp.mpf(0)
    numeric_keys=('I','J_antisymmetric','R_mode','R_canonical','conditional_mode_representative',
                  'lambda_coefficient','eta_coefficient','zeta_coefficient','kappa_coefficient')
    for e,f in ((1,1),(-1,-1),(1,-1),(-1,1)):
        for ia in range(1,11):
            for ib in range(1,11):
                if e==f and ia==ib: continue
                if e!=f and ia+ib>10: continue
                with mp.workdps(55):
                    low=row(mp.mpf(ia)/10,mp.mpf(ib)/10,e,f)
                with mp.workdps(75):
                    high=row(mp.mpf(ia)/10,mp.mpf(ib)/10,e,f)
                    error=max(abs(low[k]-high[k]) for k in numeric_keys)
                    assert error<mp.mpf('1e-48'),(ia,ib,e,f,error)
                    worst=max(worst,error)
                    high['precision_recheck_max_abs_difference']=error
                    rows.append(high)
    mp.mp.dps=75
    # Check exchange on stored, independently computed rows, including species reversal.
    index={(tuple(r['alpha']),tuple(r['epsilon'])):r for r in rows}
    for r in rows:
        reverse=index[(tuple(reversed(r['alpha'])),tuple(reversed(r['epsilon'])))]
        for key in ('J_antisymmetric','R_mode','R_canonical','lambda_coefficient',
                    'eta_coefficient','zeta_coefficient','kappa_coefficient'):
            assert abs(r[key]+reverse[key])<mp.mpf('1e-65')
    save('good_twobody_grid.json',dict(schema_version=1,claim_label='c for specified ingredients only',
         phase_units='radians',precision_dps=[55,75],max_precision_difference=worst,
         expression='delta1_P = J + R_mode + lambda*Klambda + eta*Keta + zeta*Kzeta + kappa*Kkappa + B12',
         expression_status='conditional; B12 and physical normalization uncomputed',
         exchange='swap (alpha1,epsilon1) with (alpha2,epsilon2)',
         other_local_operators='not silently excluded; new interactions require extra response functions',
         rows=rows))
    near=[];limits=[]
    for a in map(mp.mpf,('.2','.4','.5')):
        bstar=mp.pi/3-a
        s,t=sol(a,1),sol(bstar,-1)
        jstar=integrals(s,t)[1]
        cm=-momentum_mode(s)/s['k']+momentum_mode(t)/t['k']
        cc=-momentum_canonical(s)/s['k']+momentum_canonical(t)/t['k']
        cl=2*(t['v']-s['v'])/3
        fm=s['gamma']*t['M']*momentum_mode(s)-t['gamma']*s['M']*momentum_mode(t)
        # On the genuine resonance ell1=ell2, so delta0 tends to top.
        top=2*s['M']*t['M']*(s['v']-t['v'])/3
        fl=3*top+4*(s['P']*s['gamma']*t['M']-t['P']*t['gamma']*s['M'])
        limits.append(dict(alpha1=a,alpha2_limit=bstar,J_limit=jstar,
                           mode_logA_coefficient=cm,canonical_logA_coefficient=cc,
                           lambda_logA_coefficient=cl,mode_finite_limit=jstar+fm,
                           lambda_finite_limit=fl))
        errors=[]
        for exponent in (2,4,6,8,10,12):
            h=mp.mpf(10)**(-exponent)
            r=row(a,bstar-h,1,-1)
            r.update(distance_to_resonance=h,
                     mode_subtracted=r['conditional_mode_representative']-cm*r['logA'],
                     lambda_subtracted=r['lambda_coefficient']-cl*r['logA'])
            error=max(abs(r['J_antisymmetric']-jstar),
                      abs(r['mode_subtracted']-(jstar+fm)),
                      abs(r['lambda_subtracted']-fl))
            errors.append(error)
            r['max_difference_from_subtracted_limits']=error
            near.append(r)
        assert all(y<x for x,y in zip(errors,errors[1:]))
        assert errors[-1]<mp.mpf('1e-8'),errors
        # Direct integral at the endpoint is finite, despite A=infinity there.
        assert abs(quadrature(s,t)-jstar)<mp.mpf('1e-60')
    threshold=[]
    for exponent in (2,4,6,8,10,12):
        h=mp.mpf(10)**(-exponent)
        r=row(mp.mpf('.4'),mp.mpf('.4')+h,1,1)
        r['angle_difference']=h
        threshold.append(r)
    assert abs(threshold[-1]['lambda_coefficient'])<mp.mpf('1e-9')
    # An overtaking Y-line requires a heavy constituent. Do not assign it a real quantum phase.
    heavy=[]
    for exponent in (2,4,6,8,10,12):
        h=mp.mpf(10)**(-exponent)
        a=mp.mpf('.9');b=2*mp.pi/3-a-h
        c=classical(sol(a,1),sol(b,1))
        heavy.append(dict(alpha=[a,b],distance=h,A=c['A'],lambda_coefficient=c['lambda_coefficient'],
                          quantum_phase=None,status='heavy branch; classical response only'))
    falseA=interaction(sol('.2',1),sol(mp.mpf('.2')+mp.pi/3,1))
    assert abs(falseA-mp.mpf('.355465479849'))<mp.mpf('1e-12')
    save('good_twobody_resonance.json',dict(head_on_limits=limits,head_on_approaches=near,
          overtaking_equal_velocity=threshold,overtaking_heavy_Y=heavy,
          false_difference_pi_over_3_A=falseA,
          note='Limits are for specified ingredients; unknown boundary phase can alter total asymptotics.'))
    print('grid rows',len(rows),'max precision difference',mp.nstr(worst,8))
    print('elapsed seconds',round(time.monotonic()-start,2))
    print('ALL CHECKS PASSED')

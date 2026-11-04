#!/usr/bin/env python3
"""Independent Phase 2i audit. Only the permitted finite-h solver is reused.
(c) checks reproduce positive identities AND explicit discrepancies. No physical
frame or regularisation is selected. All output paths have audit_phase2i prefix.
Use --refresh to rebuild finite-h cache. mpmath 60 digits; dense solver float64.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,time,json,hashlib,itertools
from pathlib import Path
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.special import roots_legendre
from scipy.optimize import brentq
from matching_toda import FastNodesEdgeBA
mp.mp.dps=60
HERE=Path(__file__).resolve().parent
OUT=HERE/'audit_phase2i_results.json'
R={}
A=1/(2*mp.pi); C=(1-mp.log(8*mp.pi))/(2*mp.pi)
D=mp.mpf(1)/48-1/(4*mp.pi**2)
E=(4+6*mp.zeta(3)-mp.pi**2)/(96*mp.pi**3)
def n(x): return mp.nstr(x,28)
def save(): OUT.write_text(json.dumps(R,indent=2)+'\n')
def T(s):
    x=s/(2*mp.pi)
    return mp.exp(mp.loggamma(1+x)-x*(mp.log(x)-1))/mp.sqrt(s)
def il(u,order=0):
    # G, G', and ordinary G''; constant subtracted in G'' removes contact delta.
    return mp.invertlaplace(lambda s:mp.sqrt(mp.pi)*(T(s)-(1 if order==2 else 0))*s**(order-2),u,method='talbot')
def yof(u): return (3*il(u)/4)**(mp.mpf(2)/3)
def asym(y): return -A*mp.log(y)+C+D/y+E/y**2

def wh():
    rows=[]
    for w in map(mp.mpf,['.2','1','5','20']):
        prod=2*mp.pi*T(mp.j*w)*T(-mp.j*w)
        correct=2*mp.pi/(1-mp.exp(-w))
        printed=2*mp.pi**2*mp.exp(mp.pi*w)/mp.sinh(mp.pi*w)
        assert abs(prod-correct)<mp.mpf('1e-55')
        assert abs(printed-correct)>1
        # Digamma integral representation: transform of 2cos(xt)/(exp(t)-1)
        # gives the nonzero-frequency contribution 2pi/(exp(|w|)-1).
        rows.append(dict(w=n(w),product=n(mp.re(prod)),printed=n(printed)))
    assert abs(T(mp.mpf('1e9'))-1)<mp.mpf('1e-8')
    # L[4/3*u^(3/2)] = sqrt(pi)*s^-5/2, unlike printed leading coefficient 1.
    assert abs(4*mp.gamma(mp.mpf('2.5'))/3-mp.sqrt(mp.pi))<mp.mpf('1e-58')
    hh,yy,bb=sp.symbols('h y b',positive=True)
    # t=sqrt(h) avoids Puiseux ambiguity in this exact scaling check.
    tt=sp.symbols('t',positive=True);kk=tt*sp.sqrt(yy/bb)
    pn=4*bb*(kk*sp.cosh(kk)-sp.sinh(kk))
    assert sp.simplify(sp.limit(sp.sqrt(bb)*pn/tt**3,tt,0)-4*yy**sp.Rational(3,2)/3)==0
    assert sp.simplify(sp.limit(pn/(sp.sqrt(bb)*tt**3),tt,0)-4*yy**sp.Rational(3,2)/(3*bb))==0
    # The half-line boundary value makes the proposed FOUR-term product fail.
    bzero=il(mp.mpf('1e-10'),1)
    assert abs(bzero-mp.sqrt(mp.pi))<mp.mpf('1e-8')
    # Convolution theorem with causal f_full = sqrt(pi) delta + f:
    t1,t2=sp.symbols('T1 T2')
    assert sp.expand(2*(sp.sqrt(sp.pi)*t1)*(sp.sqrt(sp.pi)*t2)-2*sp.pi-2*sp.pi*(t1*t2-1))==0
    samples=[]
    for u in map(mp.mpf,['8','32','128','512','2048','8192']):
        y=yof(u);err=u-y-asym(y)
        assert abs(err)*y**3 < mp.mpf('.0004')
        samples.append(dict(u=n(u),y=n(y),du=n(u-y),remainder=n(err),e_est=n((u-y+A*mp.log(y)-C-D/y)*y*y)))
    # Independently fit log terms without imposing their vanishing.
    ys=[]; vals=[]
    for j in range(16):
        u=mp.mpf(1000)*mp.mpf('1.5')**j;y=yof(u)
        ys.append(y);vals.append(u-y+A*mp.log(y))
    basis=lambda y:[1,mp.log(y)/y,1/y,mp.log(y)**2/y**2,mp.log(y)/y**2,1/y**2]+[mp.log(y)**k/y**3 for k in range(4)]+[mp.log(y)**k/y**4 for k in range(5)]
    fit,res=mp.qr_solve(mp.matrix([basis(y) for y in ys]),mp.matrix(vals))
    targets=[C,0,D,0,0,E]
    assert max(abs(fit[i]-targets[i]) for i in range(6))<mp.mpf('2e-7')
    # Revert the claimed delta-u to get the necessary inverse coefficients.
    L=sp.Symbol('L',real=True);aa,cc,dd,ee=sp.symbols('a c d e')
    v0=aa*L-cc;v1=aa*v0-dd;v2=aa*v1-aa*v0**2/2+dd*v0-ee
    assert sp.expand(v2).coeff(L,2)==-aa**3/2
    wrong=[]
    for u in map(mp.mpf,['128','512','2048']):
        printed=(mp.log(8*mp.pi*u)-1)/(2*mp.pi)+(12*mp.log(8*mp.pi*u)-mp.pi**2)/(48*mp.pi**2*u)+(-12*mp.log(8*mp.pi*u)+6*mp.zeta(3)+mp.pi**2-8)/(96*mp.pi**3*u*u)
        diff=yof(u)-u-printed
        assert abs(diff)*u*u>mp.mpf('.01')
        wrong.append(dict(u=n(u),inverse_series_discrepancy=n(diff)))
    R['wh']=dict(symbol=rows,normalization_missing=n(mp.sqrt(mp.pi)),inverse_samples=samples,unconstrained_log_fit=[n(x) for x in fit[:6]],fit_residual=n(res),printed_y_of_u=wrong,reverted_v2=str(v2))
    print('WH symbol/normalization discrepancies reproduced; corrected-transform coefficients supported',flush=True);save()

class ExactEdge(FastNodesEdgeBA):
    def edge(self,s): return s*self.q
    def p_of_P(self,P,s):
        assert s==1
        f=lambda p:self.P_dr(p)-self.P_dr(self.q)-P
        hi=3*self.w
        while f(hi)<0: hi*=2
        return brentq(f,self.q,hi,xtol=1e-15,rtol=1e-15)

def finite():
    cache=HERE/'audit_phase2i_finite_cache.json'
    if cache.exists() and '--refresh' not in sys.argv: data=json.loads(cache.read_text())
    else:
        data=[]
        for h,mult in [(0.04,1),(0.02,1),(0.01,1),(0.005,1),(0.01,1.2)]:
            ts=time.perf_counter();N=int(mult*(4*np.pi*2*np.exp(-.5)/h+300))
            m=ExactEdge(1,h,N);b=m.b
            row=dict(h=h,N=N,mult=mult,du={},phase={},count_error=float(abs(np.dot(m.wq,m.rho)-1)))
            for y in [2,4,8,16,32,64]:
                k=np.sqrt(y*h/b);P=4*b*(k*np.cosh(k)-np.sinh(k))
                p=m.p_of_P(P,1)
                row['du'][str(y)]=float((p-m.q-2*b*(np.cosh(k)-1))/h)
            for u,v in [(4,1),(8,2),(8,4)]:
                row['phase'][f'{u},{v}']=float((m.Phi(m.q+h*u,m.q+h*v)-m.Phi(m.q+h*u,m.q))/h)
            row['seconds']=time.perf_counter()-ts;data.append(row)
            cache.write_text(json.dumps(data,indent=2)+'\n')
            print('finite-h',h,N,'seconds',row['seconds'],flush=True)
    base=[x for x in data if x['mult']==1];hs=np.array([x['h'] for x in base])
    rows=[]
    for y in [2,4,8,16,32,64]:
        vv=np.array([x['du'][str(y)] for x in base])
        estimates=[np.polynomial.polynomial.polyfit(hs,vv,3)[0],np.polynomial.polynomial.polyfit(hs[1:],vv[1:],2)[0]]
        logfit=np.linalg.solve(np.column_stack([np.ones(3),hs[1:],hs[1:]*np.log(hs[1:])]),vv[1:])[0]
        mesh=abs(data[2]['du'][str(y)]-data[4]['du'][str(y)])
        row=dict(y=y,cubic=float(estimates[0]),quadratic=float(estimates[1]),hlogh=float(logfit),power_spread=float(abs(estimates[0]-estimates[1])),mesh_difference=mesh,asymptotic=n(asym(mp.mpf(y))),e_estimate=float((estimates[0]+float(A)*np.log(y)-float(C)-float(D)/y)*y*y))
        rows.append(row)
    # Numerical convolution from corrected WH transform, independent of all Phase 2i code.
    conv=[]
    with mp.workdps(35):
        for u,v in [(4,1),(8,2),(8,4)]:
            values=[]
            for Nq in [24,48]:
                nodes,weights=roots_legendre(Nq)
                val=mp.mpf(0)
                for x,w in zip(nodes,weights):
                    z=mp.mpf(str(v*(x+1)/2));weight=mp.mpf(str(v*w/2))
                    val+=2*weight*il(u-z,2)*il(v-z,1)
                values.append(val)
            vv=np.array([row['phase'][f'{u},{v}'] for row in base])
            ext=np.polynomial.polynomial.polyfit(hs,vv,3)[0]
            ext2=np.polynomial.polynomial.polyfit(hs[1:],vv[1:],2)[0]
            quaderr=abs(values[1]-values[0]);discrep=abs(values[1]-ext)
            assert quaderr<mp.mpf('1e-12')
            # Error budget determined by two h extrapolations; no exact BA limit claimed.
            assert discrep<max(mp.mpf('3e-4'),3*abs(ext-ext2))
            conv.append(dict(u=u,v=v,Y=n(yof(u)),r=n(mp.sqrt(yof(v)/yof(u))),convolution=n(values[1]),quadrature_difference=n(quaderr),BA_cubic=float(ext),BA_quadratic=float(ext2),discrepancy=n(discrep)))
    R['finite']=dict(du=rows,convolution=conv,raw=data)
    print('finite-h and independent inverse-Laplace convolution checks complete',flush=True);save()

def c0(r): return -2*(mp.polylog(2,r)-mp.polylog(2,-r)-2*r)/mp.pi
def c1(r): return -r**3/(6*(1-r*r))-2*D*((1-r*r)/r**2*mp.log((1-r)/(1+r))+2/r)
def integrals_data():
    rows=[]
    for r in map(mp.mpf,['.2','.4','.5','.6','.8']):
        integral=4/mp.pi*mp.quad(lambda s:mp.log(s/r)*s*s/(1-s*s),[0,r])
        def fun(t):
            sig=1-r*r+t*t;delta=-mp.log(t*t/(r*r*sig))/(2*mp.pi)
            return 8/sig**mp.mpf('2.5')*(t*t*(-2*D-3/(4*mp.pi**2)-delta/(4*mp.pi)+3*delta*delta/8)-D*sig/2*(1+t*t*(1-1/(r*r)-1/sig)))
        cv=mp.quad(fun,[0,r])
        assert abs(integral-c0(r))<mp.mpf('1e-54') and abs(cv-c1(r))<mp.mpf('1e-54')
        rows.append(dict(r=n(r),c0=n(integral),c1=n(cv)))
    # Exact classical dispersion via hyperbolic cross ratio (avoids cancellation
    # in the Hirota polynomial used by the audited script).
    def classical(h,Y,r):
        h,Y,r=map(lambda x:mp.mpf(str(x)),(h,Y,r));b=mp.exp(-mp.mpf('.5'))
        k=mp.sqrt(Y*h/b);l=r*k
        la=2*mp.log(mp.sinh((k-l)/2)/mp.sinh((k+l)/2))
        return (2*b*(mp.cosh(k)-mp.cosh(l))*la+4*b*(l*mp.sinh(k)-k*mp.sinh(l)))/h
    data={x['h']:x for x in json.loads((HERE/'matching_toda_data.json').read_text())}
    data.update({x['h']:x for x in json.loads((HERE/'matching_toda_fine.json').read_text())})
    fits=[]
    specifications=[('cubic4',[.02,.01,.005,.0025],'poly3'),('quad3',[.01,.005,.0025],'poly2'),('quad4',[.02,.01,.005,.0025],'poly2'),('hlog3',[.01,.005,.0025],'log'),('hlog4',[.02,.01,.005,.0025],'log'),('cubic_coarse',[.04,.02,.01,.005],'poly3')]
    for r in [.4,.5,.6]:
        for name,hs,kind in specifications:
            if not all(h in data for h in hs):continue
            H=np.array(hs);edge=[]
            for Y in [4,8,16,32]:
                vv=[]
                for h in hs:
                    point=next(x for x in data[h]['same'] if x['Y']==Y and x['r']==r)
                    vv.append(float(mp.mpf(str(point['phase']))-classical(h,Y,r)))
                mat=np.column_stack([np.ones(len(H)),H,H*np.log(H)]) if kind=='log' else np.vander(H,int(kind[-1])+1,increasing=True)
                edge.append(float(np.linalg.lstsq(mat,vv,rcond=None)[0][0]))
            fixed=64*(edge[3]-float(c0(mp.mpf(str(r)))))-16*(edge[2]-float(c0(mp.mpf(str(r)))))
            free=32*(edge[2]-edge[3])
            # Large-Y fits estimate intercept and slope together, on two point sets.
            def large(indices,degree):
                X=1/np.array([4,8,16,32],float)[indices]
                return np.polynomial.polynomial.polyfit(X,np.array(edge)[indices],degree).tolist()
            fits.append(dict(r=r,basis=name,edge=edge,c1_fixed_c0=fixed,c1_free_two=free,large_free_3=large([1,2,3],2),large_free_4=large([0,1,2,3],2),target=float(c1(mp.mpf(str(r))))))
    for row in fits:
        if row['basis']=='cubic4':assert abs(row['c1_fixed_c0']-row['target'])<2e-4
    R['integrals']=rows;R['saved_fits']=fits
    print('integrals verified and saved data refitted across bases/point sets',flush=True);save()

W=(1,1,-2)
def angles(a):return [a+2*mp.pi/3,a-2*mp.pi/3,a]
def roots(a,e):return [(mp.tan(p/2) if e==1 else 1/mp.tan(p/2),c) for p,c in zip(angles(a),W)]
def droots(q,eta=1):
    g=mp.acos(-mp.sqrt(3)*q)
    return [(mp.tan((2*mp.pi/3-g)/2),-eta),(mp.tan((-2*mp.pi/3-g)/2),eta)]
def ker(z):
    sign=1 if mp.re(z)>0 else -1;v=sign*z
    if abs(v-1)<mp.mpf('1e-55'):return mp.mpf(0)
    return sign*((-(mp.polylog(2,v)-mp.polylog(2,-v))-mp.log(v)*mp.log((1-v)/(1+v)))/(2*mp.pi)+mp.pi/8)
def bil(r1,r2,K=ker):return sum(c*d*K(x/y) for x,c in r1 for y,d in r2)
def theta(q,a,e):
    return -sum(c*(mp.sign(q)*mp.atan(mp.sin(p)/abs(q))+e*mp.atan(mp.sin(p)*mp.cos(p)/(abs(q)*mp.sqrt(1+q*q)+q*q+mp.sin(p)**2))) for p,c in zip(angles(a),W))
def deriv(q,a,e):return sum(c*mp.sin(p)/(q*q+mp.sin(p)**2)*(1+e*mp.sign(q)*mp.cos(p)/mp.sqrt(1+q*q)) for p,c in zip(angles(a),W))
def whole(f):return mp.quad(lambda q:f(q)+f(-q),[0,mp.mpf('.1'),1,10,mp.inf])/(4*mp.pi)
def kp():
    a,b=mp.mpf('.3'),mp.mpf('.7');raw={};rows=[];errors=[]
    for e,f in itertools.product([1,-1],repeat=2):
        raw[e,f]=whole(lambda q:theta(q,a,e)*deriv(q,b,f))
        rev=whole(lambda q:theta(q,b,f)*deriv(q,a,e))
        k=mp.re(bil(roots(a,e),roots(b,f)))
        errors += [abs(raw[e,f]-(k-e*mp.pi/2)),abs((raw[e,f]-rev)/2-(k-(e-f)*mp.pi/4))]
        rows.append(dict(e=e,f=f,raw=n(raw[e,f]),kernel=n(k),offset_over_pi=n((raw[e,f]-k)/mp.pi)))
    sd=[]
    for q in map(mp.mpf,['.4','.1','-.45']):
        s=-2*mp.sqrt(3)*q
        for e,eta in itertools.product([1,-1],repeat=2):
            def dp(t):
                X=mp.sign(t)*mp.sqrt(1+t*t)/(2*mp.sqrt(3))-q;Z=-t/2
                xp=mp.sign(t)*t/(2*mp.sqrt(3)*mp.sqrt(1+t*t))
                return eta*2*(-X/2-xp*Z)/(X*X+Z*Z)
            direct=whole(lambda t:theta(t,a,e)*dp(t));k=mp.re(bil(roots(a,e),droots(q,eta)))
            winding=2*mp.pi*eta*(1 if s<-1 else -1 if s>1 else 0)
            errors.append(abs(direct-k-e*winding/4));sd.append(dict(qD=n(q),eps=e,eta=eta,raw=n(direct),offset_over_pi=n((direct-k)/mp.pi)))
    assert max(errors)<mp.mpf('1e-50')
    # Necessity is false: add constant to an admissible kernel. Both zero-sum
    # bilinears annihilate it, although oddness and inversion are destroyed.
    for rr,ss in [(roots(a,1),roots(b,-1)),(droots(mp.mpf('.2')),droots(mp.mpf('-.3')))]:
        assert abs(bil(rr,ss,lambda z:ker(z)+1)-bil(rr,ss))<mp.mpf('1e-55')
    assert abs((ker(mp.mpf('.3'))+1)+(ker(1/mp.mpf('.3'))+1)-2)<mp.mpf('1e-55')
    # Oddness is not needed even at kernel level: z^2-z^-2 is inversion odd.
    K=lambda z:z*z-1/(z*z)
    assert abs(bil(roots(a,1),roots(b,1),K)+bil(roots(b,1),roots(a,1),K))<mp.mpf('1e-52')
    # Analytic continuation of the ROOTS used for real solitons, not a shifted set.
    breath=[]
    for av,gv in [('.35','.15'),('.50','.20')]:
        al=mp.mpf(av)-mp.j*mp.mpf(gv);be=mp.conj(al)
        v=bil(roots(al,1),roots(be,1));dp=lambda x:-mp.sqrt(3)*mp.cos(x)/mp.pi*mp.log(-mp.tan((x+2*mp.pi/3)/2)/mp.tan((x-2*mp.pi/3)/2))
        pair=dp(al)+dp(be)
        breath.append(dict(a=av,g=gv,phase=n(v),momentum=n(pair)))
        assert abs(mp.re(v))<mp.mpf('1e-50') and abs(mp.im(pair))<mp.mpf('1e-50')
    # Delays from KP cross ratio; direct defect transmission ratio.
    cross=[]
    def sol(x,e):return mp.sin(x),e*mp.cos(x),(mp.sin(x)+e*mp.cos(x)/mp.sqrt(3))/2,(e*mp.cos(x)/mp.sqrt(3)-mp.sin(x))/2
    def la(a,e,b,f):
        k,v,pa,pb=sol(a,e);l,w,qa,qb=sol(b,f)
        return mp.log(abs((pa-qa)*(pb-qb)/((pa-qb)*(pb-qa))))
    def gam(x):return -3*mp.cos(2*x)/(mp.sin(x)*mp.sin(3*x))
    delay_rows=[]
    for e,f in itertools.product([1,-1],repeat=2):
        k,v,pa,pb=sol(a,e);l,w,qa,qb=sol(b,f)
        def delay(q,factor):
            z=(pb-q)/(pa-q)
            return -e*mp.log(abs(z))/k-2*factor*e*gam(a)*q
        lhs=-la(a,e,b,f)/k+gam(a)*2*l
        rhs=e*(delay(qb,mp.sqrt(3))-delay(qa,mp.sqrt(3)))
        corrected=e*(delay(qb,1)-delay(qa,1))
        expected=2*l*gam(a)*(mp.sqrt(3)-1)
        assert abs(rhs-lhs-expected)<mp.mpf('1e-50') and abs(rhs-lhs)>1
        assert abs(corrected-lhs)<mp.mpf('1e-50')
        delay_rows.append(dict(e=e,f=f,printed_discrepancy=n(rhs-lhs),corrected_error=n(corrected-lhs)))
    for x in [b+mp.pi,mp.pi-b]:
        val=mp.re(bil(roots(a,1),roots(x,1)));base=mp.re(bil(roots(a,1),roots(b,-1)))
        target=-base if x==b+mp.pi else base
        assert abs(val-target)<mp.mpf('1e-50');cross.append(n(val))
    def momentum(x,e,kind):
        if kind=='can':return -e*mp.sqrt(3)*mp.cos(x)/mp.pi*mp.log(abs(mp.tan((x+2*mp.pi/3)/2)/mp.tan((x-2*mp.pi/3)/2)))
        return -e*sum(c*mp.sin(p)*mp.log(abs(mp.tan(p/2))) for p,c in zip(angles(x),W))/(2*mp.pi)
    def renorm(x,e,y,f,kind):
        loga=la(x,e,y,f)
        dx=-loga/mp.sin(x)+2*gam(x)*mp.sin(y)
        dy=loga/mp.sin(y)-2*gam(y)*mp.sin(x)
        return dx*momentum(x,e,kind)+dy*momentum(y,f,kind)
    for kind in ['can','mode']:
        base=renorm(a,1,b,-1,kind)
        assert abs(renorm(a,1,b+mp.pi,1,kind)+base)<mp.mpf('1e-50')
        assert abs(renorm(a,1,mp.pi-b,1,kind)-base)<mp.mpf('1e-50')
    # Branch-point cancellation and already-audited profile correction.
    for eps in ['1e-8','1e-14']:
        z=1+mp.mpf(eps);assert abs(2*mp.log(z)/(1-z*z)+1)<2*mp.mpf(eps)
    x,k=sp.symbols('x k',positive=True)
    assert sp.cancel((2*sp.diff(sp.log(1+sp.exp(k*x)),x,2)-k*k/(2*sp.cosh(k*x/2)**2)).rewrite(sp.exp))==0
    R['kp']=dict(SS=rows,SD=sd,max_quadrature_error=n(max(errors)),breather=breath,crossing=cross,Darboux=delay_rows,iff_counterexample='Ktilde+1 has identical F and ISS but violates both kernel conditions; z^2-z^-2 needs no oddness')
    print('raw integrals, orientations, crossing and continuation verified; iff and printed Darboux coefficient refuted',flush=True);save()

def determinant():
    # Independently linearize the nonlinear sewing equations, using automatic
    # differentiation; no import of Phase 2i or prior determinant checker.
    r=mp.sqrt(3)
    def sewing(jet,s):
        U,UX,UXX,UXXX,UT,UXT,V,VX,VXX,VXXX,VT,VXT=jet
        al=1/r;be=2/r;mu=1/r;sig=s+mu*(U-V)
        rho=-(s/(2*mu))*(1-s*s/3)+(sig-sig**3/3)/(2*mu)
        H=mu*(be*UX**2+2*al*UX*VX+be*VX**2)/2+(1-sig**2)*(UX+VX)/2-be*sig*rho/2
        return [-UX-UX**2+UXXX-sig*VT+H-al*UXT-be*VXT,-UXX+al*UT+be*VT+sig*(be*UX+al*VX)+rho,VX+VX**2-VXXX+sig*UT-H-be*UXT-al*VXT,VXX+be*UT+al*VT+sig*(al*UX+be*VX)+rho]
    rows=[]
    for s,g in itertools.product(map(mp.mpf,['-1.5','.5','1.9']),map(mp.mpf,['.2','.7'])):
        disc=mp.sqrt(1-4*g*g);ks=[mp.sqrt((1+disc)/2),mp.sqrt((1-disc)/2)]
        cols=[]
        for side in [0,1]:
            for k in ks:
                k=(-1)**side*k;j=[1,k,k*k,k**3,g,g*k];j=j+[0]*6 if side==0 else [0]*6+j
                cols.append([mp.diff(lambda t:sewing([t*v for v in j],s)[i],0) for i in range(4)])
        mat=mp.matrix([[cols[j][i] for j in range(4)] for i in range(4)])
        actual=mp.det(mat);S=sum(ks)
        pm=S*S-(r-1)*s*S/2+(s*s-2-r)/2;pp=S*S-(r+1)*s*S/2+(s*s-2+r)/2
        pred=-mp.mpf(2)/3*S*S*(S*S-1)*pm*pp*(ks[0]-ks[1])**2
        assert abs(actual-pred)<mp.mpf('1e-50');rows.append(n(actual/pred))
    growth=[]
    for s in map(mp.mpf,['1.2','1.5','1.9','1.99']):
        disc=mp.sqrt(4-s*s)
        roots0=[((r-1)*s+sgn*(r+1)*disc)/4 for sgn in [-1,1]]+[((r+1)*s+sgn*(r-1)*disc)/4 for sgn in [-1,1]]
        rates=sorted([(S*S-1)/2 for S in roots0 if S>1])
        targets=sorted([(s*disc+r*(s*s-2))/8,abs(s*disc-r*(s*s-2))/8])
        assert len(rates)==2 and max(abs(x-y) for x,y in zip(rates,targets))<mp.mpf('1e-55')
        growth.append(dict(s=n(s),rates=[n(x) for x in rates]))
    # Printed P_+(1;s) has the wrong s factor/sign; verify independently.
    s=sp.symbols('s');Pp=1-(sp.sqrt(3)+1)*s/2+(s*s-2+sp.sqrt(3))/2
    assert sp.simplify(Pp-(s-1)*(s-sp.sqrt(3))/2)==0
    R['defect']=dict(off_shell_ratios=rows,growth=growth,threshold='g2(sqrt(3))=0: marginal, not two strictly positive rates')
    print('independent sewing determinant and admissible branches verified',flush=True);save()

def main():
    start=time.perf_counter()
    if '--resume' in sys.argv and OUT.exists():R.update(json.loads(OUT.read_text()))
    inputs=[HERE.parent/'toda_wh_and_kp_synthesis.tex',HERE/'toda_wh_and_kp_synthesis.py',HERE/'matching_toda_data.json',HERE/'matching_toda_fine.json',HERE/'toda_edge_ba.py']
    hashes={str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    for fn in [wh,finite,integrals_data,kp,determinant]:
        if '--resume' not in sys.argv or fn.__name__ not in ['finite','integrals_data']:
            section_start=time.perf_counter();fn()
            R.setdefault('section_seconds',{})[fn.__name__]=time.perf_counter()-section_start
    assert hashes=={str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    R['input_sha256']=hashes;R['precision']=60;R['seconds']=time.perf_counter()-start
    R['original_script_log']='audit_phase2i_original_run.txt'
    R['original_script_seconds']=101.87684258798254
    R['fresh_dense_solver_seconds']=sum(x['seconds'] for x in R['finite']['raw'])
    R['prior_claim_failures']=['Printed Fourier symbol','Missing sqrt(pi) in momentum Laplace transform',
      'Wrong b factor in physical momentum scale','Printed inverse series missing log(u)^2/u^2',
      'Double Laplace transform inconsistent by pi after normalization repair',
      'Kernel iff criterion has constant null-kernel counterexample',
      'Printed Darboux delay has extra sqrt(3)',
      'Four-term endpoint product violates edge boundary condition',
      'Subtraction does not remove saved-fit sign reversals','All six audits resolved is false']
    R['development_failures']=['Inherited fixed momentum bracket failed at large y; expanded bracket, unchanged root equation and tolerances.',
      'First direct Darboux test failed; corrected independently transcribed inverse transmission ratio, then isolated and asserted the remaining printed sqrt(3) discrepancy.']
    save();print('runtime',R['seconds']);print('ALL CHECKS PASSED')
if __name__=='__main__':main()

"""Reproducible synthetic evaluation for the FRUCT 40 manuscript.

Primary experiment: 30 paired independent trials, 100,000 synthetic profiles/trial,
33,334 sequential events/trial (1,000,020 events per method across trials).
All methods in a trial share the same event stream and latent outcome uniforms.
"""
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import t, wilcoxon

BASE_ARMS=np.array([0.0,2.0,5.0,8.0],dtype=float)
BASE_CMAX=8.0

def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))

def make_env(seed:int,n_users:int=100_000,T:int=33_334):
    rng=np.random.default_rng(seed)
    spend_raw=rng.lognormal(3.4,.65,n_users)
    spend=np.clip((np.log1p(spend_raw)-2.5)/2.5,0,1)
    visits=np.clip(rng.negative_binomial(3,.45,n_users)/12,0,1)
    affinity=rng.beta(2.4,2.0,n_users)
    price_noise=rng.beta(2.0,2.2,n_users)
    price=np.clip(.72*(1-spend)+.28*price_noise,0,1)
    loyalty=np.clip(.55*spend+.45*rng.beta(2.3,1.9,n_users),0,1)
    latent=rng.normal(0,.35,n_users)
    basket=np.clip(rng.lognormal(np.log(48),.38,n_users),12,160)

    uid=rng.integers(0,n_users,T)
    near=rng.random(T)<.68
    d=np.empty(T)
    d[near]=rng.beta(1.2,4.5,near.sum())
    d[~near]=rng.beta(3.2,1.7,(~near).sum())
    hour_p=np.array([.04,.05,.07,.11,.13,.08,.07,.06,.07,.09,.10,.08,.05])
    hours=rng.choice(np.arange(9,22),T,p=hour_p)
    off=np.isin(hours,[9,10,14,15,20,21]).astype(float)
    dwell=sigmoid(-.5+.9*loyalty[uid]+.8*affinity[uid]-.9*d+rng.normal(0,.6,T))
    return {
        'spend':spend[uid], 'visits':visits[uid], 'affinity':affinity[uid],
        'price':price[uid], 'loyalty':loyalty[uid], 'latent':latent[uid],
        'basket':basket[uid], 'd':d, 'offpeak':off, 'dwell':dwell,
        'competitor':(rng.random(T)<.08).astype(float),
        'weather':(rng.random(T)<.10).astype(float),
        'noise':rng.normal(0,.18,T), 'u':rng.random(T),
        'basket_noise':rng.lognormal(0,.12,T)
    }

def base_context(env,t,lam=2.0,spatial=True,temporal=True):
    prox=np.exp(-lam*env['d'][t]) if spatial else 0.0
    off=env['offpeak'][t] if temporal else 0.0
    spend=env['spend'][t]; aff=env['affinity'][t]
    # Interactions are observable feature engineering, not latent variables.
    return np.array([1.,spend,env['visits'][t],aff,prox,off,env['dwell'][t],(1-spend)*prox,aff*off],dtype=float)

def phi(g,a,scale_max):
    q=a/scale_max if scale_max>0 else 0.0
    return np.concatenate([g,q*g,np.array([q*q])])

def outcome(env,t,a):
    # Nonlinear data-generating process. Learner never observes price sensitivity,
    # loyalty, competitor/weather shocks, latent response, or noise.
    s=env['spend'][t]; v=env['visits'][t]; aff=env['affinity'][t]
    d=env['d'][t]; off=env['offpeak'][t]
    z0=(-2.85+.55*s+.40*v+.72*aff+.50*env['loyalty'][t]+.45*env['dwell'][t]
        -1.10*d**1.35-.22*off+.32*aff*env['loyalty'][t]
        -.55*env['competitor'][t]-.35*env['weather'][t]+env['latent'][t]+env['noise'][t])
    q=a/BASE_CMAX
    z=z0+q*(4.20*env['price'][t]*((1-d)**2)+1.20*env['price'][t]*off+.40*aff-.70*env['loyalty'][t])-.50*q*q
    p0=float(sigmoid(z0)); p=float(sigmoid(z))
    y0=float(env['u'][t]<p0); y=float(env['u'][t]<p)
    bv=float(env['basket'][t]*env['basket_noise'][t])
    basket=bv if y else 0.0
    cost=a*y
    inc=y-y0
    return y,basket,cost,p,inc,inc*bv

class SharedLinUCB:
    def __init__(self,arms,alpha=.35,budget=2800.,dual=False,eta=.05,strict=True):
        self.arms=np.asarray(arms,dtype=float); self.max_arm=float(self.arms.max())
        self.alpha=alpha; self.B0=float(budget); self.B=float(budget)
        self.dual=dual; self.eta=eta; self.mu=0.; self.strict=strict
        d=19; self.Ainv=np.eye(d); self.b=np.zeros(d); self.theta=np.zeros(d)
    def act(self,g):
        scores=[]
        for a in self.arms:
            if self.strict and a>self.B+1e-12:
                scores.append(-1e12); continue
            f=phi(g,a,self.max_arm)
            Af=self.Ainv@f
            ucb=self.theta@f+self.alpha*math.sqrt(max(float(f@Af),0.0))
            penalty=self.mu*(a/self.max_arm) if self.dual and self.max_arm>0 else 0.0
            scores.append(ucb-penalty)
        return int(np.argmax(scores))
    def update(self,f,r,c,T):
        Af=self.Ainv@f
        self.Ainv-=np.outer(Af,Af)/(1.0+float(f@Af))
        self.b+=r*f; self.theta=self.Ainv@self.b
        self.B-=c
        if self.dual:
            target=self.B0/T
            self.mu=max(0.0,self.mu+self.eta*(c-target)/max(self.max_arm,1e-9))

def run_method(env,method,budget=2800.,lam=2.0,alpha=.35,eta=.05,arm_scale=1.0):
    arms=BASE_ARMS*arm_scale
    T=len(env['d'])
    conv=sales=cost=inc_conv=inc_sales=0.0; offers=0; arm_sum=0.0
    if method not in ('static','geo'):
        dual=method in ('budget_linucb','proposed','no_spatial','no_temporal')
        spatial=method not in ('budget_linucb','no_spatial')
        temporal=method not in ('budget_linucb','no_temporal')
        strict=(method!='no_budget')
        model=SharedLinUCB(arms,alpha,budget,dual,eta,strict)
    for ti in range(T):
        if method=='static':
            # Fixed loyalty-tier policy: eligibility and amount depend only on historical spend tier.
            s=env['spend'][ti]
            a=(5.0*arm_scale if s>.72 else (2.0*arm_scale if s>.45 else 0.0))
            if a>budget-cost: a=0.0
        elif method=='geo':
            # Uniform geofence policy: fixed micro-incentive whenever within 125 m of a 500 m evaluation radius.
            a=5.0*arm_scale if env['d'][ti]<.25 else 0.0
            if a>budget-cost: a=0.0
        else:
            g=base_context(env,ti,lam,model is not None and method not in ('budget_linucb','no_spatial'),model is not None and method not in ('budget_linucb','no_temporal'))
            ai=model.act(g); a=arms[ai]
        y,b,c,p,ic,isales=outcome(env,ti,a)
        conv+=y; sales+=b; cost+=c; inc_conv+=ic; inc_sales+=isales; offers+=(a>0); arm_sum+=a
        if method not in ('static','geo'):
            model.update(phi(g,a,model.max_arm),y,c,T)
    return {
        'cr':conv/T,'icr':inc_conv/T,'pse':inc_sales/cost if cost>0 else np.nan,
        'gross_pse':sales/cost if cost>0 else np.nan,'cost':cost,'sales':sales,'inc_sales':inc_sales,
        'offer_rate':offers/T,'avg_arm':arm_sum/T,'budget_util':cost/budget,
        'overspend':max(0.0,cost-budget)
    }

PRIMARY_METHODS=['static','geo','linucb','budget_linucb','proposed','no_spatial','no_temporal','no_budget']

def one_trial(seed,T=33_334,budget=2800.):
    env=make_env(seed,T=T)
    rows=[]
    for m in PRIMARY_METHODS:
        r=run_method(env,m,budget=budget)
        rows.append({'seed':seed,'method':m,**r})
    return rows

def ci95(x):
    x=np.asarray(x,float); n=len(x); mean=x.mean(); se=x.std(ddof=1)/math.sqrt(n)
    half=t.ppf(.975,n-1)*se
    return mean,mean-half,mean+half

def primary_experiment(outdir):
    seeds=list(range(1001,1031))
    rows=[]
    workers=min(8,os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(one_trial,s):s for s in seeds}
        for fut in as_completed(futs): rows.extend(fut.result())
    df=pd.DataFrame(rows).sort_values(['seed','method'])
    df.to_csv(os.path.join(outdir,'primary_trials.csv'),index=False)
    summary=[]
    for m,g in df.groupby('method'):
        rec={'method':m}
        for metric in ['cr','icr','pse','cost','budget_util','offer_rate','overspend']:
            mean,lo,hi=ci95(g[metric].values)
            rec.update({metric:mean,metric+'_lo':lo,metric+'_hi':hi})
        summary.append(rec)
    sdf=pd.DataFrame(summary)
    sdf.to_csv(os.path.join(outdir,'primary_summary.csv'),index=False)
    # Paired Wilcoxon tests against key baselines.
    tests=[]
    p=df[df.method=='proposed'].set_index('seed')
    for base in ['geo','linucb','budget_linucb','no_spatial','no_temporal']:
        b=df[df.method==base].set_index('seed')
        for metric in ['cr','icr','pse']:
            stat,pv=wilcoxon(p[metric],b[metric],alternative='greater')
            diff=(p[metric]-b[metric]).mean()
            tests.append({'comparison':'proposed>'+base,'metric':metric,'mean_diff':diff,'W':stat,'p_value':pv})
    pd.DataFrame(tests).to_csv(os.path.join(outdir,'paired_tests.csv'),index=False)
    return df,sdf,pd.DataFrame(tests)

def sensitivity_experiment(outdir):
    seeds=list(range(2001,2011)); T=10_000; budget=840.0 # same ~0.084 USD/event as primary
    rows=[]
    for lam in [.5,1.,2.,3.,4.]:
        for seed in seeds:
            env=make_env(seed,T=T); r=run_method(env,'proposed',budget=budget,lam=lam)
            rows.append({'type':'lambda','x':lam,'seed':seed,**r})
    for mult in [.5,.75,1.,1.25,1.5]:
        for seed in seeds:
            env=make_env(seed,T=T); r=run_method(env,'proposed',budget=budget*mult,lam=2.)
            rows.append({'type':'budget','x':mult,'seed':seed,**r})
    for scale in [.5,.75,1.,1.25,1.5]:
        for seed in seeds:
            env=make_env(seed,T=T); r=run_method(env,'proposed',budget=budget,lam=2.,arm_scale=scale)
            rows.append({'type':'reward','x':scale,'seed':seed,**r})
    df=pd.DataFrame(rows); df.to_csv(os.path.join(outdir,'sensitivity_trials.csv'),index=False)
    sums=[]
    for (typ,x),g in df.groupby(['type','x']):
        for metric in ['icr','pse','cr']:
            mean,lo,hi=ci95(g[metric]); sums.append({'type':typ,'x':x,'metric':metric,'mean':mean,'lo':lo,'hi':hi})
    s=pd.DataFrame(sums); s.to_csv(os.path.join(outdir,'sensitivity_summary.csv'),index=False)
    return df,s

if __name__=='__main__':
    outdir='/mnt/data/fruct40_results'; os.makedirs(outdir,exist_ok=True)
    df,sdf,tests=primary_experiment(outdir)
    print('\nPRIMARY SUMMARY')
    print(sdf[['method','cr','icr','pse','cost','budget_util','offer_rate','overspend']].sort_values('cr',ascending=False).to_string(index=False))
    print('\nPAIRED TESTS')
    print(tests.to_string(index=False))
    _,sens=sensitivity_experiment(outdir)
    print('\nSENSITIVITY')
    print(sens.to_string(index=False))

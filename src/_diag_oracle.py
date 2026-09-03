"""Diagnostic: how much of the gap is model vs optimiser?
Fits with (a) everything free, (b) impression styles fixed to truth,
(c) styles + trajectory fixed to truth (h0 only)."""
import sys; sys.path.insert(0,'src')
import numpy as np, torch, torch.nn.functional as F, time
import synth as S, fuse as Fz, evaluate as E

CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
dt=[(y-CARVE)/100 for y in years]
m=d['masks']; px=d['px_mm']

def report(tag, h0):
    s,t=E.best_threshold(h0,m,h0.min()+1e-3,np.quantile(h0,0.999),40)
    c=np.mean([np.corrcoef(h0[i].ravel(),d['h0'][i].ravel())[0,1] for i in range(len(m))])
    print(f'{tag:22s} IoU={s:.3f}@{t:.2f}  corr={c:.3f}',flush=True)
    return s

def fix_styles(model, styles):
    with torch.no_grad():
        inv=lambda v,lo,hi: torch.log(torch.tensor((v-lo)/(hi-v)))
        model.a_lam.copy_(torch.stack([inv(s.rho,0.15,1.30) for s in styles]))
        model.a_eps.copy_(torch.stack([inv(s.eps,0.03,0.30) for s in styles]))
        model.a_s.copy_(torch.stack([inv(s.s,0.015,0.15) for s in styles]))
        model.a_alpha.copy_(torch.stack([inv(s.alpha,0.35,0.99) for s in styles]))
    for p in (model.a_lam,model.a_eps,model.a_s,model.a_alpha):
        p.requires_grad_(False)

t0=time.time()
r=Fz.fit(d['images'],px,sizes=(128,256),iters=(600,900),dt=dt,verbose=False)
report('free',r['h0']); print('  a=%.4f b=%.4f (gt 0.075/0.115)'%(r['a_rate'],r['b_rate']))

# oracle styles
import types
orig=Fz.Fusion
class OracleFusion(orig):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); fix_styles(self, d['styles'])
Fz.Fusion=OracleFusion
r2=Fz.fit(d['images'],px,sizes=(128,256),iters=(600,900),dt=dt,verbose=False)
report('oracle styles',r2['h0']); print('  a=%.4f b=%.4f'%(r2['a_rate'],r2['b_rate']))
Fz.Fusion=orig

# baselines
b1=E.best_threshold(d['images'][:,0],m,0.3,0.95,40); print(f'earliest             IoU={b1[0]:.3f}@{b1[1]:.2f}')
mx=d['images'].max(1); b2=E.best_threshold(mx,m,0.3,0.99,40); print(f'stack-max            IoU={b2[0]:.3f}@{b2[1]:.2f}')
rl=np.stack([E.rl_deconv(d['images'][i,0],0.313/px,30) for i in range(len(m))])
b3=E.best_threshold(rl,m,rl.min()+1e-3,np.quantile(rl,0.999),40); print(f'RL deconv(earliest)  IoU={b3[0]:.3f}@{b3[1]:.2f}')
print('%.0fs'%(time.time()-t0))
np.savez_compressed('results/_diag_oracle.npz',free=r['h0'],oracle=r2['h0'],gt=d['h0'],masks=m,images=d['images'])

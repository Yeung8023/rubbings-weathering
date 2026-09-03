import sys; sys.path.insert(0,'src')
import numpy as np, torch, time
import synth as S, fuse as Fz, evaluate as E

CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
dt=[(y-CARVE)/100 for y in years]; m=d['masks']; px=d['px_mm']

def rep(tag,f,lo=None,hi=None):
    lo = f.min()+1e-3 if lo is None else lo
    hi = np.quantile(f,0.999) if hi is None else hi
    s,t=E.best_threshold(f,m,lo,hi,40)
    print(f'{tag:26s} IoU={s:.3f} @{t:.3f}',flush=True); return s

t0=time.time()
r=Fz.fit(d['images'],px,sizes=(128,256),iters=(700,1100),dt=dt,verbose=False,
         relief='levelset',w_perim=4e-3)
print('%.0fs  a=%.4f (gt .075) b=%.4f (gt .115)'%(time.time()-t0,r['a_rate'],r['b_rate']))
print('lam gt',np.round([s.rho for s in d['styles']],2),'est',np.round(r['lam'],2))
rep('fusion levelset h0',r['h0'])
rep('fusion levelset fg',r['fg'],0.05,0.95)
rep('earliest',d['images'][:,0],0.3,0.95)
rl=np.stack([E.rl_deconv(d['images'][i,0],0.313/px,30) for i in range(len(m))])
rep('RL deconv earliest',rl)
np.savez_compressed('results/_ls.npz',h0=r['h0'],fg=r['fg'],gt=d['h0'],masks=m,images=d['images'],recon=r['recon'])

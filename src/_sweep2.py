import sys; sys.path.insert(0,'src')
import numpy as np, time
import synth as S, fuse as Fz, evaluate as E
CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
px=d['px_mm']; m=d['masks']; dt=[(y-CARVE)/100 for y in years]
def rep(tag,f,lo=None,hi=None):
    lo=float(f.min())+1e-3 if lo is None else lo; hi=float(np.quantile(f,0.999)) if hi is None else hi
    s,t=E.best_threshold(f,m,lo,hi,40); print(f'{tag:28s} IoU={s:.3f}',flush=True); return s
rep('earliest',d['images'][:,0],0.3,0.95)
rep('stack mean',d['images'].mean(1),0.3,0.95)
rep('stack median',np.median(d['images'],1),0.3,0.95)
rl=np.stack([E.rl_deconv(d['images'][i,0],d['epochs'][0].sigma_mm/px,30) for i in range(len(m))])
rep('RL oracle-sigma (earliest)',rl)
for hc,wdw in [(0.15,0.0),(0.15,0.5),(None,0.0)]:
    t=time.time()
    r=Fz.fit(d['images'],px,dt=dt,sizes=(128,256),iters=(700,1100),verbose=False,
             relief='free',w_dw=wdw,huber_c=hc,spall_stride=8)
    cor=np.mean([np.corrcoef(r['h0'][i].ravel(),d['h0'][i].ravel())[0,1] for i in range(len(m))])
    s=rep(f'fusion huber={hc} dw={wdw}',r['h0'])
    print(f'    corr={cor:.3f} a={r["a_rate"]:.4f}(gt .075) b={r["b_rate"]:.4f}(gt .115) {time.time()-t:.0f}s',flush=True)
    np.savez_compressed(f'results/_sw2_{hc}_{wdw}.npz',h0=r['h0'],gt=d['h0'],masks=m,images=d['images'],recon=r['recon'])

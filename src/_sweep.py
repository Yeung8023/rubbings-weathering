import sys; sys.path.insert(0,'src')
import numpy as np, itertools, time
import synth as S, fuse as Fz, evaluate as E
CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
px=d['px_mm']; m=d['masks']; dt=[(y-CARVE)/100 for y in years]
b1=E.best_threshold(d['images'][:,0],m,0.3,0.95,40); print('baseline earliest IoU=%.3f'%b1[0])
rl=np.stack([E.rl_deconv(d['images'][i,0],d['epochs'][0].sigma_mm/px,30) for i in range(len(m))])
b2=E.best_threshold(rl,m,rl.min()+1e-3,np.quantile(rl,0.999),40); print('RL oracle-sigma  IoU=%.3f'%b2[0])
for stride,w_dw,w_sp in [(8,0.0,2e-3),(8,0.5,2e-3),(8,0.5,2e-2),(16,0.5,2e-3),(8,2.0,2e-3)]:
    t=time.time()
    r=Fz.fit(d['images'],px,dt=dt,sizes=(128,256),iters=(700,1100),verbose=False,
             relief='free',w_dw=w_dw,w_spall=w_sp,spall_stride=stride)
    s,th=E.best_threshold(r['h0'],m,1e-3,float(np.quantile(r['h0'],0.999)),40)
    cor=np.mean([np.corrcoef(r['h0'][i].ravel(),d['h0'][i].ravel())[0,1] for i in range(len(m))])
    print(f'stride={stride} w_dw={w_dw} w_spall={w_sp}: IoU={s:.3f} corr={cor:.3f} '
          f'a={r["a_rate"]:.4f} b={r["b_rate"]:.4f} lam={np.round(r["lam"],2)} ({time.time()-t:.0f}s)',flush=True)

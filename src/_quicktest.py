import sys; sys.path.insert(0,'src')
import numpy as np, torch, synth as S, fuse as Fz, time
CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀')[:24]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
print('GT:'); [print('  ',e) for e in d['epochs']]
dt=[(y-CARVE)/100 for y in years]
t=time.time(); r=Fz.fit(d['images'],d['px_mm'],sizes=(128,256),iters=(700,1100),dt=dt); print('%.0fs'%(time.time()-t))
print('a_rate %.4f (gt 0.075)  b_rate %.4f (gt 0.115)'%(r['a_rate'],r['b_rate']))
print('lam   gt',np.round([s.rho for s in d['styles']],2),'est',np.round(r['lam'],2))
print('alpha gt',np.round([s.alpha for s in d['styles']],2),'est',np.round(r['alpha'],2))
def iou(a,b): return float((a&b).sum()/((a|b).sum()+1e-9))
m=d['masks']
best=max([(np.mean([iou(r['h0'][i]>th,m[i]) for i in range(len(m))]),th) for th in np.arange(0.05,1.3,0.05)])
print('IoU fused %.3f @%.2f'%best)
b1=max([(np.mean([iou(d['images'][i,0]>th,m[i]) for i in range(len(m))]),th) for th in np.arange(0.3,0.95,0.02)])
print('IoU earliest %.3f @%.2f'%b1)
mean=d['images'].mean(1)
b2=max([(np.mean([iou(mean[i]>th,m[i]) for i in range(len(m))]),th) for th in np.arange(0.3,0.95,0.02)])
print('IoU mean-stack %.3f @%.2f'%b2)
print('corr h0', np.round(np.mean([np.corrcoef(r['h0'][i].ravel(),d['h0'][i].ravel())[0,1] for i in range(len(m))]),3))
np.savez_compressed('results/_probe4.npz',**{k:v for k,v in r.items() if k!='model'},gt_h0=d['h0'],masks=d['masks'],images=d['images'])

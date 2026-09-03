import sys; sys.path.insert(0,'src')
import numpy as np, time
import synth as S, fuse as Fz, evaluate as E
CARVE=632; YEARS=[1050,1350,1600,1800,2000]
TEXT="九成宮醴泉銘祕書監檢校侍中鉅鹿郡公臣魏徵奉勅撰維貞觀六年孟夏之月皇帝避暑乎九成之宮"
CH=list(dict.fromkeys(TEXT))[:12]
cfgs={'clean':dict(acq=dict(texture=0.0,stain=0.0,noise=0.004),warp=False,sheet_damage=False),
      'full':dict(acq=dict(texture=0.035,stain=0.05,noise=0.010),warp=True,sheet_damage=True)}
for cname,cfg in cfgs.items():
    d=S.make_corpus(CH,YEARS,seed=200,size_px=256,carve_year=CARVE,**cfg)
    dt=[(y-CARVE)/100 for y in YEARS]; m=d['masks']
    b0,_=E.best_threshold(d['images'][:,0],m,0.3,0.95,40)
    print(f'--- {cname}: baseline earliest IoU={b0:.3f}; true lam='
          f'{np.round([s.rho for s in d["styles"]],2)}',flush=True)
    for ws in [0.0,0.05,0.3,1.0]:
        t=time.time()
        r=Fz.fit(d['images'],d['px_mm'],dt=dt,sizes=(96,256),iters=(600,1000),verbose=False,
                 relief='free',huber_c=0.15,w_spall=2e-2,spall_stride=8,spall_model='levelset',w_style=ws)
        iou,_=E.best_threshold(r['h0'],m,1e-3,float(np.quantile(r['h0'],0.999)),40)
        print(f'{cname:5s} w_style={ws:4.2f}: a={r["a_rate"]:.4f}(.075) b={r["b_rate"]:.4f}(.115) '
              f'IoU={iou:.3f} lam={np.round(r["lam"],2)} ({time.time()-t:.0f}s)',flush=True)

"""Start the optimiser AT the truth. If it stays, the objective is right and
the search is the problem; if it drifts away to a lower loss, the objective
still has slack."""
import sys; sys.path.insert(0,'src')
import numpy as np, torch, torch.nn.functional as F
import synth as S, fuse as Fz, evaluate as E

CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
px=d['px_mm']; m=d['masks']; dt=[(y-CARVE)/100 for y in years]
Y=torch.tensor(d['images'],device='cuda'); C,n=Y.shape[:2]

def build(true_init):
    mdl=Fz.Fusion(C,n,256,256,px,dt=dt,relief='free',spall=True,spall_stride=8).cuda()
    with torch.no_grad():
        if true_init:
            h0=torch.tensor(d['h0'],device='cuda')[:,None]
            mdl.a_h0.copy_(torch.log(torch.expm1(h0.clamp_min(1e-4))))
            inv=lambda v,lo,hi: torch.log(torch.tensor(float((v-lo)/(hi-v)),device='cuda'))
            for i,st in enumerate(d['styles']):
                mdl.a_lam[i]=inv(st.rho,0.15,1.30); mdl.a_eps[i]=inv(st.eps,0.03,0.30)
                mdl.a_s[i]=inv(st.s,0.015,0.15); mdl.a_alpha[i]=inv(st.alpha,0.35,0.99)
            mdl.a_rate.copy_(torch.log(torch.expm1(torch.tensor(0.075))))
            mdl.b_rate.copy_(torch.log(torch.expm1(torch.tensor(0.115))))
    return mdl

def loss_of(mdl):
    with torch.no_grad():
        tot=0.
        for i in range(0,C,4):
            ci=torch.arange(i,min(i+4,C),device='cuda')
            yh,_=mdl(ci); tot+=float(Fz.robust(yh-Y[ci],0.15))*len(ci)/C
        return tot

mtrue=build(True); print('robust loss @ TRUE (spall/warp zero): %.6f'%loss_of(mtrue),flush=True)
def iou_of(h0):
    s,t=E.best_threshold(h0,m,1e-3,float(np.quantile(h0,0.999)),40); return s
print('IoU of TRUE h0: %.3f'%iou_of(d['h0']))

# refine only spall+warp+tone from the truth
for p in (mtrue.a_h0,mtrue.a_lam,mtrue.a_eps,mtrue.a_s,mtrue.a_alpha,mtrue.a_rate,mtrue.b_rate):
    p.requires_grad_(False)
opt=torch.optim.Adam([mtrue.d_spall,mtrue.wctrl,mtrue.b],lr=0.05)
for k in range(400):
    opt.zero_grad(); tot=0.
    for i in range(0,C,4):
        ci=torch.arange(i,min(i+4,C),device='cuda')
        yh,_=mtrue(ci); l=Fz.robust(yh-Y[ci],0.15)+2e-3*F.softplus(mtrue.d_spall[ci]).mean()
        (l*len(ci)/C).backward(); tot+=float(l)*len(ci)/C
    opt.step()
print('robust loss @ TRUE after fitting spall/warp: %.6f'%tot,flush=True)

r=Fz.fit(d['images'],px,dt=dt,sizes=(128,256),iters=(700,1100),verbose=False,relief='free',huber_c=0.15)
print('IoU of FITTED h0: %.3f   a=%.4f b=%.4f'%(iou_of(r['h0']),r['a_rate'],r['b_rate']))
mf=r['model']
print('robust loss @ FITTED: %.6f'%loss_of(mf))

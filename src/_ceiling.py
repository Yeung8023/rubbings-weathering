"""Is the gap optimisation or identifiability?  Compare the loss at the true
parameters with the loss the optimiser reaches."""
import sys; sys.path.insert(0,'src')
import numpy as np, torch, torch.nn.functional as F
import synth as S, fuse as Fz, evaluate as E, torchmodel as T

CARVE=632; years=[1050,1350,1600,1800,2000]
chars=list('九成宮醴泉銘秘書監檢校侍中鉅鹿郡公臣魏徵奉勅撰')[:16]
d=S.make_corpus(chars,years,seed=3,size_px=256,carve_year=CARVE)
px=d['px_mm']; m=d['masks']; dt=[(y-CARVE)/100 for y in years]
Y=torch.tensor(d['images'],device='cuda')
C,n=Y.shape[:2]

mdl=Fz.Fusion(C,n,256,256,px,dt=dt,relief='free',spall=True).cuda()
with torch.no_grad():
    h0=torch.tensor(d['h0'],device='cuda')[:,None]
    mdl.a_h0.copy_(torch.log(torch.expm1(h0.clamp_min(1e-4))))
    inv=lambda v,lo,hi: torch.log(torch.tensor(float((v-lo)/(hi-v)),device='cuda'))
    for i,st in enumerate(d['styles']):
        mdl.a_lam[i]=inv(st.rho,0.15,1.30); mdl.a_eps[i]=inv(st.eps,0.03,0.30)
        mdl.a_s[i]=inv(st.s,0.015,0.15); mdl.a_alpha[i]=inv(st.alpha,0.35,0.99)
    mdl.a_rate.copy_(torch.log(torch.expm1(torch.tensor(0.075))))
    mdl.b_rate.copy_(torch.log(torch.expm1(torch.tensor(0.115))))
    # true spall increments
    sp=[]
    for c in range(C):
        pass
    yh,_=mdl(torch.arange(C,device='cuda'))
    print('loss at TRUE h0/styles/rates (spall unset): %.5f'%F.mse_loss(yh,Y).item())
    # with the true spall fields too
    import weather as W
    rng=np.random.default_rng(103)
    # recompute the exact spall fields used by make_corpus is not possible here;
    # instead give the model the true *states* to bound the achievable fit
    states=torch.tensor(d['states'],device='cuda').reshape(C*n,1,256,256)
    lam,eps,s,alpha=mdl.styles()
    y2=T.render_t(states,lam.repeat(C),eps.repeat(C),s.repeat(C),alpha.repeat(C),px)
    print('loss with TRUE weathered states (no warp): %.5f'%F.mse_loss(y2.reshape(C,n,256,256),Y).item())
    print('  (this is the noise+warp floor)')

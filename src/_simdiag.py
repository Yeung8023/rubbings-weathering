import sys; sys.path.insert(0,'src')
import numpy as np, corpus as CP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
A,pa=CP.scan_item('24521',verbose=False); B,pb=CP.scan_item('24592',verbose=False)
print('A',A.shape,'B',B.shape,flush=True)
S=CP.similarity(A,B)
print('S max %.3f mean %.3f'%(S.max(),S.mean()),flush=True)
mx=S.max(1); am=S.argmax(1)
print('rows with max>0.5: %d/%d ; >0.6: %d'%((mx>0.5).sum(),len(mx),(mx>0.6).sum()),flush=True)
np.save('results/_S.npy',S)
fig,ax=plt.subplots(1,3,figsize=(19,6))
ax[0].imshow(S,aspect='auto',cmap='viridis',vmin=0,vmax=0.8); ax[0].set_title('similarity A x B')
ax[1].plot(am,'.',ms=2); ax[1].set_title('argmax per row')
# show the 8 best matches
ordr=np.argsort(-mx)[:8]
gr=np.ones((2*100,8*100))
for k,i in enumerate(ordr):
    gr[0:100,k*100:(k+1)*100]=np.asarray(Image.fromarray(((A[i]-A[i].min())/(np.ptp(A[i])+1e-6)*255).astype('uint8')).resize((100,100)))/255
    j=am[i]
    gr[100:200,k*100:(k+1)*100]=np.asarray(Image.fromarray(((B[j]-B[j].min())/(np.ptp(B[j])+1e-6)*255).astype('uint8')).resize((100,100)))/255
ax[2].imshow(gr,cmap='gray'); ax[2].set_title('top-8 matches (A top / B bottom)'); ax[2].axis('off')
plt.tight_layout(); plt.savefig('results/figs/_sim.png',dpi=90); print('ok')

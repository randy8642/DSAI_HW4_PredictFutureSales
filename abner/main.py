import numpy as np
from sklearn import linear_model
import time
tStart = time.time()


nF = np.load('res.npy')
Tra = nF[nF[:,0]!=33]
Val = nF[nF[:,0]==33]
#%%
Tra_data = Tra[:,1:3]
Tra_label = Tra[:,-1]

#%%
model = linear_model.Lasso(alpha=1e-2)
model.fit(Tra_data, Tra_label)

#%%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
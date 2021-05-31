import numpy as np
import sklearn
import math
from sklearn import linear_model
import time
tStart = time.time()

def _RMSE(pred, real):
    mse = sklearn.metrics.mean_squared_error(pred, real)
    rmse = math.sqrt(mse)
    return rmse


nF = np.load('res.npy')
Tra = nF[nF[:,0]!=33]
Val = nF[nF[:,0]==33]

Tra_array = np.zeros([33, 60, 22170])


#%%
Tra_data = Tra[:,:3]
Tra_label = Tra[:,-1]

#%%
model = linear_model.Lasso(alpha=1e-2)
model.fit(Tra_data, Tra_label)

#%%
Val_data = Val[:,:3]
Val_label = Val[:,-1]
pred = model.predict(Val_data)
Gs = _RMSE(pred, Val_label)
print("RMSE >> ", Gs)

#%%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
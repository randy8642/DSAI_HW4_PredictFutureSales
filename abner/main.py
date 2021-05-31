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

def _PACK(x, nF, L=True, Tra=True):
    if Tra:
        month = 33
    else:
        month = 1
    Tra_array = np.zeros([month, 60, 22170])
    for m in range(Tra_array.shape[0]):
        nF_m = nF[nF[:,0]==m]
        for i in range(Tra_array.shape[1]):
            sh = nF_m[nF_m[:,1]==i]
            for j in range(Tra_array.shape[-1]):
                im = sh[sh[:,2]==j]
                if im.shape[0]==0:
                    Tra_array[m, i, j] = 0
                else:
                    if L:
                        Tra_array[m, i, j] = im[0,-1]
                    else:
                        Tra_array[m, i, j] = 1
    return Tra_array

nF = np.load('res.npy')
Tra = nF[nF[:,0]!=33]
Val = nF[nF[:,0]==33]


Tra_data = _PACK(Tra, nF, L=False)
Tra_label = _PACK(Tra, nF)
Val_data = _PACK(Val, nF, L=False, Tra=False)
Val_label = _PACK(Val, nF, Tra=False)


#%%
model = linear_model.Lasso(alpha=1e-2)
model.fit(Tra_data, Tra_label)

#%%
pred = model.predict(Val_data)
Gs = _RMSE(pred, Val_label)
print("RMSE >> ", Gs)

#%%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
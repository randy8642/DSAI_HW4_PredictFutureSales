import numpy as np
import sklearn
import math
import os
from sklearn import linear_model
import pandas as pd
import time
tStart = time.time()

def _RMSE(pred, real):
    mse = sklearn.metrics.mean_squared_error(pred, real)
    rmse = math.sqrt(mse)
    return rmse

def _Nor(x):
  mu = np.mean(x)
  std = np.std(x)
  nor = (x-mu)/std
  return nor

def _NorID(ID):
  mu = np.mean(ID, axis=0)[np.newaxis,:]
  std = np.std(ID, axis=0)[np.newaxis,:]
  nor = (ID-mu)/std
  return nor
  
nF = np.load('tes_Z.npy')

path = '../data'
Ftes = 'test.csv'
Data_tes = pd.read_csv(os.path.join(path, Ftes), low_memory=False)
Data_tra = np.load('tes_Z.npy')
ID = _NorID(np.array(Data_tes)[:, 1:3])

Tra_data = _Nor(nF[:, :31])
Tra_data = np.hstack((ID, Tra_data))
Tra_label = nF[:, 31]

Val_data = _Nor(nF[:, 1:32])
Val_data = np.hstack((ID, Val_data))
Val_label = nF[:, 32]

Tes_data = _Nor(nF[:, 2:])
Tes_data = np.hstack((ID, Tes_data))

model = linear_model.LassoCV()
model.fit(Tra_data, Tra_label)
#%%
pred = model.predict(Val_data)
Gs = _RMSE(pred, Val_label)
print("RMSE >> ", Gs)
'''
#%%
# pred_tes = model.predict(dtest)
pred_tes = model.predict(Tes_data)
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
# df.to_csv('Lasso_IDxMonth.csv', index=False)
df.to_csv('XG_IDxMonth.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
'''

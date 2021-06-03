import numpy as np
import sklearn
import math
import os
from sklearn import linear_model
import pandas as pd
from xgboost import XGBRegressor
import time
tStart = time.time()

def _RMSE(pred, real):
    mse = sklearn.metrics.mean_squared_error(pred, real)
    rmse = math.sqrt(mse)
    return rmse

def _Nor(x):
  x_min = np.min(x)
  x_max = np.max(x)
  nor = (x - x_min)/(x_max - x_min)
  return nor

def _NorID(ID):
  ID_min = np.min(ID, axis=0)[np.newaxis,:]
  ID_max = np.max(ID, axis=0)[np.newaxis,:]
  nor = (ID - ID_min)/(ID_max - ID_min)
  return nor
  
nF = np.load('tes_Z.npy')
nD = np.load('ID.npy')

#%%
ID = _NorID(nD)
Tra_data = _Nor(nF[:, :31])
Tra_data = np.hstack((ID, Tra_data))
Tra_label = nF[:, 31]

Val_data = _Nor(nF[:, 1:32])
Val_data = np.hstack((ID, Val_data))
Val_label = nF[:, 32]

Tes_data = _Nor(nF[:, 2:])
Tes_data = np.hstack((ID, Tes_data))


#%%
model=XGBRegressor(
    max_depth = 9,
    n_estimators = 500,
    learning_rate = 0.1,
    subsample = 0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree = 0.7)

model.fit(
    Tra_data,
    Tra_label,
    eval_metric='rmse',
    eval_set=[(Tra_data, Tra_label), (Val_data, Val_label)], 
    verbose=True, 
    early_stopping_rounds = 10)


#%%
pred_tes = model.predict(Tes_data)

id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
# df.to_csv('Lasso_IDxMonth.csv', index=False)
df.to_csv('XG_IDxMonth_Proc.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))



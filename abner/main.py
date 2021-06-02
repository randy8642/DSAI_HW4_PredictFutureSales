import numpy as np
import sklearn
import math
from sklearn import linear_model
import pandas as pd
import xgboost as xgb
import time
tStart = time.time()

def _RMSE(pred, real):
    mse = sklearn.metrics.mean_squared_error(pred, real)
    rmse = math.sqrt(mse)
    return rmse

# 演算法引數
params = {
    'booster':'gbtree',
    "reg" : "linear", 
    'gamma':0.1,
    'max_depth':6,
    'lambda':2,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
}
plst = params.items()
nF = np.load('tes_Z.npy')
Tra_data = nF[:, :31]
Tra_label = nF[:, 31]

Val_data = nF[:, 1:32]
Val_label = nF[:, 32]

Tes_data = nF[:, 2:]

dtrain = xgb.DMatrix(Tra_data, Tra_label)
dvalid = xgb.DMatrix(Val_data)
dtest = xgb.DMatrix(Tes_data)
num_rounds = 500

#%%
model = xgb.train(params, dtrain, num_rounds)
# model = linear_model.Lasso(alpha=0.1)
# model.fit(Tra_data, Tra_label)
#%%
pred = model.predict(dvalid)
# pred = model.predict(Val_data)
Gs = _RMSE(pred, Val_label)
print("RMSE >> ", Gs)
#%%
pred_tes = model.predict(dtest)
# pred_tes = model.predict(Tes_data)
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
# df.to_csv('Lasso_IDxMonth.csv', index=False)
df.to_csv('XG_IDxMonth.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

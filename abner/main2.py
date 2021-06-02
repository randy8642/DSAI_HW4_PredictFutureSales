import numpy as np
import sklearn
import math
import os
import matplotlib.pyplot  as plt
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
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

path = '../data'
Ftes = 'test.csv'
Data_tes = pd.read_csv(os.path.join(path, Ftes), low_memory=False)
Data_tra = np.load('tes_Z.npy')
ID = np.array(Data_tes)[:, 1:3]

Tra_data = nF[:, 28:31]
Tra_data = np.hstack((ID, Tra_data))
Tra_label = nF[:, 31]

Val_data = nF[:, 29:32]
Val_data = np.hstack((ID, Val_data))
Val_label = nF[:, 32]

Tes_data = nF[:, 30:33]
Tes_data = np.hstack((ID, Tes_data))

dtrain = xgb.DMatrix(Tra_data, Tra_label)
dvalid = xgb.DMatrix(Val_data)
dtest = xgb.DMatrix(Tes_data)
num_rounds = 500

# #%%
model = xgb.train(params, dtrain, num_rounds)
# plot_importance(model)
# plt.show()
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
df.to_csv('XG_IDxMonth2.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

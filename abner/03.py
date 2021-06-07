from xgboost import XGBRegressor
from xgboost import plot_importance
import pandas as pd
import numpy as np

x = np.load('inputs.npz')
tra_leng = 10913804
tes_leng = 214200

def _2D(x):
    leng = len(x)
    y = np.reshape(x, (leng, -1))
    return y

# def _inputs(x):
tra_emb = _2D(x['train_x_emb'])
tra_other = _2D(x['train_x_other'])
tra_time = _2D(x['train_x_time'])
tra_cnt = _2D(x['train_x_cnt'])
train_y = _2D(x['train_y'])
tes_emb = _2D(x['test_x_emb'])
tes_other = _2D(x['test_x_other'])
tes_time = _2D(x['test_x_time'])
tes_cnt = _2D(x['test_x_cnt'])

train_x = np.hstack([tra_emb, tra_other, tra_time, tra_cnt])
test_x = np.hstack([tes_emb, tes_other, tes_time, tes_cnt])


model=XGBRegressor(
    max_depth = 9,
    n_estimators = 500,
    learning_rate = 0.01,
    subsample = 0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree = 0.7)

model.fit(
    train_x,
    train_y,
    eval_metric='rmse',
    eval_set=[(train_x, train_y)], 
    verbose=True, 
    early_stopping_rounds = 10)

pred_tes = model.predict(test_x)
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
df.to_csv('XG_RY.csv', index=False)

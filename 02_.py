import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance

df = pd.read_hdf('preprocessData.h5', key='df', mode='r')
dlist = ['cate_subtype_id', 'cate_type_id', 'date_shop_avg_item_cnt_lag_1', 'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3', 'date_shop_item_avg_item_cnt_lag_1', 'date_shop_item_avg_item_cnt_lag_2', 'date_shop_item_avg_item_cnt_lag_3']
df = df.drop(dlist, axis=1)

def _XY(df, test=False):
    D = df.copy()
    X = D.drop(['item_cnt_month'], axis=1)
    if test:
        Y = 0
    else:
        Y = D['item_cnt_month']
    return X, Y

# print(df.keys())


train_df = df[df['date_block_num'] < 33]
valid_df = df[df['date_block_num'] == 33]
test_df = df[df['date_block_num'] == 34]

X_train, Y_train = _XY(train_df)
X_valid, Y_valid = _XY(valid_df)
X_test, _ = _XY(test_df, test=True)

model=XGBRegressor(
    max_depth = 9,
    n_estimators = 500,
    learning_rate = 0.01,
    subsample = 0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree = 0.7)

model.fit(
    X_train,
    Y_train,
    eval_metric='rmse',
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)   

plot_importance(model)
plt.show()

# np.savez_compressed('inputs.npz', X_train=X_train, Y_train=Y_train,
#                     X_valid=X_valid, Y_valid=Y_valid, X_test=X_test)


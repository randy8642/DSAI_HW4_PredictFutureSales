import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import time

tStart = time.time()
x = np.load('inputs_lit.npz')

# def _inputs(x):
X_train = (x['X_train'])
Y_train = (x['Y_train'])
X_valid = (x['X_valid'])
Y_valid = (x['Y_valid'])

#%%
model=XGBRegressor(
    max_depth = 9,
    n_estimators = 200,
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

pickle.dump(model, open('XGmodel', "wb"))

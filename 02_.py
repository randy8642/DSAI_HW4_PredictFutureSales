import pandas as pd
import numpy as np
import h5py

df = pd.read_hdf('preprocessData.h5', key='df', mode='r')

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

np.savez_compressed('inputs.npz', X_train=X_train, Y_train=Y_train,
                    X_valid=X_valid, Y_valid=Y_valid, X_test=X_test)


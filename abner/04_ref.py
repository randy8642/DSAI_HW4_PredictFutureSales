import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import time
import model_tf
import config
tStart = time.time()


def _NorID(ID):
  mu = np.mean(ID, axis=0)[np.newaxis,:]
  std = np.std(ID, axis=0)[np.newaxis,:]
  nor = (ID-mu)/std
  return nor

x = np.load('inputs_ref.npz')

# def _inputs(x):
X_train = _NorID(x['X_train'])
Y_train = (x['Y_train'])
X_valid = _NorID(x['X_valid'])
Y_valid = (x['Y_valid'])
X_test = _NorID(x['X_test'])

#%%
X_train = X_train[:,np.newaxis,:]
X_test = X_test[:,np.newaxis,:]


bz = config.batch
model = model_tf.m04(128)
optim_m = keras.optimizers.Adam(learning_rate=config.lr)
model.compile(optimizer=optim_m, 
              loss=keras.losses.MeanSquaredError())
history = model.fit(X_train, Y_train, batch_size=bz,
                    epochs=config.Epoch, verbose=1, shuffle=True,
                    validation_data=(X_valid, Y_valid))              

pred_tes = model.predict(X_test)
pred_tes = np.reshape(pred_tes, (len(pred_tes)))
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
df.to_csv('TF_RY_Ref.csv', index=False)

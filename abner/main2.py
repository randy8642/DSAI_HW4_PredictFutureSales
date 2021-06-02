import numpy as np
import sklearn
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import time
import model_tf
import config
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
Tra_data = np.hstack((ID, Tra_data))[:,np.newaxis,:]
Tra_label = nF[:, 31]

Val_data = _Nor(nF[:, 1:32])
Val_data = np.hstack((ID, Val_data))[:,np.newaxis,:]
Val_label = nF[:, 32]

Tes_data = _Nor(nF[:, 2:])
Tes_data = np.hstack((ID, Tes_data))[:,np.newaxis,:]

# #%%
bz = config.batch
model = model_tf.m04(128)
optim_m = keras.optimizers.Adam(learning_rate=config.lr)
model.compile(optimizer=optim_m, 
              loss=keras.losses.MeanSquaredError())
history = model.fit(Tra_data, Tra_label, batch_size=bz,
                    epochs=config.Epoch, verbose=1, shuffle=True,
                    validation_data=(Val_data, Val_label))

loss = np.array(history.history['loss'])


#%%
pred_tes = model.predict(Tes_data)
pred_tes = np.reshape(pred_tes, (len(pred_tes)))
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
df.to_csv('TF_IDxMonth2.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
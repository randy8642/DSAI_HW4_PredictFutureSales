import numpy as np
import sklearn
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
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
Tra_data = np.hstack((ID, Tra_data))[:,np.newaxis,:]
Tra_label = nF[:, 31]

Val_data = _Nor(nF[:, 1:32])
Val_data = np.hstack((ID, Val_data))[:,np.newaxis,:]
Val_label = nF[:, 32]

Tes_data = _Nor(nF[:, 2:])
Tes_data = np.hstack((ID, Tes_data))[:,np.newaxis,:]


#%%
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
df.to_csv('TF_IDxMonth_Pro.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))



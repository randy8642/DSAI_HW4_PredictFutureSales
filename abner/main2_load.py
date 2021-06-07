import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import time
import tensorflow as tf
tStart = time.time()

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

model = tf.keras.models.load_model("saved_model.hp5")

pred_tes = model.predict(Tes_data)
pred_tes = np.reshape(pred_tes, (len(pred_tes)))
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
df.to_csv('TF_IDxMonth2.csv', index=False)
tEnd = time.time()
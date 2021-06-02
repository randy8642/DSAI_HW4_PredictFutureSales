import numpy as np
import sklearn
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import time
import model
import config
tStart = time.time()

def _RMSE(pred, real):
    mse = sklearn.metrics.mean_squared_error(pred, real)
    rmse = math.sqrt(mse)
    return rmse

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

# #%%
bz = config.batch
model = model.m04
optim_m = keras.optimizers.Adam(learning_rate=config.lr, amsgrad=config.amsgrad)
model.compile(optimizer=optim_m, 
              loss=keras.losses.MeanSquaredError(reduction="auto", 
                name="mean_squared_error"),
              metrics=['accuracy'])
history = model.fit(Tra_data, Tra_label, batch_size=bz,
                    epochs=config.Epoch, verbose=2, shuffle=True,
                    validation_data=(Val_data, Val_label))

loss = np.array(history.history['loss'])
val_acc = np.array(history.history['val_accuracy'])

#%%
# pred_tes = model.predict(dtest)
# id_list = np.arange(0, len(pred_tes), 1).astype(str)
# D = np.vstack([id_list, pred]).T
# df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
# df.to_csv('XG_IDxMonth2.csv', index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

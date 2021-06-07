import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import time
import model_tf
import config
tStart = time.time()

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


bz = config.batch
model = model_tf.m04(128)
optim_m = keras.optimizers.Adam(learning_rate=config.lr)
model.compile(optimizer=optim_m, 
              loss=keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size=bz,
                    epochs=config.Epoch, verbose=1, shuffle=True)              

pred_tes = model.predict(test_x)
pred_tes = np.reshape(pred_tes, (len(pred_tes)))
id_list = np.arange(0, len(pred_tes), 1).astype(str)
D = np.vstack([id_list, pred_tes]).T
df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
df.to_csv('TF_RY.csv', index=False)

import imp
import tensorflow as tf
import numpy as np
from tf_models.LSTM import predictModel




data = np.load('./inputs.npz')

train_x = (data['train_x_emb'], data['train_x_other'], data['train_x_time'], data['train_x_cnt'])
train_y = data['train_y']

test_x = (data['test_x_emb'], data['test_x_other'], data['test_x_time'], data['test_x_cnt'])



model = predictModel(emb_feature_inputDim= 1 + np.max(train_x[0], axis=0), other_feature_cnt=train_x[1].shape[-1], time_feature_cnt=train_x[2].shape[-1])

model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())
model.fit(x=train_x, y=train_y, batch_size=256, epochs=10, verbose=1)

model.save_weights('model_weight.h5')
del model

import tensorflow as tf
from prepareDate import get_train_data
import numpy as np
from models import predictModel


try:
    data = np.load('./data.npz')
    train_x, train_y = data['train_x'], data['train_y']
except:
    train_x, train_y = get_train_data()
    np.savez_compressed('data.npz', train_x=train_x, train_y=train_y)


model = predictModel()
model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())
model.fit(x=train_x, y=train_y, batch_size=64, epochs=10, verbose=1)

model.save_weights('model_weight.h5')
del model

import tensorflow as tf
import tensorflow.keras as keras
from prepareDate import get_train_data

class predictModel(keras.Model):
    def __init__(self):
        super(predictModel, self).__init__()
        
        self.lstm = keras.layers.LSTM(30, activation='tanh')
        
        self.dense_1 = keras.layers.Dense(10, activation='tanh')
        self.dense_2 = keras.layers.Dense(1)
        
    def call(self,x):
        
        pred = self.lstm(x)
        pred = self.dense_1(pred)        
        pred = self.dense_2(pred)

        return pred


train_x, train_y = get_train_data()

model = predictModel()
model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())
model.fit(x=train_x, y=train_y,batch_size=64,epochs=10,verbose=1)

model.save_weights('model_weight.h5')
del model
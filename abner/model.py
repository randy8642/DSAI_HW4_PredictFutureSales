import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v1.keras as keras1
import tensorflow.keras as keras
import numpy as np

class m04(keras.Model):
    def __init__(self, out_sz):
        super(m04, self).__init__()
        FL = keras1.layers.CuDNNLSTM(out_sz, return_sequences=True)
        BL = keras1.layers.CuDNNLSTM(out_sz, go_backwards=True, return_sequences=True)
        self.LSTM = keras.layers.Bidirectional(FL, backward_layer=BL)
        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),            
            keras.layers.Dense(64),
            keras.layers.ReLU()
        ])       

    def call(self, x):
        bz = x.shape[0]
        print(bz)
        y1 = self.LSTM(x)
        y5 = self.FC(y1) 
        # y = keras.layers.Dense(bz)(y5)
        return y5   

#%% Test
if __name__ == "__main__":
    IN = np.random.rand(32,3,33)
    F = m04(64)
    Gen = F(IN)
    print('Gen >>', Gen.shape)
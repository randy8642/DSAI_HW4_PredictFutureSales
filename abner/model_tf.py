import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v1.keras as keras1
import tensorflow.keras as keras
import config
import numpy as np

class m04(keras.Model):
    def __init__(self, out_sz):
        super(m04, self).__init__()
        self.IN = keras.layers.InputLayer((2,), batch_size=config.batch)
        FL = keras1.layers.CuDNNLSTM(out_sz, return_sequences=True)
        BL = keras1.layers.CuDNNLSTM(out_sz, go_backwards=True, return_sequences=True)
        FL2 = keras1.layers.CuDNNLSTM(out_sz//2, return_sequences=True)
        BL2 = keras1.layers.CuDNNLSTM(out_sz//2, go_backwards=True, return_sequences=True)           
        self.LSTM = keras.layers.Bidirectional(FL, backward_layer=BL)
        self.LSTM2 = keras.layers.Bidirectional(FL2, backward_layer=BL2) 
        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),            
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(1)
        ])       

    def call(self, x):
        y1 = self.LSTM(x)
        y2 = self.LSTM2(y1)
        y = self.FC(y2) 
        return y

#%% Test
if __name__ == "__main__":
    IN = np.random.rand(32,3,33)
    F = m04(64)
    Gen = F(IN)
    print('Gen >>', Gen.shape)
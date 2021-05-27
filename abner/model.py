import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v1.keras as keras1
import tensorflow.keras as keras
import numpy as np

class m04(keras.Model):
    def __init__(self, out_sz):
        super(m04, self).__init__()
        FL = keras1.layers.CuDNNGRU(out_sz, return_sequences=True)
        BL = keras1.layers.CuDNNGRU(out_sz, go_backwards=True, return_sequences=True)
        FL2 = keras1.layers.CuDNNGRU(out_sz//2, return_sequences=True)
        BL2 = keras1.layers.CuDNNGRU(out_sz//2, go_backwards=True, return_sequences=True)        
        self.GRU = keras.layers.Bidirectional(FL, backward_layer=BL)
        self.GRU2 = keras.layers.Bidirectional(FL2, backward_layer=BL2) 
        self.Cv = keras.Sequential([
            keras.layers.Conv1D(32, kernel_size=5),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),   
            keras.layers.Conv1D(16, kernel_size=1)
        ])            
        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),            
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(24)
        ])       

    def call(self, x):
        y1 = self.GRU(x)
        y2 = self.GRU2(y1)
        y3 = tf.transpose(y2, perm=[0, 2, 1])
        y4 = self.Cv(y3)
        y = self.FC(y4)        
        return y     

#%% Test
if __name__ == "__main__":
    IN = np.random.rand(1,7,48)
    F = m04(128)
    Gen = F(IN)
    print('Gen >>', Gen.shape)
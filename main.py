import imp
import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class predictModel(keras.Model):
    def __init__(self, emb_feature_inputDim: list, other_feature_cnt: int, time_feature_cnt: int):
        super(predictModel, self).__init__()

        self.emb_layers = list()
        for n, dim in enumerate(emb_feature_inputDim):
            layer = keras.layers.Embedding(
                input_dim=dim, output_dim=10, input_length=1)
            self.emb_layers.append(layer)

        self.other_layers = list()
        for n in range(other_feature_cnt):
            layer = keras.layers.Dense(10)
            self.other_layers.append(layer)

        self.time_layers = list()
        for n in range(time_feature_cnt):
            layer = keras.layers.LSTM(20)
            self.time_layers.append(layer)


        self.denses = keras.Sequential([
            keras.layers.Dense(256, activation='tanh'),            
            keras.layers.Dense(128, activation='tanh'),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(1),
        ])

    def call(self, inputs):

        emb_features, other_features, time_features = inputs

        emb_features = tf.split(emb_features, num_or_size_splits=[
                                1]*emb_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.emb_layers):
            emb_features[n] = layer(emb_features[n])

        other_features = tf.split(other_features, num_or_size_splits=[
                                  1]*other_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.other_layers):
            other_features[n] = tf.expand_dims(layer(other_features[n]), axis=1)

        time_features = tf.split(time_features, num_or_size_splits=[
                                 1]*time_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.time_layers):
            time_features[n] = tf.expand_dims(layer(time_features[n]), axis=1)

        x = tf.concat(emb_features + other_features + time_features, axis=-1)
       
        x = tf.squeeze(x, axis=1)
       
        x = self.denses(x)
    

        return x


data = np.load('./inputs.npz')

train_x = (data['train_x_emb'], data['train_x_other'], data['train_x_time'])
train_y = data['train_y']

test_x = (data['test_x_emb'], data['test_x_other'], data['test_x_time'])


model = predictModel(emb_feature_inputDim= 1 + np.max(train_x[0], axis=0), other_feature_cnt=train_x[1].shape[-1], time_feature_cnt=train_x[2].shape[-1])

model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())
model.fit(x=train_x, y=train_y, batch_size=256, epochs=10, verbose=1)

model.save_weights('model_weight.h5')
del model

import tensorflow as tf
import tensorflow.keras as keras


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

        self.cnt_layer = keras.layers.LSTM(20)

        self.denses = keras.Sequential([
            keras.layers.Dense(256, activation='tanh'),
            keras.layers.Dense(128, activation='tanh'),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(1),
        ])

    def call(self, inputs):

        emb_features, other_features, time_features, cnt_features = inputs

        emb_features = tf.split(emb_features, num_or_size_splits=[
                                1]*emb_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.emb_layers):
            emb_features[n] = layer(emb_features[n])

        other_features = tf.split(other_features, num_or_size_splits=[
                                  1]*other_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.other_layers):
            other_features[n] = tf.expand_dims(
                layer(other_features[n]), axis=1)

        time_features = tf.split(time_features, num_or_size_splits=[
                                 1]*time_features.shape[-1], axis=-1)
        for n, layer in enumerate(self.time_layers):
            time_features[n] = tf.expand_dims(layer(time_features[n]), axis=1)

        cnt_features = tf.expand_dims(self.cnt_layer(cnt_features), axis=1)

        x = tf.concat(emb_features + other_features + time_features, axis=-1)
        x = tf.concat((x, cnt_features), axis=-1)

        x = tf.squeeze(x, axis=1)

        x = self.denses(x)

        return x

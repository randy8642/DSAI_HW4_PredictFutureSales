'''
https://github.com/jayparks/quasi-rnn/blob/master/layer.py
'''

import tensorflow as tf
import tensorflow.keras as keras


class QRNNLayer(keras.layers.Layer):
    def __init__(self, hidden_size, kernel_size, use_attn=False):
        super(QRNNLayer, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_attn = use_attn

        # quasi_conv_layer
        self.conv1d = keras.layers.Conv1D(
            filters=3*hidden_size, kernel_size=kernel_size,data_format='channels_first')
        self.conv_linear = keras.layers.Dense(3*hidden_size)
        self.rnn_linear = keras.layers.Dense(hidden_size)

    def _conv_step(self, inputs, memory=None):
        # inputs: [batch_size x length x hidden_size]
        # memory: [batch_size x memory_size]

        # transpose inputs to feed in conv1d: [batch_size x hidden_size x length]
        print(inputs.shape)
        inputs_ = tf.transpose(inputs, perm=[0, 2, 1])
        print(inputs_.shape)
        padded = tf.pad(inputs_, paddings=[
                        [0, 0], [0, 0], [self.kernel_size-1, 0]])
        print(padded.shape)
        # gates: [batch_size x length x 3*hidden_size]
        gates = tf.transpose(self.conv1d(padded), perm=[0, 2, 1])
        print(gates.shape)
        if memory is not None:
            # broadcast memory
            gates = gates + tf.expand_dims(self.conv_linear(memory), axis=1)
        print(gates.shape)
        # Z, F, O: [batch_size x length x hidden_size]
        Z, F, O = tf.split(gates, num_or_size_splits=3, axis=2)
        return tf.tanh(Z), tf.sigmoid(F), tf.sigmoid(O)

    def _rnn_step(self, z, f, o, c, attn_memory=None):
        # uses 'fo pooling' at each time step
        # z, f, o, c: [batch_size x 1 x hidden_size]
        # attn_memory: [batch_size x length' x memory_size]
        c_ = (1 - f) * z if c is None else f * c + (1 - f) * z
        if not self.use_attn:
            return c_, (o * c_)  # return c_t and h_t

        alpha = tf.nn.softmax(tf.squeeze(tf.matmul(c_, tf.transpose(
            attn_memory, perm=[0, 2, 1])), axis=1))  # alpha: [batch_size x length']
        # context: [batch_size x memory_size]
        context = tf.reduce_sum(tf.expand_dims(
            alpha, axis=-1) * attn_memory, axis=1)
        h_ = tf.expand_dims(self.rnn_linear(
            tf.concat([tf.squeeze(c_, axis=1), context], axis=1)), axis=1)

        # c_, h_: [batch_size x 1 x hidden_size]
        return c_, (o * h_)

    def call(self, inputs, state=None, memory=None):
        # inputs: [batch_size x input_size x length]
        # state: [batch_size x hidden_size]
        c = None if state is None else tf.expand_dims(state, axis=1)  # unsqueeze dim to feed in _rnn_step
        (conv_memory, attn_memory) = (None, None) if memory is None else memory
        
        
        # Z, F, O: [batch_size x length x hidden_size]
        Z, F, O = self._conv_step(tf.transpose(inputs, perm=[0,2,1]), conv_memory)
        
        
        c_time, h_time = [], []
        for time, (z, f, o) in enumerate(zip(tf.split(Z, num_or_size_splits=[1]*Z.shape[1], axis=1), tf.split(F, num_or_size_splits=[1]*F.shape[1], axis=1), tf.split(O, num_or_size_splits=[1]*O.shape[1], axis=1))):
            
            c, h = self._rnn_step(z, f, o, c, attn_memory)
            c_time.append(c)
            h_time.append(h)

        # return concatenated cell & hidden states: [batch_size x length x hidden_size]
        return tf.concat(c_time, axis=1), tf.concat(h_time, axis=1)


class predictModel(keras.Model):
    def __init__(self):
        super(predictModel, self).__init__()

        # self.lstm = keras.layers.GRU(30, activation='tanh')
        self.qrnn_layers = list()
        self.qrnn_layers.append(QRNNLayer(hidden_size=256, kernel_size=3, use_attn=False)) 

        self.dense_1 = keras.layers.Dense(10, activation='tanh')
        self.dense_2 = keras.layers.Dense(1)

    def call(self, inputs):
        # inputs [bs, seq, feature]
        
        h = tf.transpose(inputs, perm=[0, 2, 1])

        cell_states, hidden_states = [None], [None]
        for i,layer in enumerate(self.qrnn_layers):
            
            c, h = layer(h, state=cell_states[i], memory=hidden_states[i])  # c, h: [batch_size x length x hidden_size]     
                  

            # c_last, h_last: [batch_size, hidden_size]           
            c_last = c[:, -1, :]
            h_last = h[:, -1, :]
            cell_states.append(c_last)
            hidden_states.append((h_last, h))            

        _, hidden = hidden_states[-1]

        pred = self.dense_1(hidden)
        pred = self.dense_2(pred)

        return pred


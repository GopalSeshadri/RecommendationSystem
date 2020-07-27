import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from utilities import Util

print(tf.__version__)

class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__(name = 'auto_encoder')
        self.dropout_layer = keras.layers.Dropout(rate=0.1)
        self.EncoderDense1 = keras.layers.Dense(512, activation = tf.nn.relu)
        self.EncoderDense2 = keras.layers.Dense(256, activation = tf.nn.relu)
        self.BottleNeckDense = keras.layers.Dense(128, activation = tf.nn.relu)
        self.DecoderDense1 = keras.layers.Dense(256, activation = tf.nn.relu)
        self.DecoderDense2 = keras.layers.Dense(512, activation = tf.nn.relu)
        self.FinalDense = keras.layers.Dense(1673, activation = tf.nn.relu)

    def call(self, input):
        encoder_out_1 = self.dropout_layer(self.EncoderDense1(input))
        encoder_out_2 = self.dropout_layer(self.EncoderDense2(encoder_out_1))
        bottleneck_out = self.dropout_layer(self.BottleNeckDense(encoder_out_2))
        decoder_out_1 = self.dropout_layer(self.DecoderDense1(bottleneck_out))
        decoder_out_2 = self.dropout_layer(self.DecoderDense2(decoder_out_1))
        final_out =  self.dropout_layer(self.FinalDense(decoder_out_2))
        return final_out

NUM_EPOCHS = 300
BATCH_SIZE = 64

tfidf_matrix =  Util.loadObj('tfidf_matrix')
X = tfidf_matrix.to_numpy()

model = AutoEncoder()
optimizer = keras.optimizers.Adam(lr = 0.000003)
global_step = tf.Variable(0)
loss = lambda x, x_hat: tf.reduce_sum(keras.losses.mean_squared_error(x, x_hat))

model.compile(loss=loss, optimizer = optimizer, metrics=['mse'])
model.fit(x = X, y = X, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS)

reduced = model.BottleNeckDense(model.EncoderDense2(model.EncoderDense1(X)))

reduced_np = reduced.numpy()
indices = tfidf_matrix.index.tolist()
reduced_np_df = pd.DataFrame(reduced_np)
reduced_np_df.index = indices
Util.saveObj(reduced_np_df, 'tfidf_reduced_matrix')

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from utilities import Util

print(tf.__version__)

class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__(name = 'auto_encoder')
        self.EncoderDense1 = keras.layers.Dense(512, activation = tf.nn.relu)
        self.EncoderDense2 = keras.layers.Dense(128, activation = tf.nn.relu)
        self.BottleNeckDense = keras.layers.Dense(64, activation = tf.nn.relu)
        self.DecoderDense1 = keras.layers.Dense(128, activation = tf.nn.relu)
        self.DecoderDense2 = keras.layers.Dense(512, activation = tf.nn.relu)
        self.FinalDense = keras.layers.Dense(1673, activation = tf.nn.relu)

    def call(self, input):
        encoder_out_1 = self.EncoderDense1(input)
        encoder_out_2 = self.EncoderDense2(encoder_out_1)
        bottleneck_out = self.BottleNeckDense(encoder_out_2)
        decoder_out_1 = self.DecoderDense1(bottleneck_out)
        decoder_out_2 = self.DecoderDense2(decoder_out_1)
        final_out =  self.FinalDense(decoder_out_2)
        return final_out, bottleneck_out

NUM_EPOCHS = 100
BATCH_SIZE = 64

X = Util.loadObj('tfidf_matrix')
print(type(X))
print(X.shape)

model = AutoEncoder()
optimizer = keras.optimizers.Adam(lr = 0.00003)
global_step = tf.Variable(0)
loss = lambda x, x_hat: tf.reduce_sum(keras.losses.mean_squared_error(x, x_hat))


for i in range(NUM_EPOCHS):
    print("Epoch: {}".format(i))
    np.random.shuffle(X)
    for j in range(0, len(X), BATCH_SIZE):
        x_in = X[j : j + BATCH_SIZE]

        with tf.GradientTape() as tape:
            x_out, bottleneck_out = model(x_in)
            loss_ = loss(x_in, x_out)
            grads = tape.gradient(loss_, model.trainable_variables)


        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    X_hat, _ = model(X)
    # print(len(loss(X, X_hat).numpy()))
    print("Epoch: {}, Loss: {:.4f}".format(i, loss(X, X_hat).numpy()))

_, reduced = model(X)
Util.saveObj(reduced, 'tfidf_reduced_matrix')

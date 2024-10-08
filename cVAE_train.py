from utils import pre_sensor_data as psd
from utils import pre_keypoint as pk

import tensorflow as tf

from tensorflow.keras.optimizers.legacy import Adam
from keras import backend as K
from keras.layers import (Input, Dense, Lambda, Flatten, Reshape, concatenate)
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from utils import plot as plt
import pandas as pd

# from keras.utils import plot_model

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# batch_size = 64 
latent_size= 2  # latent space size
encoder_dim1 = 512  # dim of encoder hidden layer
decoder_dim = 512  # dim of decoder hidden layer
activ = 'relu'
optim = Adam(learning_rate=0.0001)
n_epoch = 100

mu = None
sigma = None

seconds = 3

# For X (Accelerometer) data, the frequency of data collection for your dataset is approximately 
# 11.75 data points per second and about 0.0118 data points per millisecond.
frequency_x = 10

# For y (Head Position) data, the frequency of data collection for your dataset is approximately
# 30.00 data points per second and about 0.03 data points per millisecond.
frequency_y = 30

# X_acce, frequency_x = psd.regex_get_mag(set_num=0, choosing="shaking") # sensor data
X_acce, frequency_x = psd.get(set_num=0, seconds=seconds, choosing="shaking", sensor="accel") # sensor data
# Shaking accelerometer data from time: 14:33:50:25 to 14:35:56:455
# Nodding accelerometer data from time: 14:37:43:959 to 14:39:48:394

y_acce = pk.get(set_num=0, seconds=seconds, frequency_y=frequency_y, choosing="shaking") # head position data
# y_acce = pk.get_shaking(set_num=0) # head position data
# shaking from time: 14:33:51:355 to 14:35:52:195
# nodding from time: 14:37:45:532 to 14:39:44:612

X_train_acce, X_test_acce = train_test_split(X_acce, test_size=0.2, random_state=42)
y_train_acce, y_test_acce = train_test_split(y_acce, test_size=0.2, random_state=42)

# print(X_train_acce.shape, y_train_acce.shape, X_test_acce.shape, y_test_acce.shape)
print("X_train_acce.shape: ", X_train_acce.shape, "y_train_acce.shape: ", y_train_acce.shape) #(36, 33, 3) (32, 90, 2)
print("X_test_acce.shape: ", X_test_acce.shape, "y_test_acce.shape: ", y_test_acce.shape) # (9, 33, 3) (8, 90, 2)

decoder_out_dim = seconds * frequency_x * 3  # dim of decoder output layer

def sample_z(args):
    mu, sigma = args
    eps = K.random_normal(shape=( latent_size,), mean=0., stddev=1.)#batch_size,
    return mu + K.exp(sigma / 2) * eps

def build_cvae_encoder():
    global mu, sigma
    x_input = Input(shape=(seconds * frequency_x, 3,))
    y_input = Input(shape=(seconds * frequency_y, 2,))

    y = Dense(3)(y_input)

    inputs = concatenate([x_input, y], axis=1)
    encoder_h = Dense(encoder_dim1, activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(inputs) # shape: (None, 330, 512)

    mu = Dense(latent_size, activation='linear')(encoder_h) # shape: (None, 330, 2)
    sigma = Dense(latent_size, activation='linear')(encoder_h) # shape: (None, 330, 2)

    mu_overall = K.mean(mu, axis=1) # shape: (None, 2)

    # Calculate the overall standard deviation for the current 
    # standard deviation for all data

    # The formula is:
    # sigma_overall = sqrt(1/N * sum((mu_i - mu_overall)^2 + sigma_i^2))

    # sigma_overall = K.sqrt(1 / (sigma.shape[1]) * K.sum(K.square(mu - mu_overall) + K.square(sigma), axis=1))
    sigma_overall = K.std(sigma, axis=1) # shape: (None, 2)

    print("mu_overall.shape: ", mu_overall.shape, "sigma_overall.shape: ", sigma_overall.shape)

    mu, sigma = mu_overall, sigma_overall

    print("mu.shape: ", mu.shape, "sigma.shape: ", sigma.shape)
    z = Lambda(sample_z, output_shape=(latent_size,))([mu, sigma]) # shape: (None, 2)
    print("z.shape: ", z.shape)
    encoder_model = Model([x_input, y_input], z)
    encoder_model.summary()
    return encoder_model

def build_cvae_decoder():
    z_input = Input(shape=(latent_size,))
    y_input = Input(shape=(seconds * frequency_y, 2,))
    y = Flatten()(y_input)
    zc = concatenate([z_input, y], axis=1)
    decoder_h_1 = Dense(decoder_dim, activation=activ)
    decoder_h_2 = decoder_h_1(zc)
    decoder_out_1 = Dense(decoder_out_dim, activation='sigmoid')
    decoder_out_2 = decoder_out_1(decoder_h_2)
    decoder_out_2 = Reshape((seconds * frequency_x, 3))(decoder_out_2)
    decoder_model = Model([z_input, y_input], decoder_out_2)
    decoder_model.summary()
    return decoder_model

def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1) # recon loss
    kl = (0.5 * K.sum((K.exp(sigma) + K.square(mu) - 1. - sigma ), axis=-1)) # kl loss

    return recon + kl

def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)

def kl_loss(y_true, y_pred):
    return (0.5 * K.sum((K.exp(sigma) + K.square(mu) - 1. - sigma ), -1))

def build_cvae():
    encoder_model = build_cvae_encoder()
    decoder_model = build_cvae_decoder()

    x_input, y_input = encoder_model.input
    print("x_input.shape: ", x_input.shape, "y_input.shape: ", y_input.shape)
    encoder_output = encoder_model.output
    print("encoder_output.shape: ", encoder_output.shape)

    decoder_out = decoder_model([encoder_output, y_input])

    cvae = Model([x_input, y_input], decoder_out)
    cvae.compile(optimizer=optim, loss=vae_loss, metrics=[recon_loss, kl_loss])
    cvae.summary()
    return cvae, encoder_model, decoder_model

def train_cvae(models, X_train, y_train, X_test, y_test):
    batch_size_train = min(X_train.shape[0], y_train.shape[0])
    batch_size_test = min(X_test.shape[0], y_test.shape[0])

    cvae, encoder_model, decoder_model = models

    # trim the dataset
    X_train, y_train = X_train[:batch_size_train], y_train[:batch_size_train]
    X_test, y_test = X_test[:batch_size_test], y_test[:batch_size_test]

    X_real = np.array([])
    X_fake = np.array([])

    for epoch in range(n_epoch):
        print(f"Epoch {epoch+1}/{n_epoch}")
        for batch in range(batch_size_train):
            cur_X = X_train[batch].reshape(1, seconds * frequency_x, 3)
            cur_y = y_train[batch].reshape(1, seconds * frequency_y, 2)
            # print(np.isnan(cur_X).any(), np.isnan(cur_y).any())
            loss = cvae.train_on_batch([cur_X, cur_y], cur_X)
            # print(f"batch {batch+1}/{batch_size_train}: loss = {loss}")
            # fit() could not be used in here. The validation batch size is different
            # from the training batch size. validation_batch_size could not be used in here. 
            # Did not find a way to solve this problem.

            if epoch == n_epoch - 1:
                X_recon = cvae.predict([cur_X, cur_y])
                X_fake, X_real = np.append(X_fake, X_recon), np.append(X_real, cur_X)
                corr = np.corrcoef(cur_X.flatten(), X_recon.flatten())[0, 1]
                print(f"Epoch {epoch+1}/{n_epoch}, batch {batch}: loss = {loss}, correlation = {corr}")

        print(f"Epoch {epoch+1}/{n_epoch}: loss = {loss}")

    df_real, df_fake = pd.DataFrame(X_real), pd.DataFrame(X_fake)
    df_real.to_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cvae_real.csv', index=False)
    df_fake.to_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cvae_fake.csv', index=False)
    plt.plot_3D(X_real, X_fake)
    plt.plot_1D(X_real, X_fake)

    encoder_model.save_weights('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae_encoder.h5')
    cvae.save_weights('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae.h5')
    return cvae

### Prediction
def construct_numvec(digit, z=None):
    out = np.zeros((1, seconds * frequency_y, 2))
    out[:, digit] = 1.
    return (out)


def cvae_predict(cvae_model, digit):
    sample_3 = construct_numvec(digit)

    plt.figure(figsize=(3, 3))
    rand_input = np.random.rand(seconds * frequency_x, 3)
    rand_input = np.expand_dims(rand_input, axis=0)
    plt.imshow(cvae_model.predict([rand_input, sample_3[:,:10]]).reshape(28,28), cmap = plt.cm.gray)
    plt.axis('off')
    plt.show()

# main
def main():
    print("X_train_acce.shape: ", X_train_acce.shape, "y_train_acce.shape: ", y_train_acce.shape) # (2900, 3) (1186, 3)
    models = build_cvae()
    train_cvae(models, X_train_acce, y_train_acce, X_test_acce, y_test_acce)
    cvae_model, encoder_model, decoder_model = models

    # cvae_model.load_weights('cvae.h5')

    # cvae_predict(cvae_model, 4)

if __name__ == '__main__':
    main()

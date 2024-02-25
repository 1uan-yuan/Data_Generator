import numpy as np
from utils import pre_sensor_data as psd
from utils import pre_keypoint as pk
from utils import plot as plt
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import (Flatten, Dense, Reshape, concatenate, LeakyReLU,
                          Conv2D, BatchNormalization, Activation, 
                          Conv2DTranspose, Input)
from keras.models import Model
import cVAE_train

import tensorflow as tf
from sklearn.model_selection import train_test_split

import pandas as pd

latent_size = 2
n_epoch = 10
activ = 'relu'
lr = 2e-4
optim = Adam(learning_rate=lr)
decay = 6e-8
n_classes = 10
batch_size = 8

seconds = 3
# frequency_x = 10
frequency_y = 30

# X_acce, frequency_x = psd.regex_get_mag(set_num=0, choosing="shaking") # sensor data
X_acce, frequency_x = psd.get(set_num=0, seconds=seconds, choosing="shaking", sensor="accel") # sensor data
# Shaking accelerometer data from time: 14:33:50:25 to 14:35:56:455
# Nodding accelerometer data from time: 14:37:43:959 to 14:39:48:394

y_acce = pk.get(set_num=0, seconds=seconds, frequency_y=frequency_y, choosing="shaking") # head position data
# y_acce = pk.get_shaking(set_num=0) # head position data
# shaking from time: 14:33:51:355 to 14:35:52:195
# nodding from time: 14:37:45:532 to 14:39:44:612

# Trim the data to the same length
X_acce = X_acce[:min(len(X_acce), len(y_acce))]
y_acce = y_acce[:min(len(X_acce), len(y_acce))]

X_train_acce, X_test_acce = train_test_split(X_acce, test_size=0.2, random_state=42)
y_train_acce, y_test_acce = train_test_split(y_acce, test_size=0.2, random_state=42)

def build_discriminator(x_input, y_input):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256, 512, 1024]

    x = x_input

    y = Flatten()(y_input)
    # print("y_input.shape: ", y_input.shape)
    y = Dense(seconds * frequency_x * 3)(y)
    # print("y.shape: ", y.shape)
    y = Reshape((seconds * frequency_x, 3, 1))(y)
    x = concatenate([x, y])
    for filters in layer_filters:
        strides = 1 if filters == layer_filters[-1] else 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model([x_input, y_input], x, name='discriminator')
    optimizer = tf.keras.optimizers.legacy.RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    return discriminator

def build_generator(z_input, y_input):
    kernel_size = 5
    layer_filters = [512, 256, 128, 64, 32, 1]

    y = Flatten()(y_input)
    x = concatenate([z_input, y], axis=1)
    print("x.shape: ", x.shape)
    x = Dense(seconds * frequency_x * 3)(x)
    x = Reshape((frequency_x * seconds, 3, 1))(x)
    print("x.shape: ", x.shape)
    for filters in layer_filters:
        strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)
        print("x.shape: ", x.shape)

    x = Activation('sigmoid')(x)
    generator = Model([z_input, y_input], x, name='generator')
    generator.summary()
    return generator

def build_gan(generator, discriminator):
    discriminator.trainable = False

    z_input, y_input = generator.input
    g_output = generator.output
    print("g_output.shape: ", g_output.shape, "y_input.shape: ", y_input.shape)

    gan_output = discriminator([g_output, y_input])
    gan_model = Model([z_input, y_input], gan_output)

    optimizer = tf.keras.optimizers.legacy.RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    gan_model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    gan_model.summary()
    return gan_model

def load_cvae(model_weights):
    cvae_model, encoder_model, _ = cVAE_train.build_cvae()
    encoder_model.load_weights(model_weights)
    return encoder_model

def generate_real_data(dataset, data_size):
    X_train, y_train = dataset
    idx = np.random.randint(0, X_train.shape[0], data_size)
    
    X_real = X_train[idx]
    y_real = y_train[idx]

    return X_real, y_real

def generate_fake_data(generator, cvae_model, data_size, y_input):
    z_input, y_input = get_latent_point_cvae(cvae_model, data_size, y_input)
    X_fake = generator.predict([z_input, y_input])

    # normalize the data
    min_x, max_x = np.min(X_fake), np.max(X_fake)
    X_fake = np.subtract(X_fake, min_x) / (max_x - min_x)
    return X_fake, y_input

def get_latent_point_cvae(cvae_model, data_size, y_input):
    temp = np.random.randn(data_size * seconds * frequency_x * 3)
    x_input = temp.reshape(data_size, seconds * frequency_x, 3)
    z_cvae = cvae_model.predict([x_input, y_input])

    # min_z, max_z = np.min(z_cvae), np.max(z_cvae)
    # z_cvae = np.subtract(z_cvae, min_z) / (max_z - min_z)
    return z_cvae, y_input

def train(models, dataset):
    generator, discriminator, gan, cvae = models
    X_train, y_train = dataset

    real_values = np.array([])
    fake_values = np.array([])

    batch_per_epoch = 8
    half_batch = int(batch_size/2)
    
    for epoch in range(n_epoch):
        for batch in range(batch_per_epoch):
            # generate real and fake data, each of them is half batch
            X_real, y_real = generate_real_data(dataset, half_batch)
            X_fake, y_fake = generate_fake_data(generator, cvae, half_batch, y_real)

            # reshape the data into shape (batch, seconds * frequency, 3D or 2D, 1)
            X_real = X_real.reshape(-1, seconds * frequency_x, 3, 1)
            X_fake = X_fake.reshape(-1, seconds * frequency_x, 3, 1)

            # start training discriminator
            discriminator.trainable = True

            # train discriminator
            d_out = np.ones((len(X_real), ))
            d_loss1, _ = discriminator.train_on_batch([X_real, y_real], d_out)

            d_out = np.zeros((len(X_fake), ))
            d_loss2, _ = discriminator.train_on_batch([X_fake, y_fake], d_out)

            discriminator.trainable = False
            # end training discriminator

            _, y_input = generate_real_data(dataset, batch_size) # choose batch_size y_input
            z_input, y_input = get_latent_point_cvae(cvae, batch_size, y_input)

            gan_out = np.ones((len(z_input), ))
            g_loss, _ = gan.train_on_batch([z_input, y_input], gan_out)

            # store the fake values if we are in the last epoch
            if epoch == n_epoch - 1:
                fake_values = np.append(fake_values, X_fake)
                real_values = np.append(real_values, X_real)
                # plot_3D(X_real.reshape(-1), X_fake.reshape(-1))

            # calculate the correlation between X_real and X_fake
            X_real = X_real.reshape(2, -1) # (2, 180)
            X_fake = X_fake.reshape(2, -1) # (2, 180)
            correlation_0 = np.corrcoef(X_real[0], X_fake[0])[0, 1]
            correlation_1 = np.corrcoef(X_real[1], X_fake[1])[0, 1]

            print('>Epoch: %d, batch: %d/%d, d1=%.3f, d2=%.3f g=%.3f, correlation_0=%.3f correlation_1=%.3f' %
                    (epoch + 1, batch + 1, batch_per_epoch, d_loss1, d_loss2, g_loss, 
                     correlation_0, correlation_1))
            
        # # avoid overfitting
        # if correlation_0 > 0.8 and correlation_1 > 0.8:
        #     break
            
    # generator.save('my_cgan_cvae_generator.h5')
    df_real, df_fake = pd.DataFrame(real_values), pd.DataFrame(fake_values)
    df_real.to_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cgan_cvae_real.csv', index=False)
    df_fake.to_csv('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\csv_data\\cgan_cvae_fake.csv', index=False)
    plt.plot_3D(real_values, fake_values, limit=(0, 1))
    plt.plot_1D(real_values, fake_values, limit=(0, 1))
    # plt.plot_time_series(real_values, fake_values)

def build_and_train_models(cvae_encoder_file):
    # build the discriminator
    x_input = Input(shape=(seconds * frequency_x, 3, 1), name='x_input')
    y_input = Input(shape=(seconds * frequency_y, 2), name='y_input')
    discriminator = build_discriminator(x_input, y_input)

    # build the generator
    z_input = Input(shape=(latent_size, ), name='z_input')
    generator = build_generator(z_input, y_input)

    gan = build_gan(generator, discriminator)

    # load the cvae model
    cvae_model = load_cvae(cvae_encoder_file)

    models = (generator, discriminator, gan, cvae_model)
    data = (X_train_acce, y_train_acce)
    train(models, data)

# def main():
#     # In this function, we will use build_and_train_models to train the models
#     global n_epoch
#     n_epochs = [10, 20, 30, 40, 50]
    
#     for n_epoch in n_epochs:
#         build_and_train_models("C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae_encoder.h5")

# if __name__ == "__main__":
#     main()
    
build_and_train_models("C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae_encoder.h5")
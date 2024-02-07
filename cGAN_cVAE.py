import numpy as np
import pre_keypoint as pk
import pre_sensor_data as psd
from tensorflow.keras.optimizers.legacy import Adam, SGD
from keras.layers import (Flatten, Dense, Reshape, concatenate, LeakyReLU,
                          Conv2D, BatchNormalization, Activation, 
                          Conv2DTranspose, Input)
from keras.models import Model
import cVAE_train

import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
X_acce, frequency_x = psd.regex_get_acc(set_num=0, choosing="shaking") # sensor data
# Shaking accelerometer data from time: 14:33:50:25 to 14:35:56:455
# Nodding accelerometer data from time: 14:37:43:959 to 14:39:48:394
y_acce = pk.get_shaking(set_num=0) # head position data
# shaking from time: 14:33:51:355 to 14:35:52:195
# nodding from time: 14:37:45:532 to 14:39:44:612

X_acce = np.array(X_acce, dtype=np.float32)
y_acce = np.array(y_acce, dtype=np.float32)

# normalize the data
min_x, max_x = np.min(X_acce), np.max(X_acce)
X_acce = np.subtract(X_acce, min_x) / (max_x - min_x)
min_y, max_y = np.min(y_acce), np.max(y_acce)
y_acce = np.subtract(y_acce, min_y) / (max_y - min_y)

X_train_acce, X_test_acce = train_test_split(X_acce, test_size=0.2, random_state=42)
y_train_acce, y_test_acce = train_test_split(y_acce, test_size=0.2, random_state=42)

latent_size = 2
n_epoch = 20
activ = 'relu'
lr = 2e-4
optim = Adam(learning_rate=lr)
decay = 6e-8
n_classes = 10
batch_size = 8

seconds = 3
# frequency_x = 10
frequency_y = 30

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
    # print("x.shape: ", x.shape)
    x = Dense(seconds * frequency_x * 3)(x)
    x = Reshape((frequency_x * seconds, 3, 1))(x)
    # print("x.shape: ", x.shape)
    for filters in layer_filters:
        strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)
        # print("x.shape: ", x.shape)

    x = Activation('sigmoid')(x)
    generator = Model([z_input, y_input], x, name='generator')
    generator.summary()
    return generator

def build_gan(generator, discriminator):
    discriminator.trainable = False

    z_input, y_input = generator.input
    g_output = generator.output
    # print("g_output.shape: ", g_output.shape, "y_input.shape: ", y_input.shape)

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

def generate_real_data(dataset, data_size, begin_x, begin_y):
    X_train, y_train = dataset
    X_real, y_real = np.array([]), np.array([])
    X_real = X_train[begin_x: begin_x + frequency_x * seconds * data_size]
    y_real = y_train[begin_y: begin_y + frequency_y * seconds * data_size]

    X_real = X_real.reshape(data_size, seconds * frequency_x, 3, 1)
    y_real = y_real.reshape(data_size, seconds * frequency_y, 2)
    return X_real, y_real

def generate_fake_data(generator, cvae_model, data_size, y_input):
    z_input, y_input = get_latent_point_cvae(cvae_model, data_size, y_input)
    # print("z_input.shape: ", z_input.shape, "y_input.shape: ", y_input.shape)
    X_fake = generator.predict([z_input, y_input])

    # normalize the data
    min_x, max_x = np.min(X_fake), np.max(X_fake)
    X_fake = np.subtract(X_fake, min_x) / (max_x - min_x)
    return X_fake, y_input

def get_latent_point_cvae(cvae_model, data_size, y_input):
    temp = np.random.random(data_size * seconds * frequency_x * 3)
    x_input = temp.reshape(data_size, seconds * frequency_x, 3)
    z_cvae = cvae_model.predict([x_input, y_input]) # (data_size, 2)

    # min_z, max_z = np.min(z_cvae), np.max(z_cvae)
    # z_cvae = np.subtract(z_cvae, min_z) / (max_z - min_z)
    return z_cvae, y_input

def train(models, dataset):
    generator, discriminator, gan, cvae = models

    batch_per_epoch = 8
    half_batch = int(batch_size/2)

    real_values = np.array([])
    fake_values = np.array([])
    
    for epoch in range(n_epoch):
        begin_x = 0
        begin_y = 0
        for batch in range(batch_per_epoch):
            # generate real and fake data, each of them is half batch
            X_real, y_real = generate_real_data(dataset, half_batch, begin_x, begin_y)
            X_fake, y_fake = generate_fake_data(generator, cvae, half_batch, y_real)

            # start train discriminator
            discriminator.trainable = True

            d_out = np.ones((len(X_real), ))
            d_loss1, _ = discriminator.train_on_batch([X_real, y_real], d_out)

            d_out = np.zeros((len(X_fake), ))
            d_loss2, _ = discriminator.train_on_batch([X_fake, y_fake], d_out)

            discriminator.trainable = False
            # end training discriminator

            z_input, y_input = get_latent_point_cvae(cvae, half_batch, y_real)

            gan_out = np.ones((len(z_input), ))
            g_loss, _ = gan.train_on_batch([z_input, y_input], gan_out)

            begin_x += frequency_x * seconds * half_batch
            begin_y += frequency_y * seconds * half_batch

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
    plot_3D(real_values, fake_values)
    plot_1D(real_values, fake_values)
    plot_time_series(real_values, fake_values)

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

def plot_3D(real_values, fake_values):
    # print("real_values.shape: ", real_values.shape, "fake_values.shape: ", fake_values.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1, z1 = real_values[::3], real_values[1::3], real_values[2::3]
    x2, y2, z2 = fake_values[::3], fake_values[1::3], fake_values[2::3]

    ax.scatter(x1, y1, z1, c='r', label='real')
    ax.scatter(x2, y2, z2, c='b', label='fake')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

def plot_1D(real_values, fake_values):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    x1, y1, z1 = real_values[::3], real_values[1::3], real_values[2::3]
    x2, y2, z2 = fake_values[::3], fake_values[1::3], fake_values[2::3]

    # Plotting x1 and x2 in the first subplot
    axs[0].plot(x1, label='Real x', marker='o')
    axs[0].plot(x2, label='Fake x', marker='^')
    axs[0].set_title('X comparison')
    axs[0].legend()

    # Plotting y1 and y2 in the second subplot
    axs[1].plot(y1, label='Real y', marker='o')
    axs[1].plot(y2, label='Fake y', marker='^')
    axs[1].set_title('Y comparison')
    axs[1].legend()

    # Plotting z1 and z2 in the third subplot
    axs[2].plot(z1, label='Real z', marker='o')
    axs[2].plot(z2, label='Fake z', marker='^')
    axs[2].set_title('Z comparison')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_time_series(real_values, fake_values):
    ts1 = pd.Series(real_values) # real data
    ts2 = pd.Series(fake_values) # fake data

    plt.figure(figsize=(12, 6))
    plt.plot(ts1.index, ts1.values, label='Real data', linewidth=2, linestyle='-')
    plt.plot(ts2.index, ts2.values, label='Fake data', linewidth=2, linestyle='--')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()



build_and_train_models("C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae_encoder.h5")
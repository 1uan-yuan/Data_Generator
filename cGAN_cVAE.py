import numpy as np
import pre_keypoint as pk
import pre_sensor_data as psd
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import (Flatten, Dense, Reshape, concatenate, LeakyReLU,
                          Conv2D, BatchNormalization, Activation, 
                          Conv2DTranspose, Input)
from keras.models import Model
import cVAE_train

import tensorflow as tf
from sklearn.model_selection import train_test_split

X_acce = psd.regex_get_acc(set_num=0, choosing="shaking") # sensor data
# Shaking accelerometer data from time: 14:33:50:25 to 14:35:56:455
# Nodding accelerometer data from time: 14:37:43:959 to 14:39:48:394
y_acce = pk.get_shaking(set_num=0) # head position data
# shaking from time: 14:33:51:355 to 14:35:52:195
# nodding from time: 14:37:45:532 to 14:39:44:612

X_acce = np.array(X_acce, dtype=np.float32)
y_acce = np.array(y_acce, dtype=np.float32)

X_train_acce, X_test_acce = train_test_split(X_acce, test_size=0.2, random_state=42)
y_train_acce, y_test_acce = train_test_split(y_acce, test_size=0.2, random_state=42)

latent_size = 2
n_epoch = 10
activ = 'relu'
lr = 2e-4
optim = Adam(learning_rate=lr)
decay = 6e-8
n_classes = 10
batch_size = 8

seconds = 3
frequency_x = 10
frequency_y = 30

def build_discriminator(x_input, y_input):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

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
    layer_filters = [128, 64, 32, 1]

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

def generate_real_data(dataset, data_size, begin_x, begin_y):
    X_train, y_train = dataset
    X_real, y_real = np.array([]), np.array([])
    X_real = X_train[begin_x: begin_x + frequency_x * seconds * data_size]
    y_real = y_train[begin_y: begin_y + frequency_y * seconds * data_size]

    X_real = X_real.reshape(data_size, seconds * frequency_x, 3, 1)
    y_real = y_real.reshape(data_size, seconds * frequency_y, 2)
    return X_real, y_real

def generate_fake_data(generator, cvae_model, data_size):
    z_input, y_input = get_latent_point_cvae(cvae_model, data_size)
    # print("z_input.shape: ", z_input.shape, "y_input.shape: ", y_input.shape)
    X_fake = generator.predict([z_input, y_input])

    # normalize the data
    min_x, max_x = np.min(X_fake), np.max(X_fake)
    X_fake = np.subtract(X_fake, min_x) / (max_x - min_x)
    return X_fake, y_input

def get_latent_point_cvae(cvae_model, data_size):
    temp = np.random.randn(data_size * seconds * frequency_x * 3)
    x_input = temp.reshape(data_size, seconds * frequency_x, 3)
    temp = np.random.randn(data_size * 2 * seconds * frequency_y)
    y_input = temp.reshape(data_size, seconds * frequency_y, 2)
    print("x_input.shape: ", x_input.shape, "y_input.shape: ", y_input.shape)
    z_cvae = cvae_model.predict([x_input, y_input])
    print("z_cvae.shape: ", z_cvae.shape)
    print("z_cvae: ", z_cvae)

    min_z, max_z = np.min(z_cvae), np.max(z_cvae)
    z_cvae = np.subtract(z_cvae, min_z) / (max_z - min_z)
    return z_cvae, y_input

def train(models, dataset):
    generator, discriminator, gan, cvae = models
    X_train, y_train = dataset


    batch_per_epoch = 8
    half_batch = int(batch_size/2)
    
    for epoch in range(n_epoch):
        begin_x = 0
        begin_y = 0
        for batch in range(8):
            # generate real and fake data, each of them is half batch
            X_real, y_real = generate_real_data(dataset, half_batch, begin_x, begin_y)
            X_fake, y_fake = generate_fake_data(generator, cvae, half_batch)

            # train discriminator
            d_out = np.ones((len(X_real), ))
            d_loss1, _ = discriminator.train_on_batch([X_real, y_real], d_out)

            d_out = np.zeros((len(X_fake), ))
            d_loss2, _ = discriminator.train_on_batch([X_fake, y_fake], d_out)

            z_input, y_input = get_latent_point_cvae(cvae, batch_size)

            gan_out = np.ones((len(z_input), ))
            g_loss, _ = gan.train_on_batch([z_input, y_input], gan_out)

            begin_x += frequency_x * seconds * half_batch
            begin_y += frequency_y * seconds * half_batch

            print('>Epoch: %d, batch: %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (epoch + 1, batch + 1, batch_per_epoch, d_loss1, d_loss2, g_loss))
            
    # generator.save('my_cgan_cvae_generator.h5')

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

build_and_train_models("C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\cvae_encoder.h5")
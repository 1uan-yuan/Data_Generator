from keras.layers import (Input, Dense, Reshape, Flatten,
                        concatenate, Conv2D, LeakyReLU, 
                        BatchNormalization, Conv2DTranspose, Activation)
from keras.models import Model
import tensorflow as tf

import pre_sensor_data as psd
import pre_keypoint as pk
import numpy as np

from sklearn.model_selection import train_test_split

X_train_acce = psd.regex_get_acc(set_num=1, choosing="nodding")
y_train_acce = pk.get_nodding(set_num=1)

X_train_acce = np.array(X_train_acce, dtype=np.float32)
y_train_acce = np.array(y_train_acce, dtype=np.float32)

X_training_acce, X_testing_acce = train_test_split(X_train_acce, test_size=0.2, random_state=42)
y_training_acce, y_testing_acce = train_test_split(y_train_acce, test_size=0.2, random_state=42)

print("X_training_acce.shape: ", X_training_acce.shape, "y_training_acce.shape: ", y_training_acce.shape)
print("X_testing_acce.shape: ", X_testing_acce.shape, "y_testing_acce.shape: ", y_testing_acce.shape)

lr = 2e-4
decay = 6e-8
latent_size = 2
seconds = 3
frequency_x = 10
frequency_y = 30

def build_discriminator(x_input, y_input):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = x_input

    # Right now, the shape of y_input is (None, 30, 2)
    # This dense layer will change the shape of y_input
    # from (None, 30, 2) to (None, 30, 3)
    y = Dense(3)(y_input)

    y = Reshape((seconds * frequency_y, 3, 1))(y)
    x = concatenate([x, y], axis=1)

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
    x = Dense(3 * frequency_x * seconds)(x)
    x = Reshape((frequency_x * seconds, 3, 1))(x)
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

def build_and_train():
    X_input = Input(shape=(seconds * frequency_x, 3, 1), name='X_input')
    y_input = Input(shape=(seconds * frequency_y, 2), name='y_input')

    discriminator_model = build_discriminator(X_input, y_input)

    z_input = Input(shape=(latent_size, ), name='z_input')
    generator_model = build_generator(z_input, y_input)
    # print("generator_model.output.shape: ", generator_model.output.shape)

    gan_model = build_gan(generator_model, discriminator_model)

    models = (generator_model, discriminator_model, gan_model)
    data = (X_training_acce, y_training_acce)
    train(models, data)

def generate_real_data(dataset, data_size, begin_x, begin_y):
    X_train, y_train = dataset
    X_real, y_real = np.array([]), np.array([])
    X_real = X_train[begin_x: begin_x + frequency_x * seconds * data_size]
    y_real = y_train[begin_y: begin_y + frequency_y * seconds * data_size]

    X_real = X_real.reshape(data_size, seconds * frequency_x, 3, 1)
    y_real = y_real.reshape(data_size, seconds * frequency_y, 2)
    return X_real, y_real

def generate_fake_data(generator, data_size):
    z_input, y_input = generate_latent_points(data_size)
    print("z_input.shape: ", z_input.shape, "y_input.shape: ", y_input.shape)
    X_fake = generator.predict([z_input, y_input])
    return X_fake, y_input

def generate_latent_points(data_size):
    temp = np.random.randn(data_size * latent_size)
    z_input = temp.reshape(data_size, latent_size)
    temp = np.random.randn(data_size * 2 * seconds * frequency_y)
    y_input = temp.reshape(data_size, seconds * frequency_y, 2)
    return z_input, y_input

def train(models, data):
    generator, discriminator, gan = models
    X_train, y_train = data

    # The relationship of batch_size, num_batches and data_size:
    # 
    #     batch_num = min(size of X / seconds * frequency_x * half_batch, size of y / seconds * frequency_y * half_batch)
    batch_size = 4
    epochs = 10
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        begin_x = 0
        begin_y = 0
        for batch in range(8):
            X_real, y_real = generate_real_data(data, half_batch, begin_x, begin_y)
            X_fake, y_fake = generate_fake_data(generator, half_batch)

            begin_x += frequency_x * seconds * half_batch
            begin_y += frequency_y * seconds * half_batch

            d_out = np.ones((len(X_real), ))
            d_loss1, _ = discriminator.train_on_batch([X_real, y_real], d_out)

            d_out = np.zeros((len(X_fake), ))
            d_loss2, _ = discriminator.train_on_batch([X_fake, y_fake], d_out)

            # Get the correlation between X_real and X_fake
            # Change X_real and X_fake to 2D
            # Reshape the data
            data1_flat = X_real.reshape(2, -1)
            data2_flat = X_fake.reshape(2, -1)

            correlations = np.zeros(data1_flat.shape[0])

            # Compute correlation coefficient for each corresponding row
            for i in range(data1_flat.shape[0]):
                correlations[i] = np.corrcoef(data1_flat[i], data2_flat[i])[0, 1]
                print("correlations[i]: ", correlations[i])
                # print("correlation: ", np.corrcoef(data1_flat[i], data2_flat[i]))

            # # Writing correlations to a text file
            # with open('C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\correlation_data.txt', 'a') as file:
            #     for i, corr in enumerate(correlations):
            #         file.write(f"Correlation between row {i+1} of dataset 1 and dataset 2: {corr}\n")

            #     file.write(f"Average correlation: {np.mean(correlations)}\n\n")

            # with open("C:\\Users\\xueyu\\Desktop\\evasion\\Data_Generator\\CGAN_weights.txt", "w") as file:
            #     for layer in gan.layers:
            #         weights = layer.get_weights()
            #         name = layer.name
            #         file.write(f"Layer name: {name}\n")
            #         file.write("Weights:\n")
            #         file.write(str(weights))
            #         file.write("\n")
            #         file.flush()

            z_input, y_input = generate_latent_points(batch_size)
            gan_out = np.ones((len(z_input,)))
            g_loss, _ = gan.train_on_batch([z_input, y_input], gan_out)
            print("epoch: %d, batch: %d, d_loss1: %f, d_loss2: %f, g_loss: %f" % (epoch, batch, d_loss1, d_loss2, g_loss))
    # generator.save('cgan.h5')

if __name__ == '__main__':
    build_and_train()
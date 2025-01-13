import numpy as np
import matplotlib.pyplot as plt
import requests
from keras.layers import Input, Conv2D, Dropout, Dense, BatchNormalization, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Sequential, Model

url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/pear.npy"
response = requests.get(url)
file_name = "pear.npy"
with open(file_name, "wb") as file:
    file.write(response.content)

print(f"File downloaded and saved as {file_name}")

X = np.load(file_name)
X = X / 255.0
X = X(-1, 28, 28, 1)

def build_discriminator(img_shape=(28, 28, 1)):
    inputs = Input(img_shape)
    x = Conv2D(64, 5, strides=2, padding='same', activation='relu')(inputs)
    x = Dropout(0.4)(x)
    x = Conv2D(128, 5, strides=2, padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(256, 5, strides=2, padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def build_generator(latent_dim=100):
    inputs = Input((latent_dim,))
    x = Dense(7 * 7 * 64, activation='relu')(inputs)
    x = BatchNormalization(momentum=0.9)(x)
    x = Reshape((7, 7, 64))(x)
    x = Dropout(0.4)(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(32, 5, padding='same', activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(16, 5, padding='same', activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(8, 5, padding='same', activation=None)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    output = Conv2D(1, 5, padding='same', activation='sigmoid')(x)
    model = Model(inputs, output)
    return model

def advs(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

discriminator = build_discriminator()
generator = build_generator()
adversarial_model = advs(generator, discriminator)     


if __name__ == "__main__":

    EPOCH = 2000
    BATCH_SIZE = 64
    LAT_DIM = 100

    for i in range(EPOCH):
        b = X[np.random.choice(X.shape[0], BATCH_SIZE)]
        real = np.reshape(b, (BATCH_SIZE, 28, 28, 1))

        noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, LAT_DIM])
        fake = generator.predict(noise, verbose=False)
        
        x = np.concatenate((real, fake))
        y = np.ones([2 * BATCH_SIZE, 1])
        y[BATCH_SIZE:, :] = 0
        discriminator.train_on_batch(x, y)

        y = np.ones([BATCH_SIZE, 1])
        adversarial_model.train_on_batch(noise, y)
        
        if (i+1) % 400 == 0:
            noise = np.random.uniform(-1.0, 1.0, size=[16, LAT_DIM])
            generated = generator.predict(noise, verbose=False)
            _, axs = plt.subplots(4, 4, figsize=(4, 4))
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(generated[i * 4 + j, :, :, 0], cmap='gray_r')
                    axs[i, j].axis('off')
            plt.show()
            
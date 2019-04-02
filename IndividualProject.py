from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.python.client import device_lib
from keras import backend as K

print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()
# Load data
(X_train, _), (_, _) = mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 784)
X_train = X_train.astype('float32') / 255.

# Set the dimensions of the noise
z_dim = 100

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
g = Sequential()
# first block
g.add(Dense(256, input_dim=z_dim))
g.add(LeakyReLU(0.2))
# second block
g.add(Dense(512))
g.add(LeakyReLU(0.2))

g.add(Dense(1024))
g.add(LeakyReLU(0.2))

g.add(Dense(784, activation='tanh'))
# g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Discrinimator
d = Sequential()
d.add(Dense(1024 ,input_dim=784))
d.add(LeakyReLU(0.3))
d.add(Dropout(0.4))

d.add(Dense(512))
d.add(LeakyReLU(0.15))
d.add(Dropout(0.4))

d.add(Dense(256))
d.add(LeakyReLU(0.1))
d.add(Dropout(0.25))

d.add(Dense(128))
d.add(LeakyReLU(0.05))

d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# GAN
d.trainable = False
inputs = Input(shape=(z_dim,))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Generate images
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(15,15)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('ProjectImages2/gan_generated_image %d.png' %epoch)

# Training
def train(epochs=20, plt_frq=1, BATCH_SIZE=128):
    g.summary()
    d.summary()
    gan.summary()
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)

    for e in (range(1, epochs + 1)):
        print("Epoch:", e)
        for _ in range(batchCount):
            # Create a batch by drawing random index numbers from the training set
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))

            # Generate the images from the noise
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            # Create labels
            y = np.zeros(2 * BATCH_SIZE)
            y[:BATCH_SIZE] = 1

            # Train discriminator on generated images
            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, g)
t1 = time.time()
train(epochs=400)
t2 = time.time()
# serialize model to JSON
model_json = g.to_json()
with open("ProjectImages2/generator.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
g.save_weights("generator.h5")
print("Saved model to disk")
print("Time used: " + str(t2 - t1))

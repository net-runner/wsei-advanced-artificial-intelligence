import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

latent_dim = 2

def build_encoder(latent_dim):
    encoder_input = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(encoder_input)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(200, activation='relu')(x)
    encoder_output = layers.Dense(latent_dim, activation='tanh')(x)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    return encoder

def build_decoder(latent_dim):
    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(200, activation='relu')(decoder_input)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(28*28*1, activation='relu')(x)
    x = layers.Reshape((28, 28, 1))(x)
    decoder_output = x
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    return decoder

encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
autoencoder_input = encoder.input
autoencoder_output = decoder(encoder.output)
autoencoder = keras.Model(autoencoder_input, autoencoder_output, name='autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

n = 10
indices = np.random.randint(0, x_test.shape[0], size=n)
sample_images = x_test[indices]
reconstructed_images = autoencoder.predict(sample_images)

plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

encoded_imgs = encoder.predict(x_test)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='tab10', alpha=0.5)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Visualization")
plt.show()

num_decodings = 15
random_latent_vectors = np.random.uniform(-1, 1, size=(num_decodings, latent_dim))
generated_images = decoder.predict(random_latent_vectors)

plt.figure(figsize=(15, 3))
for i in range(num_decodings):
    ax = plt.subplot(1, num_decodings, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Images from Random Latent Vectors")
plt.show()
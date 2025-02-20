import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


IMAGE_SIZE = 64
IMAGE_CHANNELS = 3  

image_files = [os.path.join('cats', file) for file in os.listdir('cats') if file.endswith(('png', 'jpg', 'jpeg'))]


training_data = []

for file in image_files:
    image = Image.open(file).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    training_data.append(np.array(image))

# Konwersja do tablicy NumPy i normalizacja
training_data = np.array(training_data)
training_data = (training_data - 127.5) / 127.5  # Normalizacja do przedziału [-1, 1]


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def build_generator():
    model = Sequential()

    model.add(Dense(8 * 8 * 256, activation='relu', input_dim=100))  # Wektor losowego szumu o wymiarze 100
    model.add(Reshape((8, 8, 256)))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(IMAGE_CHANNELS, kernel_size=5, strides=1, padding='same', activation='tanh'))

    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Budowa i kompilacja modelu GAN
generator = build_generator()

# W modelu GAN tylko generator jest trenowany
discriminator.trainable = False

gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

def sample_images(generator, epoch, grid_size=4):
    noise = np.random.normal(0, 1, (grid_size * grid_size, NOISE_DIM))
    gen_images = generator.predict(noise)

    # Skalowanie obrazów z przedziału [-1,1] do [0,1]
    gen_images = 0.5 * gen_images + 0.5

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    cnt = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].imshow(gen_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.tight_layout()
    plt.show()
    # Możesz również zapisać obrazy do pliku
    # fig.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()

# Parametry treningu
EPOCHS = 10000
BATCH_SIZE = 64
SAMPLE_INTERVAL = 1000

# Etykiety dla prawdziwych i fałszywych obrazów
real = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):

    # Pobieranie losowego zbioru prawdziwych obrazów
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    real_images = training_data[idx]

    # Generowanie losowych szumów i tworzenie fałszywych obrazów
    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
    fake_images = generator.predict(noise)

    # Trenowanie dyskryminatora
    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


    # Generowanie losowych szumów
    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))

    # Trenowanie generatora poprzez maksymalizację błędu dyskryminatora na fałszywych obrazach
    g_loss = gan.train_on_batch(noise, real)

    # Wyświetlanie postępu
    if (epoch + 1) % SAMPLE_INTERVAL == 0:
        print(f"{epoch + 1}/{EPOCHS}, D Loss: {d_loss[0]}, D Acc.: {100 * d_loss[1]}, G Loss: {g_loss}")
        # Zapisywanie wygenerowanych obrazów
        sample_images(generator, epoch + 1)

# Generowanie nowych obrazów po treningu
noise = np.random.normal(0, 1, (1, 100))
generated_image = generator.predict(noise)

# Skalowanie i wyświetlenie obrazu
generated_image = 0.5 * generated_image + 0.5
plt.imshow(generated_image[0])
plt.axis('off')
plt.show()
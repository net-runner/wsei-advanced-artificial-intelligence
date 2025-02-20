import numpy as np
from tensorflow import keras as k
from keras.datasets import cifar10
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

classes_to_use = [0, 1, 2, 3]

indices_train = np.isin(y_train_full.flatten(), classes_to_use)
x_train_full = x_train_full[indices_train]
y_train_full = y_train_full[indices_train]

indices_test = np.isin(y_test_full.flatten(), classes_to_use)
x_test = x_test_full[indices_test]
y_test = y_test_full[indices_test]

x_train = x_train_full[:2000]
y_train = y_train_full[:2000]

mean = np.mean(x_train, axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

N_CLASSES = len(classes_to_use)
y_train_one_hot = k.utils.to_categorical(y_train, N_CLASSES)
y_test_one_hot = k.utils.to_categorical(y_test, N_CLASSES)

def cnn_model(num_filters=32, N_CLASSES=4):
    drop_dense = 0.5
    drop_conv = 0
    model = k.models.Sequential()
    model.add(k.layers.Conv2D(num_filters, (3, 3), activation='relu', 
                              kernel_regularizer=None, input_shape=(32, 32, 3), padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(num_filters, (3, 3), activation='relu',
                              kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Conv2D(2*num_filters, (3, 3), activation='relu',
                              kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(2*num_filters, (3, 3), activation='relu',
                              kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Conv2D(4*num_filters, (3, 3), activation='relu',
                              kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.Conv2D(4*num_filters, (3, 3), activation='relu',
                              kernel_regularizer=None,padding='same'))
    model.add(k.layers.BatchNormalization(axis=-1))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Dropout(drop_conv))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512, activation='relu',kernel_regularizer=None))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Dropout(drop_dense))
    model.add(k.layers.Dense(N_CLASSES, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=k.optimizers.Adam(learning_rate=0.001, decay=0, 
                                    beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    )
    return model

rotation_range = [0, 15, 30, 45]
width_shift_range = [0.0, 0.1, 0.2]
height_shift_range = [0.0, 0.1, 0.2]
zoom_range = [0.0, 0.1, 0.2]

n_iter = 10 

results = []
params_list = []

K = 3
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

for i in range(n_iter):
    rot_range = np.random.choice(rotation_range)
    w_shift = np.random.choice(width_shift_range)
    h_shift = np.random.choice(height_shift_range)
    z_range = np.random.choice(zoom_range)
    
    params = {
        'rotation_range': rot_range,
        'width_shift_range': w_shift,
        'height_shift_range': h_shift,
        'zoom_range': z_range
    }
    print(f"Iteration {i+1}/{n_iter}, testing parameters: {params}")
    val_acc_list = []
    
    for train_index, val_index in kfold.split(x_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train_one_hot[train_index], y_train_one_hot[val_index]
        
        datagen = ImageDataGenerator(
            rotation_range=rot_range,
            width_shift_range=w_shift,
            height_shift_range=h_shift,
            zoom_range=z_range,
            horizontal_flip=True
        )

        datagen.fit(x_train_fold)
        
        model = cnn_model(num_filters=32, N_CLASSES=N_CLASSES)
        
        callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
        
        history = model.fit(
            datagen.flow(x_train_fold, y_train_fold, batch_size=32),
            epochs=10,
            validation_data=(x_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=0
        )
        
        val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold, verbose=0)
        val_acc_list.append(val_acc)
        
    avg_val_acc = np.mean(val_acc_list)
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    
    results.append(avg_val_acc)
    params_list.append(params)

best_index = np.argmax(results)
best_params = params_list[best_index]
best_val_acc = results[best_index]

print(f"\nBest augmentation parameters found:")
print(f"Parameters: {best_params}")
print(f"Validation Accuracy: {best_val_acc:.4f}")

print("\nTraining final model with best augmentation parameters...")

from sklearn.model_selection import train_test_split
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
    x_train, y_train_one_hot, test_size=0.1, random_state=42
)

train_datagen = ImageDataGenerator(
    rotation_range=best_params['rotation_range'],
    width_shift_range=best_params['width_shift_range'],
    height_shift_range=best_params['height_shift_range'],
    zoom_range=best_params['zoom_range'],
    horizontal_flip=True
)
train_datagen.fit(x_train_final)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_val_final)

model = cnn_model(num_filters=32, N_CLASSES=N_CLASSES)

history = model.fit(
    train_datagen.flow(x_train_final, y_train_final, batch_size=32),
    epochs=20,
    validation_data=(x_val_final, y_val_final),
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)],
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

#Test accuracy: 0.7595
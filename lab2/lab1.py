import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

#1
df_train = pd.read_csv('mnist_train.csv', header=None)
df_test = pd.read_csv('mnist_test.csv', header=None)

X_train = df_train.iloc[:, 1:].values 
y_train = df_train.iloc[:, 0].values   

X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shapes -> X_test: {X_test.shape}, y_test: {y_test.shape}")

#2
digit1 = 0
digit2 = 1

train_filter = np.isin(y_train, [digit1, digit2])
X_train_filtered = X_train[train_filter]
y_train_filtered = y_train[train_filter]

y_train_filtered = np.where(y_train_filtered == digit1, 1, 0)

test_filter = np.isin(y_test, [digit1, digit2])
X_test_filtered = X_test[test_filter]
y_test_filtered = y_test[test_filter]

y_test_filtered = np.where(y_test_filtered == digit1, 1, 0)

print(f"Filtered Shapes -> X_train: {X_train_filtered.shape}, y_train: {y_train_filtered.shape}")
print(f"Filtered Shapes -> X_test: {X_test_filtered.shape}, y_test: {y_test_filtered.shape}")

#3
num_samples = 5
for i in range(num_samples):
    image = X_train_filtered[i].reshape(28, 28)
    label = y_train_filtered[i]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

#4 Normalizacja
X_train_normalized = X_train_filtered / 255.0
X_test_normalized = X_test_filtered / 255.0

#5 Perceprton inicjalizacja
input_size = X_train_normalized.shape[1]  
perceptron = Perceptron(input_size)

#6

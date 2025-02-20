import numpy as np
import matplotlib.pyplot as plt

xx = np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
d = np.array([0, 1, 1, 0])

def sigmoid(x, beta):
    return 1 / (1 + np.exp(-beta * x))

def sigmoid_diff(y, beta):
    return beta * y * (1 - y)

def tanh(x, beta):
    return np.tanh(beta * x)

def tanh_diff(y, beta):
    return beta * (1 - y**2)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
    W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size + 1))
    return W1, W2

def forward(x, W1, W2, beta):
    net_h = np.dot(W1, x)
    y_h = tanh(net_h, beta)
    y_h_b = np.concatenate(([1], y_h))
    net_o = np.dot(W2, y_h_b)
    y_o = sigmoid(net_o, beta)
    return y_h, y_h_b, y_o

def backward(x, y_h, y_h_b, y_o, d, W2, beta):

    e = d - y_o

    delta_o = e * sigmoid_diff(y_o, beta)

    W2_no_bias = W2[:,1:]
    delta_h = tanh_diff(y_h, beta) * np.dot(W2_no_bias.T, delta_o)
    return delta_o, delta_h


def train_sample(xx, d, eta, beta, epochs):

    W1, W2 = initialize_weights(3, 2, 1)
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for xi, di in zip(xx, d):

            y_h, y_h_b, y_o = forward(xi, W1, W2, beta)

            delta_o, delta_h = backward(xi, y_h, y_h_b, y_o, di, W2, beta)

            W2 += eta * delta_o * y_h_b.reshape(1, -1)
            W1 += eta * delta_h.reshape(-1,1) * xi.reshape(1, -1)

            total_error += 0.5 * (di - y_o)**2
        errors.append(total_error[0])

        if total_error == 0:
            break
    return W1, W2, errors


def train_epoch(xx, d, eta, beta, epochs):

    W1, W2 = initialize_weights(3, 2, 1)
    errors = []
    for epoch in range(epochs):
        total_error = 0
        grad_W1 = np.zeros_like(W1)
        grad_W2 = np.zeros_like(W2)
        for xi, di in zip(xx, d):

            y_h, y_h_b, y_o = forward(xi, W1, W2, beta)

            delta_o, delta_h = backward(xi, y_h, y_h_b, y_o, di, W2, beta)

            grad_W2 += delta_o * y_h_b.reshape(1, -1)
            grad_W1 += delta_h.reshape(-1,1) * xi.reshape(1, -1)

            total_error += 0.5 * (di - y_o)**2

        W2 += eta * grad_W2
        W1 += eta * grad_W1
        errors.append(total_error[0])

        if total_error == 0:
            break
    return W1, W2, errors

eta = 0.1
beta = 1  
epochs = 100000 

# Trening z aktualizacją wag po każdej próbce
W1_sample, W2_sample, errors_sample = train_sample(xx, d, eta, beta, epochs)
W1_epoch, W2_epoch, errors_epoch = train_epoch(xx, d, eta, beta, epochs)

# Wykres błędu
plt.figure(figsize=(12, 6))
plt.plot(errors_sample, label='Aktualizacja po próbce')
plt.plot(errors_epoch, label='Aktualizacja po epoce')
plt.xlabel('Epoka')
plt.ylabel('Błąd')
plt.title('Zmiany wartości błędu podczas uczenia')
plt.legend()
plt.grid(True)
plt.show()

def classify_output(y_o):
    if y_o > 0.9:
        return 1
    elif y_o < 0.1:
        return 0
    else:
        return None  

# Testowanie sieci po treningu
def test_network(xx, W1, W2, beta):
    for xi in xx:
        y_h, y_h_b, y_o = forward(xi, W1, W2, beta)
        y_pred = classify_output(y_o)
        print(f'Wejście: {xi[1:]}, Wyjście sieci: {y_o[0]:.4f}, Klasa: {y_pred}')

print("\nTestowanie sieci (aktualizacja po próbce):")
test_network(xx, W1_sample, W2_sample, beta)

print("\nTestowanie sieci (aktualizacja po epoce):")
test_network(xx, W1_epoch, W2_epoch, beta)
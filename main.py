import numpy as np

import mnist_loader
import network

# Загружаем данные MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создаем нейронную сеть
net = network.Network([784, 30, 10])

# Выводим количество нейронов в каждом слое
for i in range(net.num_layers):
    print('Количество нейронов в слое', i + 1, ':', net.sizes[i])  # выводим количество нейронов в каждом слое

# Выводим веса и смещения нейронной сети до обучения (до градиентного спуска)
for i in range(net.num_layers - 1):
    print('\nWeight', i + 1, ':')
    print(np.round(net.weights[i], 2))
    print('\nBias', i + 1, ':')
    print(np.round(net.biases[i], 2))

# Обучаем нейронную сеть
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)

# Сохраняем нейронную сеть
net.save('mnist_net.json')

# Загружаем нейронную сеть
net.load('mnist_net.json')

# Выводим веса и смещения нейронной сети после обучения (после градиентного спуска)
for i in range(net.num_layers - 1):
    print('\nWeight', i + 1, ':')
    print(np.round(net.weights[i], 2))
    print('\nBias', i + 1, ':')
    print(np.round(net.biases[i], 2))

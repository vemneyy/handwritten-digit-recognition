import mnist_loader
import network
import numpy as np

# Загружаем данные MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создаем нейронную сеть
net = network.Network([784, 60, 30, 10])

for i in range(net.num_layers):
    print('Количество нейронов в слое', i, ':', net.sizes[i])

for i in range(net.num_layers - 1):
    print('Weight_', i + 1, ':')
    print(np.round(net.weights[i], 2))
    print('Bias_', i + 1, ':')
    print(np.round(net.biases[i], 2))


# Обучаем нейронную сеть
net.sgd(training_data, 30, 20, 3.0, test_data=test_data)





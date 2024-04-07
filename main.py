import mnist_loader
import network
import numpy as np

# Загружаем данные MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создаем нейронную сеть
net = network.Network([784, 30, 10])

for i in range(net.num_layers):
    print('Количество нейронов в слое', i + 1, ':', net.sizes[i])

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





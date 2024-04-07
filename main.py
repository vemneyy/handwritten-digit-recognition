import os
import mnist_loader
import network

# Загружаем данные MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создаем нейронную сеть
net = network.Network([784, 30, 10])

# Обучаем нейронную сеть
net.sgd(training_data, 30, 10, 5.0, test_data=test_data)

# Оцениваем прогресс в обучении
print('Точность:', net.evaluate(test_data) / 10000)

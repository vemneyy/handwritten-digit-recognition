import numpy as np


class Network(object):  # используется для описания нейронной сети
    def __init__(self, sizes):  # конструктор класса
        # self – указатель на объект класса
        # sizes – список размеров слоев нейронной сети
        self.num_layers = len(sizes)  # задаем количество слоев нейронной сети
        self.sizes = sizes  # задаем список размеров слоев нейронной сети
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x) for x, y in
                        zip(sizes[:-1], sizes[1:])]  # задаем случайные начальные веса связей


net = Network([2, 3, 1])  # создаем нейронную сеть из трех слоев

print('Сеть net:')
print('Количетво слоев:', net.num_layers)

for i in range(net.num_layers):
    print('Количество нейронов в слое', i, ':', net.sizes[i])

for i in range(net.num_layers - 1):
    print('W_', i + 1, ':')
    print(np.round(net.weights[i], 2))
    print('b_', i + 1, ':')
    print(np.round(net.biases[i], 2))


def sigmoid(z):  # определение сигмоидальной функции активации
    return 1.0 / (1.0 + np.exp(-z))

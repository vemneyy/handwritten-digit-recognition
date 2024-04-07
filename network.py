import random

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

    def feedforward(self, a):  # прямое распространение сигнала
        # self – указатель на объект класса
        # a – входной вектор сигнала
        # Возвращает выходной вектор сигнала
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data):
        test_data = list(test_data)  # создаем список объектов тестирующей выборки
        n_test = len(test_data)  # вычисляем длину тестирующей выборки
        training_data = list(training_data)  # создаем список объектов обучающей выборки
        n = len(training_data)  # вычисляем размер обучающей выборки

        for j in range(epochs):  # цикл по эпохам
            random.shuffle(training_data)  # перемешиваем элементы обучающей выборки
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # создаем подвыборки

            for mini_batch in mini_batches:  # цикл по подвыборкам
                self.update_mini_batch(mini_batch, eta)  # один шаг градиентного спуска
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))  # смотрим прогресс в обучении

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # список для хранения градиентов
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # список для хранения градиентов

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # послойно вычисляем градиенты

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # обновляем веса и смещения

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # обновляем веса и смещения

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # обновляем веса и смещения

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]  # обновляем веса и смещения

    def cost_derivative(self, output_activations, y):  # Производная функции стоимости
        # output_activations – выходные активации нейронов выходного слоя
        # y – вектор правильных ответов
        # Возвращает вектор частных производных функции стоимости по активациям нейронов выходного слоя
        return output_activations - y

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in
                   self.biases]  # список градиентов dC/db для каждого слоя(первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in
                   self.weights]  # список градиентов dC/dw для каждого слоя(первоначально заполняются нулями)

        # определение переменных
        activation = x  # выходные сигналы слоя (первоначально соответствует входному вектору)
        activations = [x]  # список выходных сигналов по всем слоям (первоначально заполняется входным вектором)

        zs = []  # список активационных потенциалов по всем слоям (первоначально пустой)

        # прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # считаем активационные потенциалы слоя
            zs.append(z)  # добавляем элемент (активационные потенциалы слоя) в конец списка
            activation = sigmoid(z)  # считаем выходные сигналы текущего слоя
            activations.append(activation)  # добавляем элемент (выходные сигналы слоя) в конец списка

        # обратное распространение
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # считаем меру влияния
        # нейронов выходного слоя L на функцию стоимости (BP1) и производную сигмоидальной функции активации (BP2)

        nabla_b[-1] = delta  # градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градиент dC/dw для слоя L (BP4)

        for j in range(2, self.num_layers):
            z = zs[
                -j]  # активационные потенциалы l-го слоя
            # (двигаемся по списку активационных потенциалов в обратном направлении)

            sp = sigmoid_prime(z)  # считаем сигмоидальную функцию от активационных потенциалов l-го слоя

            delta = np.dot(self.weights[-j + 1].transpose(),
                           delta) * sp  # считаем меру влияния нейронов l-го слоя на функцию стоимости

            nabla_b[-j] = delta  # градиент dC/db для l-го слоя (BP3)
            nabla_w[-j] = np.dot(delta, activations[-j - 1].transpose())  # градиент dC/dw для l-го слоя (BP4)

        return nabla_b, nabla_w

    def evaluate(self, test_data):  # Оценка прогресса в обучении
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]  # сравниваем результаты работы нейронной сети с правильными ответами

        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):  # определение сигмоидальной функции активации
    # z – взвешенная сумма входов нейрона
    # Возвращает значение сигмоидной функции активации
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # Производная сигмоидальной функции
    # z – взвешенная сумма входов нейрона
    # Возвращает значение производной сигмоидной функции активации
    return sigmoid(z) * (1 - sigmoid(z))


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

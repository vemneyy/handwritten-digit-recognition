import json  # импортируем модуль json
import os  # импортируем модуль os
import random  # импортируем модуль random

import matplotlib.pyplot as plt
import numpy as np  # импортируем библиотеку numpy для работы с массивами и матрицами numpy и линейной алгебры

# Убедитесь, что папка logs существует, если нет - создайте ее
os.makedirs("logs", exist_ok=True)


# Определение функции стоимости
def cost_derivative(output_activations, y):
    return output_activations - y  # возвращает вектор частных производных функции стоимости по активациям


def sigmoid(z):  # определение сигмоидальной функции активации
    # z – взвешенная сумма входов нейрона
    # Возвращает значение сигмоидной функции активации
    return 1.0 / (1.0 + np.exp(-z))  # возвращает значение сигмоидной функции активации


def sigmoid_prime(z):  # Производная сигмоидальной функции
    # z – взвешенная сумма входов нейрона
    # Возвращает значение производной сигмоидной функции активации
    return sigmoid(z) * (1 - sigmoid(z))  # возвращает значение производной сигмоидной функции активации


'''
Класс Network представляет собой нейронную сеть прямого распространения. Инициализация класса принимает список sizes, 
который задает количество нейронов в соответствующих слоях. Например, если список sizes равен [2, 3, 1], то это
означает, что сеть имеет два нейрона входного слоя, три нейрона в скрытом слое и один нейрон в выходном слое. Веса и
смещения для сети инициализируются случайными значениями, используя стандартное нормальное распределение средним 0 и
стандартным отклонением 1. Обратите внимание, что первый слой считается слоем ввода, и он не имеет смещений, поскольку
смещения используются только в вычислении выходных значений из нейронов. Метод feedforward(self, a) возвращает выходной
вектор сигнала, когда a – входной вектор сигнала. Метод sgd(self, training_data, epochs, mini_batch_size, eta, test_data)
обучает нейронную сеть с использованием стохастического градиентного спуска. training_data – список кортежей (x, y),
где x – входной вектор сигнала, y – правильный ответ. epochs – количество эпох обучения. mini_batch_size – размер
подвыборки. eta – скорость обучения. test_data – список кортежей (x, y), где x – входной вектор сигнала, y – правильный
ответ. Если test_data предоставлен, то сеть будет оцениваться после каждой эпохи, и промежуточные результаты будут
выводиться. Метод update_mini_batch(self, mini_batch, eta) обновляет веса и смещения нейронной сети, применяя градиентный
спуск с использованием обратного распространения к одной подвыборке. Метод backprop(self, x, y) возвращает кортеж (nabla_b,
nabla_w), представляющий градиент функции стоимости по смещениям и весам. Метод evaluate(self, test_data) возвращает
количество правильных ответов нейронной сети на тестовых данных test_data. Метод save(self, param) сохраняет параметры
нейронной сети в файл. Метод load(self, param) загружает параметры нейронной сети из файла.
'''


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
            a = sigmoid(np.dot(w, a) + b)  # считаем выходные сигналы слоя
        return a  # возвращает выходной вектор сигнала

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data):
        test_data = list(test_data)  # преобразуем тестовую выборку в список
        n_test = len(test_data)  # количество элементов в тестовой выборке
        training_data = list(training_data)  # преобразуем обучающую выборку в список
        n = len(training_data)  # количество элементов в обучающей выборке

        for j in range(epochs):  # цикл по эпохам обучения
            random.shuffle(training_data)  # перемешиваем элементы обучающей выборки
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # создаем подвыборки из обучающей выборки

            for mini_batch in mini_batches:  # цикл по подвыборкам
                self.update_mini_batch(mini_batch, eta)  # один шаг градиентного спуска

            if test_data:
                epoch_accuracy = self.evaluate(test_data) / n_test * 100  # оценка прогресса в обучении в процентах
                print("Epoch {0}: {1} / {2}".format(j + 1, self.evaluate(test_data), n_test))  # выводим результаты

                # Сохранение результатов в файл
                with open(f'logs/epoch_{j + 1}.txt', 'w') as f:
                    f.write(f'Epoch {j + 1}: {self.evaluate(test_data)} / {n_test}\n')  # записываем результаты в файл
                    f.write(f'Accuracy: {epoch_accuracy}%\n')  # записываем результаты в файл
                    for i in range(n_test):
                        f.write(
                            f'Prediction: {np.argmax(self.feedforward(test_data[i][0]))} Answer: {test_data[i][1]}\n')
                        # записываем результаты в файл

            else:
                print("Epoch {0} complete".format(j))  # выводим результаты

        print('\nAccuracy:', self.evaluate(test_data) / 100, '%')  # выводим результаты в конце обучения

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # список для хранения градиентов dC/db
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # список для хранения градиентов dC/dw

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # послойно вычисляем градиенты dC/db и dC/dw для x, y

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # обновляем веса и смещения

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # обновляем веса и смещения

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # обновляем веса и смещения

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]  # обновляем веса и смещения

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
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])  # считаем меру влияния
        # нейронов выходного слоя L на функцию стоимости (BP1) и производную сигмоидальной функции активации (BP2)

        nabla_b[-1] = delta  # градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градиент dC/dw для слоя L (BP4)

        for j in range(2, self.num_layers):
            z = zs[-j]  # активационные потенциалы j-го слоя (BP1)
            # (двигаемся по списку активационных потенциалов в обратном направлении)

            sp = sigmoid_prime(z)  # считаем сигмоидальную функцию от активационных потенциалов j-го слоя (BP2)

            delta = np.dot(self.weights[-j + 1].transpose(),
                           delta) * sp  # считаем меру влияния нейронов j-го слоя на функцию стоимости (BP1)

            nabla_b[-j] = delta  # градиент dC/db для l-го слоя (BP3)
            nabla_w[-j] = np.dot(delta, activations[-j - 1].transpose())  # градиент dC/dw для l-го слоя (BP4)

        return nabla_b, nabla_w  # возвращает градиенты dC/db и dC/dw для всех слоев нейронной сети

    def evaluate(self, test_data):  # Оценка прогресса в обучении нейронной сети
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # сравниваем результаты работы нейронной сети с правильными ответами из тестовой выборки

        return sum(int(x == y) for (x, y) in test_results)  # возвращает количество правильных ответов

    def save(self, param):
        # Сохранение параметров нейронной сети в файл
        with open(param, 'w') as f:
            json.dump({'sizes': self.sizes,
                       'weights': [w.tolist() for w in self.weights],
                       'biases': [b.tolist() for b in self.biases]},
                      f)  # записываем параметры нейронной сети в файл в формате JSON

    def load(self, param):
        # Загрузка параметров нейронной сети из файла
        with open(param, 'r') as f:
            data = json.load(f)
        self.sizes = data['sizes']  # задаем размеры слоев нейронной сети
        self.weights = [np.array(w) for w in data['weights']]  # задаем веса связей
        self.biases = [np.array(b) for b in data['biases']]  # задаем смещения
        self.num_layers = len(self.sizes)  # задаем количество слоев нейронной сети

    def recognize(self, test_data, i):
        # Распознавание цифры
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # сравниваем результаты работы нейронной сети с правильными ответами из тестовой выборки

        print(f'Prediction: {test_results[i][0]} Answer: {test_results[i][1]}')
        # выводим результаты распознавания

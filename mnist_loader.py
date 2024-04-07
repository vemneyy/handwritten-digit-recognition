import gzip  # библиотека для сжатия и распаковки файлов gzip и gunzip
import pickle  # библиотека для сохранения и загрузки сложных объектов Python

import numpy as np


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')  # открываем сжатый файл gzip в двоичном режиме
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')  # загружам таблицы из файла
    f.close()  # закрываем файл

    return training_data, validation_data, test_data  # возвращаем таблицы


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()  # инициализация наборов данных в формате MNIST
    training_inputs = [np.reshape(x, (784, 1)) for x in
                       tr_d[0]]  # преобразование массивов размера 1 на 784 к массивам размера 784 на 1
    training_results = [vectorized_result(y) for y in
                        tr_d[1]]  # представление цифр от 0 до 9 в виде массивов размера 10 на 1
    training_data = zip(training_inputs, training_results)  # формируем набор обучающих данных из пар (x, y)
    validation_inputs = [np.reshape(x, (784, 1)) for x in
                         va_d[0]]  # преобразование массивов размера 1 на 784 к массивам размера 784 на 1
    validation_data = zip(validation_inputs, va_d[1])  # формируем набор данных проверки из пар (x, y)
    test_inputs = [np.reshape(x, (784, 1)) for x in
                   te_d[0]]  # преобразование массивов размера 1 на 784 к массивам размера 784 на 1
    test_data = zip(test_inputs, te_d[1])  # формируем набор тестовых данных из пар (x, y)

    return training_data, validation_data, test_data # возвращаем наборы данных обучения, проверки и тестирования


def vectorized_result(j):
    e = np.zeros((10, 1))  # создаем массив нулей размера 10 на 1
    e[j] = 1.0  # присваиваем элементу с индексом j значение 1
    return e  # возвращаем массив

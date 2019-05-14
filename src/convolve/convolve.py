import numpy as np


def convolve2d(x, h):
    """

    :param x: signal (image, etc.)
    :param h: filter
    :return:
    """
    N1, N2 = x.shape
    K1, K2 = h.shape

    y = np.zeros(shape=x.shape)

    for n1 in range(N1):
        print('Convolving...' + str(1.0 * n1 / N1) + ' %')
        for n2 in range(N2):

            for k1 in range(K1):
                for k2 in range(K2):

                    if n1 - k1 >= 0 and n2 - k2 >= 0:
                        y[n1, n2] += h[k1, k2] * x[n1 - k1, n2 - k2]
    return y


def generate_impluse_filter():
    """
    This filter does not make any effect on the original image
    :return:
    """
    filter = np.zeros(shape=(10, 10))
    filter[0, 0] = 1
    return filter


def generate_gaussian_matrix_2d_filter(std=6):
    """
    Blur the image
    :param std: is proportional to the level of blur
    :return:
    """
    gaussian_matrix = np.zeros(shape=(10, 10))
    z = 1 / (2 * 3.14 * std * std)

    for i in range(10):
        for j in range(10):
            gaussian_matrix[i, j] = z * np.exp(-(i ** 2 + j ** 2) / (2 * std * std))

    gaussian_matrix /= gaussian_matrix.sum()
    return gaussian_matrix

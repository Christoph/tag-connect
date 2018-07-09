import numpy as np


def normalize(matrix):
    return matrix/matrix.sum()

def forward(E, T, emission, init):
    alpha = []

    a = normalize(np.diag(E[:, int(emission[0])-1].getA1()) * init)
    alpha.append(a)

    for i in range(1, len(emission)):
        a = normalize(np.diag(E[:, int(emission[i])-1].getA1()) * (T.transpose() * alpha[i-1]))
        alpha.append(a)

    return alpha

def backward(E, T, emission, back_init):
    beta = np.zeros(len(emission), dtype=object).tolist()

    a = normalize(T * np.diag(E[:, int(emission[-1])-1].getA1()) * back_init)
    beta[-1] = a

    for i in range(len(emission)-2, -1, -1):
        a = normalize(T * np.diag(E[:, int(emission[i])-1].getA1()) * beta[i+1])
        beta[i] = a

    return beta

def combine(f, b):
    gamma = []

    for i in range(0, len(f)):
        ga = normalize(np.multiply(f[i], b[i]))
        gamma.append(ga)

    return gamma

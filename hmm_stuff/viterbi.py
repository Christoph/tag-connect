import numpy as np


def viterbi(T, E, emission, init):

    delta = []
    backtrace = []

    d1 = np.multiply(E[:, 0].T, init)
    delta.append(d1)
    backtrace.append(np.argmax(d1))

    for position, item in enumerate(emission):
        if position >= 1:
            temp = np.multiply(np.multiply(E[:, int(item)].T, T), delta[position-1].T)
            d = np.amax(temp, axis=0)
            backtrace.append(np.argmax(d))
            delta.append(d)

    return delta, backtrace

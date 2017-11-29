import numpy as np
import matplotlib.pyplot as plt

# Saw Tooth signal


def sawtooth_signal():

    x = np.linspace(-20, 20, 1000)

    y = np.hstack((x, x, x, x, x))

    plt.plot(y)
    plt.show()
    plt.close("all")

    return y

sawtooth_signal()
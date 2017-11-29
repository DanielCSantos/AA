import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn.metrics




def main():
    data = pickle.load(open('Ficha3/mnist_small.p', 'rb'))

    data_image = data["train1"][:, 0:400]
    exa(data_image)
    exb(data_image)
    exc(data_image)
    exd(data_image)
    exe(data_image)
    exf(data_image)
    exg(data_image)
    exh(data_image)
    exj(data_image)
    exi(data_image)



def exa(data_image):

    # i) False
    matrix_cov = np.cov(data_image)
    print("i) The dimension of the matrix's cov is: \n {}".format(matrix_cov.shape))

    # ii) True
    print("ii) The determinant from the Covariance matrix is:\n{}".format(np.linalg.det(matrix_cov)))

    # iii) False

    # iv) False

def exb(data_image):
    matrix_cov = np.cov(data_image)
    v, w = np.linalg.eig(matrix_cov)
    idx = np.argsort(-v)
    v = v[idx]

    v = v.real
    v = v / np.sum(v)

    w = w[:, idx]
    w = w[:, v >= 1 / np.float(np.power(10, 10))]
    w = w.real

    l = np.cumsum(v)
    k = np.sum(l <= 0.86)
    # print ("b)\n{}".format(np.sum(l <= 0.86)))
    w = w[:, 0:k].real
    mx = np.mean(data_image, axis=1)
    xn = data_image - mx[:, np.newaxis]

    y = np.dot(np.transpose(w), data_image)

    # transformation of the digit/ reconstruction
    xr = np.dot(w, y)





    # print("c) {}".format(sklearn.metrics.mean_squared_error(data_image[:, 148], xr)))
    print("b)")
    print("b) {}".format(sklearn.metrics.mean_absolute_error(data_image[:, 59], xr[:, 59])))



def exc(data_image):

    # iii)
    matrix_cov = np.cov(data_image)
    v, w = np.linalg.eig(matrix_cov)
    v = v.real
    idx = np.argsort(-v)
    v = v[idx]

    w = w[:, idx]
    w = w[:, v >= 1 / np.float(np.power(10, 10))]
    w = w.real

    w = w[:, 0:12]
    mx = np.mean(data_image, axis=1)
    xn = data_image - mx[:, np.newaxis]

    y = np.dot(np.transpose(w), data_image)

    # transformation of the digit/ reconstruction
    xr = np.dot(w, y)

    # v = v / np.sum(v)
    #
    #
    # l = np.cumsum(v)

    w = w[:, 0:12]
    mx = np.mean(data_image, axis=1)
    xn = data_image - mx[:, np.newaxis]

    y = np.dot(np.transpose(w), data_image)



    # transformation of the digit/ reconstruction
    xr = np.dot(w, y)
    # xr = xr + mx[:,np.newaxis]

    # xr = xr - xr.min()
    # xr = 255.0 * xr/xr.max()

    # print("c) {}".format(sklearn.metrics.mean_squared_error(data_image[:, 148], xr)))
    print("c)")
    print("c) {}".format(sklearn.metrics.mean_squared_error(data_image[:, 148], xr[:, 148])))


def exd(data_image):
    # data = data_image[data_image]
    matrix_cov = np.cov(data_image)
    v, w = np.linalg.eig(matrix_cov)
    idx = np.argsort(-v)
    v = v[idx]

    w = w[:, idx]
    w = w[:, v >= 1/np.power(1, 10)]

    v = v.real
    v = v / np.sum(v)

    l = np.cumsum(v)



    plt.plot(l)
    plt.show()

    # iii) True
    print("d) iii) The minimum number of principal components in order for the projected data"
          " to be above 70% of the total data variance is:\n{}".format(np.sum(l <= 0.71)))




def exe(data_image):
    constant_values = 0
    # i)True
    for i in range(0, 256):
        constant_values = np.add(constant_values, np.sum(np.sum(data_image == i, axis=1) == 400))
    print("e) i) The total amout of pixels that do not change along all images is:\n{}".format(np.round(constant_values, decimals=2)))

    # ii)False
    print("e) ii) {}".format(np.corrcoef(data_image[679, :], data_image[418, :])))

    # iii) False

    # iv) False

def exf(data_image):
    # pass
    # i)False
    print("f) i) The norm of the mean vector data is: \n {}".
          format(np.round(np.linalg.norm(np.mean(data_image, axis=1)), decimals=0)))

    # ii) True
    print("ii) The standard deviation of the dimension 210 is:\n {}".
          format(np.round(np.std(data_image[210]), decimals=0)))

    # iii) False

    # iv) False

def exg(data_image):

    number_zero_pixels = np.sum(np.sum(data_image == 0, axis=1) == 400)
    print("g) The number of pixels that doesn't change from 0 on all the images is: \n{}".format(number_zero_pixels))

    # i)True
    # ii)False
    # iii)False
    # iv)False



def exh(data_image):
    # pass
    matrix_cov = np.cov(data_image)

    print("h) i) The number of ______ superior to  10^-10 is:\n{}".format(np.sum(np.linalg.eigvals(matrix_cov) > 10**-10)))
    # i) False
    # ii) False
    # iii) True
    # iv) False


def exi(data_image):

    x1 = np.sum(data_image[:, 214])
    #
    x2 = np.sum(data_image[:, 289])
    #
    x3 = np.sum(data_image[:, 320])
    #
    x4 = np.sum(data_image[:, 334])


    # # i)
    print("i) i)Internal product of x1, x2 is:\n {}".format(np.round(np.inner(x1, x2), decimals=0)))
    #
    # # ii)
    print("i) ii)Internal product of x1, x2 is:\n {}".format(np.round(np.inner(x3, x4), decimals=0)))



def exj(data_image):

    digit = np.reshape(data_image[:,228], (28, 28))

    plt.imshow(255-digit, interpolation="none",  cmap="gray")
    plt.show()
    # i) False
    print("j) i) The inverted image of the 229th digit of the sub data is False")

    # ii) False
    matrix_cov = np.cov(data_image)
    matrix_value,matrix_vec = np.linalg.eig(matrix_cov)
    matrix_vec = matrix_vec.real
    digit_cov = np.reshape(matrix_vec[:,3], (28,28))

    # iii) False
    # iv) False

    plt.imshow(digit_cov, interpolation="none", cmap="gray")
    plt.show()








main()
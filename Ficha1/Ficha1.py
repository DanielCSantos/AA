import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    ex1()
    ex2()
    ex3()

def ex1():

    data = pickle.load(open('A32078_Q002_data.p', 'rb'))


    data_matrix = data["dados"]
    data_classes = data["trueClass"]

    # ------Question 1-------
    # Point A
    mean_data = np.mean(data_matrix, axis=1)

    print("a)The mean of all data is:\n{}\n").format(np.round(mean_data, decimals=2))
    # print("")
    # print("a) Mean point: {}").format(np.round(mean_point, decimals=2))

    # print(np.mean(data["dados"], axis=1))

    # Point B
    cov2 = np.cov(data_matrix[:, data_classes == 2])
    print("b)Covariance of all data of Class 2 is:\n{}\n").format(np.round(cov2, decimals=2))

    # Point C
    cov_all = np.cov(data_matrix)
    print("c)Covariance of all data is:\n{}\n").format(np.round(cov_all, decimals=2))

    # Point D
    mean_data1 = np.mean(data_matrix[:, data_classes == 1], axis=1)
    print("d)The mean of all data of Class 1 is:\n{}\n").format(np.round(mean_data1, decimals=2))

    # Point E
    data_matrix_totalsize = data_matrix.size
    data_matrix_size2 = data_matrix[:, data_classes == 2].size
    percentage2 = float(data_matrix_size2)/data_matrix_totalsize * 100

    print("e) The percentage of points in Class 2 is:\n{}%\n").format(np.round(percentage2, decimals=2))

    # Point F
    mean_data2 = np.mean(data_matrix[:, data_classes == 2], axis=1)
    mean_data3 = np.mean(data_matrix[:, data_classes == 3], axis=1)

    distance = np.sqrt(np.power(mean_data2[0] - mean_data3[0], 2) + np.power(mean_data2[1] - mean_data3[1], 2))
    print("f) The distance between the means of Class 2 and 3 is:\n{}\n").format(np.round(distance, decimals=2))


def ex2():

    # Several ways to calculate Covariance of a matrix(example: matrix x dimension:(2x10)
    x = np.array([[1, 2, 3, 5, 7, 6, 1, 1, 8, 9],
                 [1, 5, 6, 7, 8, 2, 3, 5, 8, 7]])
    print("X matrix (2x10) is:\n")
    print(x)
    print("")
    # 1st way
    cov1 = np.cov(x)


    # 2nd way
    yn = (x.T - np.mean(x, 1)).T
    cov2 = np.dot(yn, yn.T)/9.

    # print("Expected Results:{}\n"+cov1+"\n")
    print("Expected Results:\n{}\n").format(cov1, decimals=2)
    print ("{}\n").format(cov2, decimals=2)

    # Point A
    cxa = np.cov(x.T, rowvar=False, ddof=0)
    print("a)\n{}\n").format(cxa, decimals=2)

    # Point B
    mx = np.mean(x, axis=1)
    # Gives an Error because we are adding a new dimension while trying to perform matrix operations with different
    # shapes

    # xn = x.T - mx[:, np.newaxis]

    # ctmp = np.dot(xn, xn.T)
    # cxb = ctmp / (xn.shape[1] - 1)
    # print("teste")
    # print("b)\n{}\n").format(cxb, decimals=2)
    # Point C

    mx = np.mean(x, axis=1)
    xn = (x.T - mx).T
    ctmp = np.dot(xn, xn.T)
    cx = ctmp / (x.shape[1] - 1)
    print("c\n{}\n").format(np.round(cx, decimals=2))

    # Point D
    cx = np.cov(x, rowvar=False)
    print("d)\n{}\n").format(np.round(cx, decimals=2))

    # Point E
    mx = np.mean(x, axis=1)
    mx = mx[:, np.newaxis]
    ctmp = np.dot(x, x.T) / (x.shape[1] - 1)
    cx = ctmp - np.dot(mx, mx.T)
    print("e)\n{}\n").format(np.round(cx, decimals=2))

    # Point F
    ctmp = np.mean(x ** 2, axis=1)
    ctmp = ctmp[:, np.newaxis]
    cx = np.dot(ctmp, ctmp.T)
    print("f)\n{}\n").format(np.round(cx, decimals=2))


def ex3():

    square = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    # square.astype(float)
    scalematrix = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 1]])
    rotationmatrix = np.array([[np.cos(np.deg2rad(-60)), -1 * np.sin(np.deg2rad(-60)), 0],
                               [np.sin(np.deg2rad(-60)), np.cos(np.deg2rad(-60)), 0], [0, 0, 1]])
    translmatrix = np.array([[1, 0, -3], [0, 1, 0], [0, 0, 1]])
    print("square")
    print(square)


    print("Scaling matrix")
    print(scalematrix)

    print("result from scaling:")
    transf1 = np.dot(square, scalematrix)
    print(transf1)


    print("translation matrix:\n{}").format(translmatrix)

    transf3 = np.dot(translmatrix, rotationmatrix)



    print("Rotation matrix\n{}").format(np.round(rotationmatrix, decimals=2))


    transf2 = np.dot(transf1, rotationmatrix)
    print("result from adding rotation to scaled square:\n{}").format((np.round(transf2, decimals=2)))

    combomatrix = np.dot(translmatrix, rotationmatrix)
    combomatrix = np.dot(combomatrix, scalematrix)
    print("Combo matrix:\n{}").format(np.round(combomatrix, decimals=2))




    #
    #
    # print("square1")
    # print(square)
    # print(square[0])
    # print(square[0][0])
    # print (square[1][0])
    #
    # figure = plt.figure()
    # print("test scaling")
    # scaling(2, 4, square, figure)
    #
    #
    # draw_rectan1(square, figure, None)
    #
    # plt.plot()
    # plt.xticks(np.arange(-6, 7, 1.0))
    # plt.yticks(np.arange(-6, 7, 1.0))
    # plt.grid()
    # plt.show()
    #
    # # transla(-3, 0, square)
    #
    # # rotation(square, 60)
    # transla(-1, 0, square)
    #
    # figure = plt.figure()
    # draw_rectan1(square, figure, None)
    #
    # plt.plot()
    # plt.xticks(np.arange(-6, 7, 1.0))
    # plt.yticks(np.arange(-6, 7, 1.0))
    # plt.grid()
    # plt.show()
    # # raw_input()
    # plt.close("all")
    #
    # # rotation(square, 60)
    #
    # figure = plt.figure()
    # draw_rectan1(square, figure, -60)
    #
    # transla(+5, +2, square)
    #
    # plt.plot()
    # plt.xticks(np.arange(-6, 7, 1.0))
    # plt.yticks(np.arange(-6, 7, 1.0))
    # plt.grid()
    # plt.show()
    # plt.close()



# Translation of Square
def transla(moveX, moveY, square):
    print("")
    # transla_matrix = np.array([[1, 0, 0], [0, 1, 0], [moveX, moveY, 1]])
    # print(transla_matrix)
    # print("square")
    # print(square)
    # print(square[0])
    # print(square.shape)
    # print("translation matrix")
    # print(transla_matrix)
    # square = np.dot(square, transla_matrix)
    # # for i in range(0, 4):
    # #     square[i] = np.dot(square[i], transla_matrix)
    # #     # square[i][0] = square[i][0] + moveX
    # #     # square[i][1] = square[i][1] + moveY
    #
    # print(square)





def draw_rectan1(square, figure,angle):
    print("drawing rectangle")
    newsquare = np.array([])
    # for in range(0,4):
    #    newsquare = square[0]

    x = square[0][0]
    y = square[0][1]
    width = square[1][0] - x
    height = square[2][1] - y
    print(height)
    print(width)
    aux = figure.add_subplot(111, aspect="equal")
    if angle is None:
        aux.add_patch(patches.Rectangle((x, y), width, height))
    else:
        aux.add_patch(patches.Rectangle((x, y), width, height, angle))
        # aux.add_patch(patches.Rectangle((x, y), width, height))

    # aux.axis([-6, 6, -6, 6])


    # plt.plot()
    # plt.xticks(np.arange(-6, 7, 1.0))
    # plt.yticks(np.arange(-6, 7, 1.0))
    # plt.grid()
    # plt.show()
    # print(square[1][0])

#erros p1= 1e3
# int(1000*p1)
def scaling(scaleX, scaleY, square, figure):
    # a = np.array([[0, 0], [2, 1], [2, 4]])
    # print(a[:, [0]]*2)
    a = square
    square.astype(float)
    for i in range(0, 4):
        square[i][0] = square[i][0] * scaleX
        square[i][1] = square[i][1] * scaleY

    # print(a)
    # print(a[:, 0:1])
    print(square)

def rotation(square, degree):
    initPointX = square[0][0]
    initPointY = square[0][1]
    print("square")
    print(square)
    print("rotate")
    print(square[1][0])
    print("centerX")
    # centerX = square[0][0]+(square[1][0] - square[0][0])/2
    centerX = (square[0][0]+square[1][0])/2
    centerY = (square[0][1]+square[2][1])/2
    print("centerX")
    print(centerX)
    print("centerY")
    print (centerY)
    # width =
    print("rotation test")
    print(np.sin(degree))

    for i in range(0, 4):
        a = square[i][0] = float(square[i][0]) * float(np.cos(degree)) - float(square[i][1]) * float(np.sin(degree))
        print(a)
        square[i][0] = float(square[i][0]) * float(np.cos(degree)) - float(square[i][1]) * float(np.sin(degree))
        square[i][1] = float(square[i][0]) * float(np.sin(degree)) + square[i][1] * float(np.cos(degree))

    print(square)



main()

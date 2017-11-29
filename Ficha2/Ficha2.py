import pickle
import numpy as np
import numpy.linalg as la
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors.nearest_centroid import NearestCentroid as NC


def main():
    ex03()


def ex01():

    # data02 = pickle.load(open('A32078_Q002_data.p', 'rb'))
    # data03 = pickle.load(open('A32078_Q003_data.p', 'rb'))
    # data04 = pickle.load(open('A32078_Q004_data.p', 'rb'))

    iris = datasets.load_iris()

    X = iris['data']
    true_class = iris['target']

    # code from slides
    X = np.transpose(X)

    m0 = np.mean(X[:, true_class == 0], axis=1)
    m1 = np.mean(X[:, true_class == 1], axis=1)
    m2 = np.mean(X[:, true_class == 2], axis=1)

    X0 = X - m0[:, np.newaxis]
    X1 = X - m1[:, np.newaxis]
    X2 = X - m2[:, np.newaxis]

    D0 = np.sqrt(np.sum(X0 * X0, axis=0))
    D1 = np.sqrt(np.sum(X1 * X1, axis=0))
    D2 = np.sqrt(np.sum(X2 * X2, axis=0))

    D = np.vstack((D0, D1, D2))

    est_class = np.argmin(D, axis=0)

    m0 = m0/la.norm(m0)
    m1 = m1/la.norm(m1)
    m2 = m2/la.norm(m2)
    x = X/la.norm(X, axis=0)

    d0 = np.dot(np.transpose(m0), X)
    d1 = np.dot(np.transpose(m1), X)
    d2 = np.dot(np.transpose(m2), X)
    D = 1.0 - np.vstack((d0, d1, d2))
    estClass1 = np.argmin(D, axis=0)

    print("hello")


def ex02():
    data02 = pickle.load(open('A32078_Q002_data.p', 'rb'))
    print("")
    confusion_matrix = metrics.confusion_matrix(data02["trueClass"], data02["estClass"])
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    # A)
    # Number of False positives
    print("The number of False positives is:\n {}".format(FP))

    # B)
    # Probability of correct answers
    correct_answers = np.trace(confusion_matrix)
    probability = np.float(correct_answers) / np.sum(confusion_matrix) * 100

    print("The probability of correct answers is:\n {}".format(np.round(probability, decimals=1)))

    # C)
    # Rate of true Negatives
    rate_tn = np.float(TN) / (TN + FP)

    print("The rate of true Negatives is:\n {}".format(np.round(rate_tn, decimals=3)))

    # D)
    # Rate of False Positives
    rate_fp = np.float(FP) / (FP + TN)

    print("D) The rate of false positives is:\n{}".format(np.round(rate_fp, decimals=3)))

    # E)
    #Probability of the total error

    total_error_probability = np.float((FP + FN)) / (TP + FP + TN + FN) * 100

    print("The probability of the total errors is:\n {}%".format(np.round(total_error_probability, decimals=1)))

    # F)
    # Prior probability of the negative class
    prior_negative_probability = np.float(TN + FP) / np.sum(confusion_matrix) * 100

    print("The prior probability of negatives is:\n{}".format(np.round(prior_negative_probability, decimals=1)))






    print(confusion_matrix)
    test = data02["trueClass"] == 1
    # print(test.count(1))

    print(np.array([["true negatives","false positives"],["false negatives", "true positives"]]))
    print(np.count_nonzero(data02["trueClass"]))
    print(np.trace(confusion_matrix * 1.0)/ np.sum(confusion_matrix)) *100

    print(np.sum(confusion_matrix[0]))
    print(np.sum(confusion_matrix[0][0]) * 1.0 /np.sum(confusion_matrix[0]))

    print(np.sum(confusion_matrix[0][1]) * 1.0 / np.sum(confusion_matrix[0]))

    print(np.sum(confusion_matrix[0][1] + confusion_matrix[1][0]) * 1.0 / np.sum(confusion_matrix)) * 100


def ex03():


    data03 = pickle.load(open('A32078_Q003_data.p', 'rb'))
    data_matrix = data03["dados"]
    data_classes = data03["trueClass"]

    print("cov")
    print(np.cov(data_matrix))


    print(data_matrix.shape)
    print(np.round(data_matrix[:, data_classes == 2].size/(data_matrix.size*1.0), 3))*100
    print("")


    print(np.mean(data_matrix[:, data_classes==1], axis=1))

    print("global mean")
    print(np.mean(data_matrix, axis=1))

    print("---------------Exercise 3 second part-----------------")

    m0 = np.mean(data_matrix[:, data_classes == 0], axis=1)
    m1 = np.mean(data_matrix[:, data_classes == 1], axis=1)
    m2 = np.mean(data_matrix[:, data_classes == 2], axis=1)

    X0 = data_matrix - m0[:, np.newaxis]
    X1 = data_matrix - m1[:, np.newaxis]
    X2 = data_matrix - m2[:, np.newaxis]

    SI0 = np.cov(data_matrix[:, data_classes == 0])
    SI1 = np.cov(data_matrix[:, data_classes == 1])
    SI2 = np.cov(data_matrix[:, data_classes == 2])

    D0 = np.sqrt(np.sum(X0 * np.dot(SI0, X0), axis=0))
    D1 = np.sqrt(np.sum(X1 * np.dot(SI0, X1), axis=0))
    D2 = np.sqrt(np.sum(X2 * np.dot(SI0, X2), axis=0))

    D = np.vstack((D0, D1, D2))

    est_class = np.argmin(D, axis=0)

    confusion_matrix = metrics.confusion_matrix(data_classes, est_class)

    print(confusion_matrix)

    # B)

    # i)

    numerator = np.trace(confusion_matrix)
    denominator = np.sum(confusion_matrix)
    probability = np.float(numerator) / denominator * 100

    print("The probability of correct classifications is:\n {}%".format(np.round(probability, decimals=1)))

    # ii)

    numerator = confusion_matrix[2][0]
    denominator = np.sum(confusion_matrix[2])

    probability = np.float(numerator) / denominator * 100

    print("The probability of points from class 2 being misclassified as class 0:\n"
          " {}%".format(np.round(probability, decimals=1)))


    # iii)

    numerator = confusion_matrix[1][1]
    denominator = np.sum(confusion_matrix[1])

    probability = np.float(numerator) / denominator * 100

    print("The probability of points being correctly classified as class 1:\n{}%".format(np.round(probability, decimals=1)))

    # iV)

    numerator = np.sum(confusion_matrix[1]) - confusion_matrix[1][1]
    denominator = np.sum(confusion_matrix[1])

    probability = np.float(numerator) / denominator * 100

    print("The probability of points of Class 1 being misclassified is:\n{}%".format(np.round(probability, decimals=1)))


def ex04():
    data04 = pickle.load(open('A32078_Q004_data.p', 'rb'))
    # data_matrix = data04["dados"]
    # data_classes = data04["trueClass"]
    # print()
    # data1 = data04["estClass"]
    # print(data1)
    confusion_matrix = metrics.confusion_matrix(data04["trueClass"], data04["estClass"])
    # A)
    # Probability of correctness on class 1
    numerator = np.sum(confusion_matrix[1][1])
    denominator = np.sum(confusion_matrix[1])
    probability = np.float(numerator) / denominator * 100
    print("The probability of correct classification of points from Class 1:\n{}".format(np.round(probability, decimals=1)))

    # B)
    # Total probability of errors
    numerator = np.sum(confusion_matrix) - np.trace(confusion_matrix)
    denominator = np.sum(confusion_matrix)
    probability = np.float(numerator) / denominator * 100
    print("The total probability of errors:\n{}".format(np.round(probability, decimals=1)))

    # C)
    # Probability of class 3 points getting misclassified
    # c = class number
    c = 3
    numerator = np.sum(confusion_matrix[c]) - confusion_matrix[c][c]
    denominator = np.sum(confusion_matrix[c])
    probability = np.float(numerator) / denominator * 100
    print("The probability of class 3 points getting misclassified:\n{}".format(np.round(probability, decimals=1)))

    # D)
    # Prior Probability of Class 3 points
    # C = class number

    c = 3
    numerator = np.sum(confusion_matrix[c])
    denominator = np.sum(confusion_matrix)
    probability = np.float(numerator) / denominator * 100
    print("The prior probability of class 3 points:\n{}".format(np.round(probability, decimals=1)))

    print("matrix confusao")
    print(confusion_matrix)








main()
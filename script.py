import pandas
import numpy
import matplotlib.pyplot as plt

def pol_regression(features_train, y_train, degree):
    X = getPolynomialDataMatrix(features_train, degree)
    XX = X.transpose().dot(X)
    w = numpy.linalg.solve(XX, X.transpose().dot(y_train))

    return w


def getPolynomialDataMatrix(x, degree):
    X = numpy.ones(x.shape)
    for i in range(1, degree + 1):
        X = numpy.column_stack((X, x ** i ))
    return X

data = pandas.read_csv("pol_regression.csv")

x_train = data["x"]
y_train = data["y"]

X = numpy.column_stack((numpy.ones(x_train.shape), x_train))
Y = numpy.column_stack((numpy.ones(y_train.shape), y_train))

w = pol_regression(X, Y, 1)
print(w)

# plt.clf()
# plt.plot(x_train, y_train, 'bo')
# plt.legend(('training points'))
# plt.show()

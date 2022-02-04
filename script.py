import pandas
import numpy
import matplotlib.pyplot as plt

def pol_regression(features_train, y_train, degree):
    X = getPolynomialDataMatrix(features_train, degree)
    XX = X.transpose().dot(X)
    w = numpy.linalg.inv(XX).dot(X.transpose().dot(y_train))

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

x_space = numpy.linspace(-5, 5, 100)

#w0 = pol_regression(x_train, y_train, 0)
#x_new0 = getPolynomialDataMatrix(x_space, 0)
#y_predicted0 = x_new0.dot(w0)

w1 = pol_regression(x_train, y_train, 1)
x_new1 = getPolynomialDataMatrix(x_space, 1)
y_predicted1 = x_new1.dot(w1)

w2 = pol_regression(x_train, y_train, 2)
x_new2 = getPolynomialDataMatrix(x_space, 2)
y_predicted2 = x_new2.dot(w2)

w3 = pol_regression(x_train, y_train, 3)
x_new3 = getPolynomialDataMatrix(x_space, 3)
y_predicted3 = x_new3.dot(w3)

w6 = pol_regression(x_train, y_train, 6)
x_new6 = getPolynomialDataMatrix(x_space, 6)
y_predicted6 = x_new6.dot(w6)

w10 = pol_regression(x_train, y_train, 10)
x_new10 = getPolynomialDataMatrix(x_space, 10)
y_predicted10 = x_new10.dot(w10)


# error = y_train - X.dot(w)
# sse = error.dot(error)
# print(sse)

plt.clf()
plt.plot(x_train, y_train, 'go')
#plt.plot(x_space, y_predicted0, 'b')
plt.plot(x_space, y_predicted1, 'y')
plt.plot(x_space, y_predicted2, 'r')
plt.plot(x_space, y_predicted3, 'g')
plt.plot(x_space, y_predicted6, 'b')
plt.plot(x_space, y_predicted10, 'm')
plt.legend(('training points'))
plt.show()

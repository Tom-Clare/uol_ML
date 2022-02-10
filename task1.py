import pandas
import numpy
import matplotlib.pyplot as plt

def pol_regression(features_train, y_train, degree):
    X = getPolynomialDataMatrix(features_train, degree) # get prepared array

    # perform least squares algorithm to calculate weights
    XX = X.transpose().dot(X)
    w = numpy.linalg.inv(XX).dot(X.transpose().dot(y_train))

    return w

def getPolynomialDataMatrix(x, degree):
    # prepare data
    X = numpy.ones(x.shape) # create array with same size as x
    for i in range(1, degree + 1): # for every degree, starting with 1
        # add new dimension to array, containing given data raised to the degree
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

plt.clf()
plt.plot(x_train, y_train, 'go')
#plt.plot(x_space, y_predicted0, 'b')
plt.plot(x_space, y_predicted1, 'y')
plt.plot(x_space, y_predicted2, 'r')
plt.plot(x_space, y_predicted3, 'g')
plt.plot(x_space, y_predicted6, 'b')
plt.plot(x_space, y_predicted10, 'm')
plt.legend(['training points', "1 degrees", "2 degrees", "3 degrees", "6 degrees", "10 degrees"])
plt.show()



count = len(x_train)
train_split = int(count * 0.7)
x_train2 = x_train[:train_split]
y_train2 = y_train[:train_split]
x_test2 = x_train[train_split:]
y_test2 = y_train[train_split:]

#w0 = pol_regression(x_train, y_train, 0)
#x_new0 = getPolynomialDataMatrix(x_space, 0)
#y_predicted0 = x_new0.dot(w0)

w1 = pol_regression(x_train2, y_train2, 1)
x_new1 = getPolynomialDataMatrix(x_space, 1)
y_predicted1 = x_new1.dot(w1)

w2 = pol_regression(x_train2, y_train2, 2)
x_new2 = getPolynomialDataMatrix(x_space, 2)
y_predicted2 = x_new2.dot(w2)

w3 = pol_regression(x_train2, y_train2, 3)
x_new3 = getPolynomialDataMatrix(x_space, 3)
y_predicted3 = x_new3.dot(w3)

w6 = pol_regression(x_train2, y_train2, 6)
x_new6 = getPolynomialDataMatrix(x_space, 6)
y_predicted6 = x_new6.dot(w6)

w10 = pol_regression(x_train2, y_train2, 10)
x_new10 = getPolynomialDataMatrix(x_space, 10)
y_predicted10 = x_new10.dot(w10)

MSSEtrain = []
MSSEtest = []

def eval_pol_regression(parameters, x, y, degree):
    
    prepared_x = getPolynomialDataMatrix(x, degree)
    msse = numpy.mean(((prepared_x).dot(parameters) - y) ** 2)

    return msse

MSSEtrain.append(eval_pol_regression(w1, x_train2, y_train2, 1))
MSSEtest.append(eval_pol_regression(w1, x_test2, y_test2, 1))

MSSEtrain.append(eval_pol_regression(w2, x_train2, y_train2, 2))
MSSEtest.append(eval_pol_regression(w2, x_test2, y_test2, 2))

MSSEtrain.append(eval_pol_regression(w3, x_train2, y_train2, 3))
MSSEtest.append(eval_pol_regression(w3, x_test2, y_test2, 3))

MSSEtrain.append(eval_pol_regression(w6, x_train2, y_train2, 6))
MSSEtest.append(eval_pol_regression(w6, x_test2, y_test2, 6))

MSSEtrain.append(eval_pol_regression(w10, x_train2, y_train2, 10))
MSSEtest.append(eval_pol_regression(w10, x_test2, y_test2, 10))

print(MSSEtrain)
print(MSSEtest)

plt.figure()
plt.semilogy([1,2,3,6,10], MSSEtrain)
plt.semilogy([1,2,3,6,10], MSSEtest)
plt.legend(('MSSE on training set', 'MSSE on test set'))
plt.show()

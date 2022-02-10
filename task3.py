import pandas
import matplotlib.pyplot as plt
import numpy

dtypes = {"Image number": int, "Bifurcation number": int, "Artery (1)/ Vein (2)": int, "Alpha": float, "Beta": float, "Lambda": float, "Lambda1": float, "Lambda2": float, "Participant Condition": str}
data = pandas.read_csv("Task3 - dataset - HIV RVG.csv", dtype=dtypes) # read data with custom dtypes
# create column shorthand column names for easier handling
data.columns = ["image", "bifurcation", "arteryvein", "alpha", "beta", "lambda", "lambda1", "lambda2", "status"]

data_summary = data.agg(["mean", "std", "min", "max"])

alpha_boxplot = data.boxplot(column="alpha", by="status")
beta_boxplot = data.boxplot(column="beta", by="status")
#plt.show()

def clean_normalise(data, column):
    std = 3 # remove row not within x standard deviations
    # calculate z-score, relative to population mean, and remove all rows that fall outside threshold
    data = data[((data[column] - data[column].mean()) / data[column].std()).abs() < std]

    # use the maximum value to normalise the data[column] data to values between 0 and 1
    data[column] = (data[column] - data[column].mean()) / data[column].std() 
    return data

data = clean_normalise(data, "alpha")
data = clean_normalise(data, "beta")
data = clean_normalise(data, "lambda")
data = clean_normalise(data, "lambda1")
data = clean_normalise(data, "lambda2")

alpha_boxplot = data.boxplot(column="alpha", by="status")
beta_boxplot = data.boxplot(column="beta", by="status")
plt.show()

data_summary = data.agg(["mean", "std", "min", "max"])

## Section 2

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data.loc[(data["status"] == "Control"), "status"] = 0
data.loc[(data["status"] == "Patient"), "status"] = 1
data["status"] = data["status"].astype(int)

#create a dataframe with all rows for our input data that contain a NaN value
mask = numpy.all(numpy.isnan(data), axis=1)
# and remove all rows from our main dataset that exist in that NaN dataframe
data = data[~mask] # leaving only values that contain no NaN values

features = data.drop("status", 1) # remove label column
x_vector = features.values # prepare input vector
y_vector = data["status"].values # prepare output vector

def train_epochs(x, y, epochs):
    scores = []
    for iterations in epochs:
        # create ANN classifier
        clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(500,2), activation="logistic", max_iter=iterations)
        # split dataset into training and testing data, 9:1 ratio, with a replicable split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=30)
        clf.fit(x_train, y_train) # "train" or fit classifier into model
        scores.append(clf.score(x_test, y_test))
    return scores

epochs = [50, 100, 150, 200, 250, 300]
scores = train_epochs(x_vector, y_vector, epochs)

fig = plt.figure()
fig1 = fig.add_subplot()
fig1.set_xlabel("No. of Epochs")
fig1.set_ylabel("Score")
fig1.set_title("Mean accuracy of Epochs")
plt.plot(epochs, scores)
plt.show()
print(scores)
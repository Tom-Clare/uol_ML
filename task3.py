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
    data[column] = (data[column] - data[column].min(axis=0)) / (data[column].max(axis=0) - data[column].min(axis=0))

    return data

data = clean_normalise(data, "alpha")
data = clean_normalise(data, "beta")
data = clean_normalise(data, "lambda")
data = clean_normalise(data, "lambda1")
data = clean_normalise(data, "lambda2")

alpha_boxplot = data.boxplot(column="alpha", by="status")
beta_boxplot = data.boxplot(column="beta", by="status")
#plt.show()

data_summary = data.agg(["mean", "std", "min", "max"])

## Section 2

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# prepare the labels into numerical form for the classifiers
data.loc[(data["status"] == "Control"), "status"] = 0
data.loc[(data["status"] == "Patient"), "status"] = 1
data["status"] = data["status"].astype(int)

#create a dataframe with all rows for our input data that contain a NaN value
mask = numpy.all(numpy.isnan(data), axis=1)
# and remove all rows from our main dataset that exist in that NaN dataframe
data_nonnan = data[~mask] # leaving only values that contain no NaN values

features = data_nonnan.drop(["image", "bifurcation", "arteryvein", "status"], 1) # remove label column

x_vector = features.values # prepare input vector
y_vector = data_nonnan["status"].values # prepare output vector

def mlpc_train_epochs(x_train, x_test, y_train, y_test, epochs):
    scores = []
    for iterations in epochs:
        # create ANN classifier
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,2), activation="logistic", max_iter=iterations)
        clf.fit(x_train, y_train) # "train" or fit classifier into model
        scores.append(clf.score(x_test, y_test)) # save score
    return scores

def forest_train(x_train, x_test, y_train, y_test, min_samples):
    scores = []
    for min_sample in min_samples:
        # create ANN classifier
        clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=min_sample)
        clf.fit(x_train, y_train) # "train" or fit classifier into model
        scores.append(clf.score(x_test, y_test)) # save score
    return scores

# split dataset into training and testing data, 9:1 ratio, with a replicable split
x_train, x_test, y_train, y_test = train_test_split(x_vector, y_vector, test_size=0.1, random_state=30)
epochs = [5, 10, 15, 20, 25, 50, 100] # define epoch counts
mlpc_scores = mlpc_train_epochs(x_train, x_test, y_train, y_test, epochs) # train

min_samples = [5, 10]
forest_scores = forest_train(x_train, x_test, y_train, y_test, min_samples)

fig = plt.figure()
fig1, fig2 = fig.subplots(2)
fig1.set_xlabel("No. of Epochs")
fig1.set_ylabel("Score")
fig1.set_title("Mean accuracy of Epochs")
fig1.plot(epochs, mlpc_scores)

fig2.set_xlabel("No. of samples")
fig2.set_ylabel("Score")
fig2.set_title("Mean accuracy of min leaf samples")
fig2.plot(min_samples, forest_scores)
plt.show()

## Section 3

# Create KFold object with 10 splits/iterations
k = 10
kf = KFold(n_splits=k)

# create score arrays
mlpc_50_accuracy = []
mlpc_500_accuracy = []
mlpc_1000_accuracy = []
forest_50_accuracy = []
forest_500_accuracy = []
forest_1000_accuracy = []

for train_i, test_i in kf.split(data):
    # split data according to this split's indexes
    x_train = x_vector[train_i]
    x_test = x_vector[test_i]
    y_train = y_vector[train_i]
    y_test = y_vector[test_i]

    
    with numpy.printoptions(threshold=numpy.inf):
        print(y_test)

    # create neural networks
    mlpc_50 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,2), activation="logistic", max_iter=100)
    mlpc_500 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,2), activation="logistic", max_iter=100)
    mlpc_1000 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(1000,2), activation="logistic", max_iter=100)

    # create random forests
    forest_50 = RandomForestClassifier(n_estimators=50, min_samples_leaf=10)
    forest_500 = RandomForestClassifier(n_estimators=500, min_samples_leaf=10)
    forest_1000 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=10)

    # fit (train) classifiers with this split's training data
    mlpc_50.fit(x_train, y_train)
    mlpc_500.fit(x_train, y_train)
    mlpc_1000.fit(x_train, y_train)
    forest_50.fit(x_train, y_train)
    forest_500.fit(x_train, y_train)
    forest_1000.fit(x_train, y_train)

    # test classifiers with this split's testing data
    mlpc_50_pred = mlpc_50.predict(x_test)
    mlpc_500_pred = mlpc_500.predict(x_test)
    mlpc_1000_pred = mlpc_1000.predict(x_test)
    forest_50_pred = forest_50.predict(x_test)
    forest_500_pred = forest_500.predict(x_test)
    forest_1000_pred = forest_1000.predict(x_test)

    # append accuracy score classifier-specific arrays
    mlpc_50_accuracy.append(accuracy_score(mlpc_50_pred, y_test))
    mlpc_500_accuracy.append(accuracy_score(mlpc_500_pred, y_test))
    mlpc_1000_accuracy.append(accuracy_score(mlpc_1000_pred, y_test))
    forest_50_accuracy.append(accuracy_score(forest_50_pred, y_test))
    forest_500_accuracy.append(accuracy_score(forest_500_pred, y_test))
    forest_1000_accuracy.append(accuracy_score(forest_1000_pred, y_test))

# mlpc_50_accuracy = [0.16560509554140126, 0.09872611464968153, 0.35987261146496813, 0.07006369426751592, 0.38338658146964855, 0.28434504792332266, 0.4185303514376997, 0.5814696485623003, 0.38338658146964855, 0.12460063897763578]
# mlpc_500_accuracy = [0.16560509554140126, 0.09872611464968153, 0.35987261146496813, 0.07006369426751592, 0.38338658146964855, 0.28434504792332266, 0.4185303514376997, 0.5814696485623003, 0.38338658146964855, 0.12460063897763578]
# mlpc_1000_accuracy = [0.16560509554140126, 0.09872611464968153, 0.35987261146496813, 0.07006369426751592, 0.38338658146964855, 0.28434504792332266, 0.4185303514376997, 0.5814696485623003, 0.38338658146964855, 0.12460063897763578]
# forest_50_accuracy = [0.4554140127388535, 0.4745222929936306, 0.410828025477707, 0.4299363057324841, 0.48881789137380194, 0.4057507987220447, 0.5207667731629393, 0.44089456869009586, 0.48562300319488816, 0.3514376996805112]
# forest_500_accuracy = [0.4968152866242038, 0.4745222929936306, 0.42356687898089174, 0.3885350318471338, 0.5175718849840255, 0.4057507987220447, 0.5335463258785943, 0.46006389776357826, 0.46006389776357826, 0.34824281150159747]
# forest_1000_accuracy = [0.47770700636942676, 0.4713375796178344, 0.4140127388535032, 0.4012738853503185, 0.4984025559105431, 0.38977635782747605, 0.5271565495207667, 0.4440894568690096, 0.48242811501597443, 0.34185303514376997]

average_scores = []
average_scores.append(sum(mlpc_50_accuracy) / len(mlpc_50_accuracy))
average_scores.append(sum(mlpc_500_accuracy) / len(mlpc_500_accuracy))
average_scores.append(sum(mlpc_1000_accuracy) / len(mlpc_1000_accuracy))
average_scores.append(sum(forest_50_accuracy) / len(forest_50_accuracy))
average_scores.append(sum(forest_500_accuracy) / len(forest_500_accuracy))
average_scores.append(sum(forest_1000_accuracy) / len(forest_1000_accuracy))

plt.bar(["MLPC 50" , "MLPC 500" , "MLPC 1000" , "Forest 50" , "Forest 500" , "Forest 1000" ], average_scores)
plt.xlabel("Algorithm")
plt.ylabel("Mean Accuracy Score")
plt.title("Accuracy of Hyper-parameter adjustment")
plt.show()

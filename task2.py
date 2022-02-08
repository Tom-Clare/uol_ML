from random import random
import numpy
import pandas
import math
import random
import matplotlib.pyplot as plt

column_names = ["height", "tail", "leg", "nose"]
data = pandas.read_csv("Task2 - dataset - dog_breeds.csv", names=column_names, header=0)
print(data)

def compute_euclidean_distance(vec_1, vec_2):
    
    # calculate euclidean distance
    xsumsqaured = (vec_2[0] - vec_1[0]) ** 2
    ysumsquared = (vec_2[1] - vec_1[1]) ** 2
    distance = math.sqrt(xsumsqaured + ysumsquared)

    return distance

def compute_euclidean_distance_4d(vec_1, vec_2):
    
    # calculate euclidean distance
    xsumsqaured = (vec_1[0] - vec_2[0]) ** 2
    ysumsquared = (vec_1[1] - vec_2[1]) ** 2
    zsumsquared = (vec_1[2] - vec_2[2]) ** 2
    wsumsquared = (vec_1[3] - vec_2[3]) ** 2
    distance = math.sqrt(xsumsqaured + ysumsquared + zsumsquared + wsumsquared)

    return distance

def initialise_centroids(dataset, k):
    centroids = []
    # randomly initialise centroids
    for i in range(0, k):
        max_height = max(dataset["height"])
        min_height = min(dataset["height"])
        height = random.uniform(min_height, max_height)

        max_tail = max(dataset["tail"])
        min_tail = min(dataset["tail"])
        tail = random.uniform(min_tail, max_tail)

        max_leg = max(dataset["leg"])
        min_leg = min(dataset["leg"])
        leg = random.uniform(min_leg, max_leg)
        
        max_nose = max(dataset["nose"])
        min_nose = min(dataset["nose"])
        nose = random.uniform(min_nose, max_nose)
        
        centroids.append([height, tail, leg, nose])

    return centroids

def kmeans(dataset, k):

    changed = True
    old_k = []

    while(changed):

        ####### Assign each data point to a cluster
        results = [] # This will hold cluster assignments
        results.clear()
        # For every record in dataset
        for index, row in dataset.iterrows():
            distances = [] # create intermediary place to store distances
            for cluster in k:
                # get distance for each cluster
                distances.append(compute_euclidean_distance_4d(row, cluster))

            # get cluster id of smallest in array
            closest_cluster_index = numpy.argmin(distances)
            # save cluster id
            results.append(closest_cluster_index)
            distances.clear()

        print(results)

        if k == old_k:
            changed = False
            print("not changed")
        else:
            changed = True
            print("changed")
            old_k.clear()
            ####### Calculate mean of each cluster of datapoints and return as new centroid
            cluster_values = []
            for cluster_index in range(0, len(k)):
                # iterate through results
                height = []
                tail = []
                leg = []
                nose = []
                for result_index in range(0, len(results)):
                    # if result belongs to this cluster
                    if results[result_index] == cluster_index:
                        # use id of this result to get result from dataset
                        row = dataset.iloc[result_index]
                        # collect data to aggregate later
                        height.append(row["height"])
                        tail.append(row["tail"])
                        leg.append(row["leg"])
                        nose.append(row["nose"])
                # divide vars to find new averages
                if len(height) != 0:
                    height_avg = numpy.mean(height)
                    tail_avg = numpy.mean(tail)
                    leg_avg = numpy.mean(leg)
                    nose_avg = numpy.mean(nose)
                    # put these new centroid values into new k values
                    k[cluster_index] = [height_avg, tail_avg, leg_avg, nose_avg]

                height.clear()
                tail.clear()
                leg.clear()
                nose.clear()

        old_k = k

    return k, results
    

k = initialise_centroids(data[1:], 3)
j, results = kmeans(data[1:], k)


colours = ["g", "r", "b"]
for index in range(0, len(results)):
    this_colour = colours[results[index]]
    row = data.iloc[index]
    plt.scatter(row["height"], row["tail"], c=this_colour)

for cluster in j:
    plt.scatter(cluster[0], cluster[1], c="m")

plt.show()


for index in range(0, len(results)):
    this_colour = colours[results[index]]
    row = data.iloc[index]
    plt.scatter(row["height"], row["leg"], c=this_colour)

for cluster in j:
    plt.scatter(cluster[0], cluster[2], c="m")

plt.show()

from random import random
import numpy
import pandas
import math
import random

data = pandas.read_csv("Task2 - dataset - dog_breeds.csv", sep=r'\s*,\s*', header=0)
print(data)

def compute_euclidean_distance(vec_1, vec_2):
    
    # calculate euclidean distance
    xsumsqaured = (vec_2[0] - vec_1[0]) ** 2
    ysumsquared = (vec_2[1] - vec_1[1]) ** 2
    distance = math.sqrt(xsumsqaured + ysumsquared)

    return distance

def compute_euclidean_distance_4d(vec_1, vec_2):
    
    # calculate euclidean distance
    xsumsqaured = (vec_2[0] - vec_1[0]) ** 2
    ysumsquared = (vec_2[1] - vec_1[1]) ** 2
    zsumsquared = (vec_2[2] - vec_1[2]) ** 2
    wsumsquared = (vec_2[3] - vec_1[3]) ** 2
    distance = math.sqrt(xsumsqaured + ysumsquared + zsumsquared + wsumsquared)

    return distance

def initialise_centroids(dataset, k):
    centroids = []
    # randomly initialise centroids
    for i in range(0, k):
        max_height = max(dataset["height"])
        min_height = min(dataset["height"])
        height = random.uniform(min_height, max_height)

        max_tail = max(dataset["tail length"])
        min_tail = min(dataset["tail length"])
        tail = random.uniform(min_tail, max_tail)

        max_leg = max(dataset["leg length"])
        min_leg = min(dataset["leg length"])
        leg = random.uniform(min_leg, max_leg)
        
        max_nose = max(dataset["nose circumference"])
        min_nose = min(dataset["nose circumference"])
        nose = random.uniform(min_nose, max_nose)
        
        centroids.append([height, tail, leg, nose])

    return centroids

def kmeans(dataset, k):

    results = []

    # cluster data into k groups
    for index, row in dataset.iterrows():
        distances = [] # create intermediary place to store distances
        for cluster in k:
            # get distance for each cluster
            distances.append(compute_euclidean_distance_4d(row, cluster))

        # get cluster id of smallest in array
        closest_cluster_index = numpy.argmin(distances)
        # save cluster id
        results.append(closest_cluster_index)
    
    return results
    

k = initialise_centroids(data[1:], 3)
print(kmeans(data[1:], k))
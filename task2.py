from ctypes import sizeof
from random import random
import numpy
import pandas
import math
import random
import matplotlib.pyplot as plt

def compute_euclidean_distance(vec_1, vec_2):

    distance = float('inf')
    if (len(vec_1) == len(vec_2)):
        sum = 0
        for feature_index in range(0, len(vec_1) - 1):
            squared = (vec_1[feature_index] - vec_2[feature_index]) ** 2
            sum = sum + squared
        
        distance = math.sqrt(squared)

    return distance

def initialise_centroids(dataset, k):
    centroids = []
    # randomly initialise centroids using range mechanism
    for i in range(0, k):
        height = random.uniform(min(dataset["height"]), max(dataset["height"]))
        tail = random.uniform(min(dataset["tail"]), max(dataset["tail"]))
        leg = random.uniform(min(dataset["leg"]), max(dataset["leg"]))
        nose = random.uniform(min(dataset["nose"]), max(dataset["nose"]))
        
        centroids.append([height, tail, leg, nose])

    return centroids

def kmeans(dataset, k):

    dataset = dataset.values

    changed = True  # tracks stability
    old_k = []

    cluster_assignment_history = []
    k_history = [k]

    while(changed): # Assign each data point to a cluster and calculate new centroid position
        cluster_assignment = [] # This will hold cluster assignments for each record
        # For every record in dataset
        for index in range(len(dataset)):
            distances = [] # create intermediary place to store distances
            for centroid in k:
                # calculate distance from each centroid
                distances.append(compute_euclidean_distance(dataset[index], centroid))

            # get id of shortest distance in array (which is equal to the respective centroid id)
            closest_centroid_index = numpy.argmin(distances)
            cluster_assignment.append(closest_centroid_index) # save centroid id 

        # re-center centroids
        numpy_results = numpy.array(cluster_assignment)
        # Use cluster assignment to group values and produce averages. Assign averages to centroids
        k = pandas.DataFrame(data.values).groupby(numpy_results).mean().values
        k_history.append(k)

        if numpy.array_equiv(k, old_k): # If stability reached
            changed = False  # stop looping next loop

        cluster_assignment_history.append(cluster_assignment) # save cluster info for eval
        old_k = k # save current centroid values to see if they change

    # Evaluate K-means
    eval_kmeans(dataset, cluster_assignment_history, k_history)

    return k, cluster_assignment

def eval_kmeans(dataset, cluster_assignment_history, k_history):
    # Calculate the sum of squared distances from each data point to centroid
    iteration_values = []
    error_sum_values = []
    for iteration in range(len(cluster_assignment_history)):
        error_sum = 0
        for index in range(len(cluster_assignment_history[iteration])):
            centroid_index = cluster_assignment_history[iteration][index]
            row = dataset[index]
            error = compute_euclidean_distance(row, k_history[iteration][centroid_index])
            error_sum = error_sum + error
        iteration_values.append(iteration + 1)
        error_sum_values.append(error_sum)

    # Plot data
    k = len(k_history[0])
    fig = plt.figure()
    fig3 = fig.add_subplot()
    fig3.set_xlabel("Iteration no.")
    fig3.set_ylabel("Sum of errors")
    fig3.set_title(f"Error intensity per iteration (K={k})")
    plt.plot(iteration_values, error_sum_values)

def show_plots(data, cluster_assignment, k):

    k_num = len(k)
    fig = plt.figure()
    fig1, fig2 = fig.subplots(2)

    ## Plot height vs tail length
    colours = ["c", "m", "y"]
    for index in range(0, len(cluster_assignment)):
        this_colour = colours[cluster_assignment[index]]
        row = data.iloc[index]
        fig1.scatter(row["height"], row["tail"], c=this_colour)

    for cluster in k:
        fig1.scatter(cluster[0], cluster[1], c="k", marker="X")

    fig1.set_xlabel("Height")
    fig1.set_ylabel("Tail Length")
    fig1.set_title(f"Height vs Tail Length in Dogs (K={k_num})")

    ## Plot Height vs leg length
    for index in range(0, len(cluster_assignment)):
        this_colour = colours[cluster_assignment[index]]
        row = data.iloc[index]
        fig2.scatter(row["height"], row["leg"], c=this_colour)

    for cluster in k:
        fig2.scatter(cluster[0], cluster[2], c="k", marker="X")

    fig2.set_xlabel("Height")
    fig2.set_ylabel("Leg Length")
    fig2.set_title(f"Height vs Leg Length in Dogs (K={k_num})")

    ## Show results
    plt.show()

column_names = ["height", "tail", "leg", "nose"]
data = pandas.read_csv("Task2 - dataset - dog_breeds.csv", names=column_names, header=0)

k_num = 2
k = initialise_centroids(data, k_num)
k, cluster_assignment = kmeans(data, k)
show_plots(data, cluster_assignment, k)

k_num = 3
k = initialise_centroids(data, k_num)
k, cluster_assignment = kmeans(data, k)
show_plots(data, cluster_assignment, k)

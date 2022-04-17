import numpy as np
import sys
from math import*
from numpy import random

#Euclidean distance
def euclidean_distance(x,y):
    result = 0
    #print(len(x), len(y))
    for i in range(0,len(x)):
        result += (x[i]-y[i])**2
    result = math.sqrt(result)
    return result

#Manhattan distance
def manhattan_disatance(x,y):
    result = 0
    for i in range(0,len(x)):
        result += abs(x[i]-y[i])
    return result

# Random centers
'''
* Generate k random initial centers
* Each dimension is a random number in range [low,high]
* Return the centers
'''

def random_centers(k,d,low=0, high=1):
    centers = np.empty((k,d))
    for i in range(0,k):
        for j in range(0,d):
            centers[i][j] = random.rand()*(high-low)+low        
    return centers 


# Assign centers
'''
* Given k centers, assign each data point to its nearest center
* Return the predicted labels

'''
def assign_centers(data, centers, dist_function):
    new_list = []
    for row in range(0,len(data)):
        total = sys.maxsize
        nearest = 0
        current = 0
        for c in range(0,len(centers)):
            current = dist_function(data[row],centers[c])
            if current < total:
                total = current
                nearest = c
            current = 0
        new_list.append(nearest)
    
    return new_list 

# Recalculate centers
'''
* Now each data point has its assignment to its nearest centers
* We need group the same-cluer data points to calculate a new center
* Return the new centers
'''
def recalculate_centers(data, labels, k):
    labes_list = []
    centers = []
    center_now = []
    for i in range(0,len(labels)):
        if labels[i] not in labes_list:
            labes_list.append(labels[i])
    for i in range(0,len(labes_list)):
        labes_list[i] = i
        
    total =0
    for a in range(0,len(labes_list)):
        for b in range(0,len(data)):
            if labels[b] == labes_list[a]:
                if len(center_now) == 0:
                    center_now = (data[b])
                    total =1
                else:
                    for c in range(0,len(data[b])):
                        center_now[c] = center_now[c] + data[b][c]
                        total +=1
        for x in range(0, len(center_now)):
            center_now[x] = center_now[x]/total
        centers.append(center_now)
        
        #print(centers)
        total = 0
        center_now = []
        
    return centers

# Cost function
'''
* Given the data points and their cluster assignments
* Calculate the total cost
'''
def compute_cost(data, labels, centers, dist_function):
    cost = 0
    for i in range(0,len(data)):
        tmp = dist_function(data[i], centers[0])
        #print("centers[0]: " + str(centers[0]))
        for j in range(0,len(centers)):
            tmp2 = dist_function(data[i],centers[j])
            if tmp2 < tmp:
                tmp = tmp2
        cost += tmp
    return cost


# Main function
def kmeans(data, dist_function, k, THRESHOLD=0.001, low=0, high=1):
    num_data, dimensions = data.shape
    centers = random_centers(k, dimensions, low, high)
    cost_difference = 10^6
    labels = np.zeros(num_data).astype(int)
    old_cost = compute_cost(data, labels, centers, dist_function)
    while cost_difference > THRESHOLD:
        labels = assign_centers(data, centers, dist_function)
        centers = recalculate_centers(data, labels, k)
        
        #print(centers)
        new_cost = compute_cost(data, labels, centers, dist_function)
        cost_difference = abs(new_cost - old_cost)
        old_cost = new_cost
    return centers, labels


# Helper function
import math
# Given predicted centers, predicted labels, true centers and true labels
# finds a mapping between predicted labels and actual labels
# and returns the number of true predictions and their percentage

def evaluation(data, pred_centers, pred_labels, true_labels, dist_function=euclidean_distance):
    # The predicted centers and actual centers may not match
    # The center we label as i can be equal to a different index j in the actual centers and labels
    # Therefore we need to do a mapping, so that we can calculate the accuracy.
    mapping = {}
    #k = pred_centers.shape[0]
    k = len(pred_centers)
    K = len(np.unique(true_labels))
    
    true_centers = recalculate_centers(data, true_labels, K)
    # To achieve a mapping, simply try to find which center actually belongs to which cluster
    # by mapping predicted centers to true centers, based on the distance.
    for c in range(k):
        # Distance off predicted center to true center
        min_dist = math.inf
        idx = c
        for tc in range(K) :
            dist = dist_function(pred_centers[c], true_centers[tc])
            if min_dist > dist:
                min_dist = dist
                idx = tc
        mapping[c] = idx
        
    accurate_points = 0
    for i in range(len(pred_labels)):
        # Get the actual cluster label
        mapped_value = mapping[pred_labels[i]]
        if mapped_value == true_labels[i]:
            accurate_points += 1
            
    accuracy = accurate_points/len(pred_labels)
    print("Accuracy is " + str(accuracy*100) + "%")
    return accuracy
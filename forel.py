import numpy as np
import matplotlib.pyplot as plt
import random

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def mean(data):
    means = [0] * len(data[0])
    for i in range(len(data)):
        for j in range(len(data[i])):
            means[j] += data[i][j]

    for i in range(len(means)):
        means[i] /= len(data)

    return means  

def distance(num1, num2):
    return (num1 - num2) ** 2

def euclideandistance(objA, objB):
    sumOfDistance = 0
    for a, b in zip(objA, objB):
        sumOfDistance += distance(a, b)

    return sumOfDistance ** 0.5

def forel(data, r):
    labels = [-1] * len(data)
    centroids = []
    cluster_num = 1
    remaining_indices = list(range(len(data)))

    while (len(remaining_indices)!=0):

        # assign a random centroid from the points
        random_index = random.choice(remaining_indices)
        centroid = data[random_index]
        arr_selected_els = []
        indexes_to_delete = []

        # find all points which have less or equal to r distance
        for h in range(100):
          centroidCopy = centroid.copy()
          arr_selected_els = []
          indexes_to_delete = []
          
          for i in range(len(remaining_indices)):
            dist = euclideandistance(data[remaining_indices[i]], centroid)
            
            if dist <= r:
              indexes_to_delete.append(remaining_indices[i])

          for i in range(len(indexes_to_delete)):
            #   label them
              labels[indexes_to_delete[i]] = cluster_num
            #   add them to sep array (for later calc of their mean)
              arr_selected_els.append(data.tolist()[indexes_to_delete[i]])
            #   save their indices (to delete them afterwards from the pool)

        # adjust the centroid of found cluster till stops changing
          centroid = mean(arr_selected_els)

          is_equal = True
          for i in range(len(centroid)):
            if centroid[i] != centroidCopy[i]:
              is_equal = False

          if is_equal:
            cluster_num+=1
            centroids.append(centroid)
            break

        print(indexes_to_delete)
        # delete already marked points from the pool
        for i in range(len(indexes_to_delete)):
            remaining_indices.remove(indexes_to_delete[i])

    print(labels, centroids)
    return labels, centroids

    
# Создаем искусственные данные для демонстрации
np.random.seed(0)
data = np.concatenate([np.random.randn(50, 2) + [2, 2],
                       np.random.randn(50, 2) + [-2, -2],
                       np.random.randn(50, 2) + [2, -2]])

labels, centroids = forel(data, 3)

# # Визуализация исходных данных
if True:
    plt.figure(facecolor='#11111B')
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#11111B')
    plt.scatter(data[:, 0], data[:, 1])
    plt.title('Original Data', color='#FFFFFF')
    plt.xlabel('X', color='#FFFFFF')
    plt.ylabel('Y', color='#FFFFFF')
    plt.tick_params(axis='x', colors='#FFFFFF')  # Change x-axis tick color to red
    plt.tick_params(axis='y', colors='#FFFFFF')  # Change y-axis tick color to blue


# # Визуализация результатов кластеризации
if True:
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#11111B')
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.legend()
    plt.title('Forel Clustering', color='#FFFFFF')
    plt.xlabel('X', color='#FFFFFF')
    plt.ylabel('Y', color='#FFFFFF')
    plt.tick_params(axis='x', colors='#FFFFFF')  # Change x-axis tick color to red
    plt.tick_params(axis='y', colors='#FFFFFF')  # Change y-axis tick color to blue

    plt.tight_layout()
    plt.show()
    plt.close()
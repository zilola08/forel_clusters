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

def forel(data, r):
    labels = [-1] * len(data)
    centroids = []
    cluster_num = 1
    remaining_indices = list(range(len(data)))
    # print("remaining_indices: ", remaining_indices)

    while (len(remaining_indices)!=0):
        random_index = random.choice(remaining_indices)
        centroid = data[random_index]
        centroidCopy = centroid.copy()
        arr_selected_els = []
        indexes_to_delete = []

        for i in range(len(remaining_indices)):
            dist = manhattan_distance(data[remaining_indices[i]], centroid)
            if dist <= r:
              labels[remaining_indices[i]] = cluster_num
              arr_selected_els.append(data.tolist()[i])
              indexes_to_delete.append(remaining_indices[i])
              
              for h in range(100):
                centroid = mean(arr_selected_els)
                print(centroid)

                is_equal = True
                for i in range(len(centroid)):
                  if centroid[i] != centroidCopy[i]:
                    is_equal = False

                if is_equal:
                  cluster_num+=1
                  centroids.append(centroid)
                  break

        for i in range(len(indexes_to_delete)):
            remaining_indices.remove(indexes_to_delete[i])
        # print("indexes_to_delete: ", indexes_to_delete)
        # print("remaining_indices: ", remaining_indices)

    print(labels, centroids)
    return labels, centroids

    
# Создаем искусственные данные для демонстрации
np.random.seed(0)
data = np.concatenate([np.random.randn(50, 2) + [2, 2],
                       np.random.randn(50, 2) + [-2, -2],
                       np.random.randn(50, 2) + [2, -2]])

labels, centroids = forel(data, 1)

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

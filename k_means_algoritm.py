import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# k-means
# Датасет - ирисы.

# 2. Написать самостоятельно алгоритм для оптимального количества кластеров (п.1) k-means без использования библиотек (но можно пользоваться библиотеками, не связанными с самим алгоритмом - отрисовки, подсчетов и т.д.). 
# Рисунки выводятся на каждый шаг – сдвиг центроидов, смена точек своих кластеров. Сколько шагов, столько рисунков (можно в виде gif). Точки из разных кластеров разными цветами.

dataset_iris = load_iris()
array_iris = dataset_iris.data

clusters_count = 3  # Количество кластеров
max_iter = 100
images = []
np.random.seed(0) #фиксирует случайные числа
centroids = array_iris[np.random.choice(range(len(array_iris)), clusters_count, replace=False)] #выбирает случайные точки из данных в качестве начальных центроид, они гарантированно не повторяются


def metrics(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) #расстояние между двумя точками x и y

#для каждой точки находим расстояние для всех центроидов
def appropriation_point_clusters(array_iris, centroids):
    clusters = []
    for point in array_iris:
        distances = [metrics(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # находим и добавляем индексы ближайшего (минимального) центроида
        clusters.append(cluster)
    return np.array(clusters) #массив, где каждой точке соответствует номер её кластера

#обновление координат центроидов
def update_centroids(array_iris, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = array_iris[clusters == i] #для каждого кластера i выбираются точки, принадлежащие этому кластеру
        centroid = np.mean(cluster_points, axis=0) #вычисляем среднее значение координат точек кластера (центр)
        centroids.append(centroid)
    return np.array(centroids) #массив новых центроид


for iteration in range(max_iter):
    clusters = appropriation_point_clusters(array_iris, centroids) #На каждой итерации точки распределяются по кластерам с текущими центроидами
    plt.figure(figsize=(6, 6))
    colors = ['orange', 'g', 'b']

    for i in range(clusters_count):
        cluster_points = array_iris[clusters == i]  # принадлежность точки кластеру
        plt.scatter(cluster_points[:, 0], cluster_points[:,1], c=colors[i], label=f'Кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Центроиды')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-means Clustering - Итерация {iteration + 1}')
    plt.legend()

    plt.savefig(f'iteration_{iteration}.png')
    plt.close()

    new_centroids = update_centroids(array_iris, clusters, clusters_count)

    if np.all(centroids == new_centroids):
        print("Отработал", iteration + 1) #если координаты не изменились, алгоритм завершает работу.
        break
    #обновляем центроиды для следующей итерации
    centroids = new_centroids

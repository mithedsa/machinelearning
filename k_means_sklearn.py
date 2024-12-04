from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib.pyplot as plt

# k-means
# Датасет - ирисы.
# 1. Найти оптимальное количество кластеров при помощи готовых библиотек (sklearn).

dataset_iris = datasets.load_iris()
array = dataset_iris.data #массив для обучения модели

# Массив для суммы квадратов расстояний точек до ближайшего центра кластера
sum_squared_distances = []

for clusters in range(1,11):
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(array) # Обучает модель на данных
    sum_squared_distances.append(kmeans.inertia_) #инерция-метрика(сумма квадратов расстояний от точек до центроидов кластеров)


# Построение графика "методом локтя"
plt.plot(range(1, 11), sum_squared_distances, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний')
plt.title('Метод локтя')
plt.savefig('elbow_method.png')  # Сохранение графика
# plt.show()

# Рассчитаем относительные изменения суммы квадратов расстояний
differences = [sum_squared_distances[i] - sum_squared_distances[i + 1] for i in range(len(sum_squared_distances) - 1)]
relative_changes = [differences[i] / sum_squared_distances[i] for i in range(len(differences))]

# Найдем индекс максимального изменения
optimal_clusters = relative_changes.index(max(relative_changes[1:])) + 2  # +2, т.к. кластеры начинаются с 1 и пропускаем 1 кластер

# Вывод оптимального количества кластеров
print(f"Оптимальное количество кластеров: {optimal_clusters}")

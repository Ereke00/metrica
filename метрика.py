#Шекербек Ержан АЖ-35
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Представим себе набор данных о животныхх, где каждый элемент - это [высота, длина хвоста, класс]
# Класс 0 - собаки, Класс 1 - кошки 
data = np.array([
    [30, 10, 0],
    [20, 5, 1],
    [25, 8, 0],
    [15, 3, 1],
    [35, 12, 0]
])

X = data[:, :-1]
y = data[:, -1]

k = 3  
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X, y)

new_animal_1 = np.array([[22, 7]])
prediction_1 = knn.predict(new_animal_1)
print("Предсказанный класс для нового видца 1:", prediction_1)  

new_animal_2 = np.array([[18, 4]])
prediction_2 = knn.predict(new_animal_2)
print("Предсказанный класс для нового видца 2:", prediction_2)  

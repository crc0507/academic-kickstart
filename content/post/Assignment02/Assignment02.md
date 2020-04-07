---
date: 2017-12-01
title: Assignment02
---

### This blog illustrates KNN from scratch with different hyperparameters.
name = "download path"
url = "files/Chang_02.ipynb"


```python
from random import seed
from csv import reader
from math import sqrt
import random
from collections import Counter
from numpy import *
import numpy as np
from prettytable import PrettyTable
```

# 1. Load a CSV file and divide the data to dev_data and test_data


```python
def load_file(filename):
    alldata = list()
    with open(filename, 'r') as file:
        r = reader(file)
        for row in r:
            if not row:
                continue
            alldata.append(row)
    return alldata

filename = '/Users/jizhimeicrc/Desktop/iris.data.csv'
dataset = load_file(filename)

def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_to_int(dataset, column):
    classes = [row[column] for row in dataset]
    types = set(classes)
    diction = dict()
    for i, value in enumerate(types):
        diction[value] = i
    for row in dataset:
        row[column] = diction[row[column]]
    return diction

for i in range(len(dataset[0])-1):
    str_to_float(dataset, i)
# convert class column to integers
str_to_int(dataset, len(dataset[0])-1)
length = int(len(dataset))
# divide the dataset randomly
a = list(range(length))
random.seed(50)
random.shuffle(a)
dev_data = list()
test_data = list()
for index in a[:int(length*0.7)]:
    dev_data.append(dataset[index])
for index in a[int(length*0.7):]:
    test_data.append(dataset[index])
```

# 2. Implement the Euclidean distance


```python
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


# Locate the most similar neighbors
def get_euclidean_neighbors(dev, single_row, num_neighbors):
    distances = list()
    for dev_row in dev:
        dist = euclidean_distance(single_row, dev_row)
        distances.append((dev_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
```

# 3. Implement the Normalized Euclidean distance


```python
def normalize_euclidean(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        average = (row1[i] - row2[i])/2
        deno = sqrt((row1[i] - average) ** 2 + (row2[i] - average) ** 2)
        molecule = row1[i] - row2[i]
        distance += ((molecule / deno) ** 2)
    return sqrt(distance)


# Locate the most similar neighbors
def get_normalized_neighbors(dev, single_row, num_neighbors):
    distances = list()
    for dev_row in dev:
        dist = normalize_euclidean(single_row, dev_row)
        distances.append((dev_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
```

# 4. Implement the Cosine distance


```python
def cosine_distance(row1, row2):
    product = 0.0
    row1_dist = 0.0
    row2_dist = 0.0
    for i in range(len(row1)-1):
        product += row1[i] * row2[i]
    for i in range(len(row1)-1):
        row1_dist += row1[i] * row1[i]
    for i in range(len(row2)-1):
        row2_dist += row2[i] * row2[i]
    distance = product/(sqrt(row1_dist * row2_dist))
    return 1 - distance


def get_cosine_neighbors(dev, single_row, num_neighbors):
    distances = list()
    for dev_row in dev:
        dist = cosine_distance(single_row, dev_row)
        distances.append((dev_row, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
```

# 5. kNN Algorithm


```python
def knn(data, num_neighbors, algorithm):
    correct_predict = 0
    for element in data:
        newdata = data.copy()
        newdata.remove(element)
        neighbors = algorithm(newdata, element, num_neighbors)
        neighbors_labels = list()
        for rows in neighbors:
            neighbors_labels.append(rows[-1])
        counter = Counter(neighbors_labels)
        pred_label = counter.most_common(1)[0][0]
        if pred_label == element[-1]:
            correct_predict += 1

    return correct_predict
```

# 6. Calculate all the accuracy

## 6.1 euclidean distance


```python
accuracy11 = knn(dev_data, 1, get_euclidean_neighbors)/len(dev_data)
print(accuracy11)
accuracy12 = knn(dev_data, 3, get_euclidean_neighbors)/len(dev_data)
print(accuracy12)
accuracy13 = knn(dev_data, 5, get_euclidean_neighbors)/len(dev_data)
print(accuracy13)
accuracy14 = knn(dev_data, 7, get_euclidean_neighbors)/len(dev_data)
print(accuracy14)
```

    0.9619047619047619
    0.9523809523809523
    0.9714285714285714
    0.9809523809523809


## 6.2 cosine distance


```python
accuracy21 = knn(dev_data, 1, get_cosine_neighbors)/len(dev_data)
print(accuracy21)
accuracy22 = knn(dev_data, 3, get_cosine_neighbors)/len(dev_data)
print(accuracy22)
accuracy23 = knn(dev_data, 5, get_cosine_neighbors)/len(dev_data)
print(accuracy23)
accuracy24 = knn(dev_data, 7, get_cosine_neighbors)/len(dev_data)
print(accuracy24)
```

    0.9238095238095239
    0.9523809523809523
    0.9523809523809523
    0.9619047619047619


## 6.3 normalized euclidean distance


```python
accuracy31 = knn(dev_data, 1, get_normalized_neighbors)/len(dev_data)
print(accuracy31)
accuracy32 = knn(dev_data, 3, get_normalized_neighbors)/len(dev_data)
print(accuracy32)
accuracy33 = knn(dev_data, 5, get_normalized_neighbors)/len(dev_data)
print(accuracy33)
accuracy34 = knn(dev_data, 7, get_normalized_neighbors)/len(dev_data)
print(accuracy34)
```

    0.9714285714285714
    0.9714285714285714
    0.9714285714285714
    0.9714285714285714


## 6.4 accuracy result table


```python
x=PrettyTable()
x.add_column(" ", ["k=1", "k=3", "k=5", "k=7"])
list1 = [accuracy11, accuracy12, accuracy13, accuracy14]
list2 = [accuracy21, accuracy22, accuracy23, accuracy24]
list3 = [accuracy31, accuracy32, accuracy33, accuracy34]
x.add_column("euclidean distance", list1)
x.add_column("cosine distance", list2)
x.add_column("normalized euclidean distance", list3)
print(x)
```

    +-----+--------------------+--------------------+-------------------------------+
    |     | euclidean distance |  cosine distance   | normalized euclidean distance |
    +-----+--------------------+--------------------+-------------------------------+
    | k=1 | 0.9619047619047619 | 0.9238095238095239 |       0.9714285714285714      |
    | k=3 | 0.9523809523809523 | 0.9523809523809523 |       0.9714285714285714      |
    | k=5 | 0.9714285714285714 | 0.9523809523809523 |       0.9714285714285714      |
    | k=7 | 0.9809523809523809 | 0.9619047619047619 |       0.9714285714285714      |
    +-----+--------------------+--------------------+-------------------------------+


## 6.5 So the best accuracy is using euclidean distance with k = 7. Let's applying this model to the test data.


```python
accuracy_test = knn(test_data, 7, get_euclidean_neighbors)/len(test_data)
print(accuracy_test)
```

    0.9555555555555556



```python

```

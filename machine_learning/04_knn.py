import pandas as pd
import numpy as np

path1 = 'machine_learning/datasets/titanic.csv' #'/content/drive/MyDrive/CS167/datasets/titanic.csv'
titanic = pd.read_csv(path1) 
# print(titanic.head())

path2 = 'machine_learning/datasets/irisData.csv' #'/content/drive/MyDrive/CS167/datasets/iris.csv'
iris= pd.read_csv(path2)
# print(iris.head())

path3 = 'machine_learning/datasets/vehicles.csv' #'/content/drive/MyDrive/CS167/datasets/vehicles.csv'
vehicles= pd.read_csv(path3, low_memory=False)
# print(vehicles.head())

###k-Nearest-Neighbor Algorihm: Predict the most commonly appearing class among the k closest training examples.


#print(pd.get_dummies(titanic.embark_town))
subset = vehicles[['year', 'cylinders', 'displ', 'comb08']]
# print(subset.head())
# print(subset.comb08.sort_values().unique())


# 1. Calculate Distances

iris['distance_to_new'] = np.sqrt(
    (5.1 - iris['petal length'])**2 
    +(7.2 - iris['sepal length'])**2 
    +(1.5 - iris['petal width'])**2
    +(2.5 - iris['sepal width'])**2)

print(iris.head())

# 2. Sort the Data by the distance

k = 15
sorted_data = iris.sort_values(['distance_to_new'])
print(sorted_data.head()) #shortest distances first

# 3. Display the most common species among these 5
print(sorted_data.iloc[0:5]['species'].mode()[0])
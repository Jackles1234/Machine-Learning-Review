import pandas as pd
import numpy as np
path = 'machine_learning/datasets/penguins_size.csv' #'/content/drive/MyDrive/CS167/datasets/penguins_size.csv'
penguins = pd.read_csv(path)
#print(penguins.head())

path1 = 'machine_learning/datasets/irisData.csv' #'/content/drive/MyDrive/CS167/datasets/irisData.csv'
iris = pd.read_csv(path1)
#print(iris.head())

path2 = 'machine_learning/datasets/titanic.csv' #'/content/drive/MyDrive/CS167/datasets/titanic.csv'
titanic = pd.read_csv(path2)
#print(titanic.head())

#Normalization Motivation:
#In datasets that have numeric data, the columns that have the largest magnitude will have a greater 'say' in the decision of what to predict.

def knn(specimen, data, k):
    # write your code in here to make this function work
    # 1. calculate distances
    data_copy = data.copy() #good practice to make a copy of the data
    data_copy['distance_to_new'] = np.sqrt(
        (specimen['petal length'] - data_copy['petal length'])**2 
        +(specimen['sepal length'] - data_copy['sepal length'])**2 
        +(specimen['petal width'] - data_copy['petal width'])**2
        +(specimen['sepal width'] - data_copy['sepal width'])**2)

    # 2. sort
    sorted_data = data_copy.sort_values(['distance_to_new'])
    
    # 3. predict
    prediction = sorted_data.iloc[0:k]['species'].mode()[0]

    #return prediction
    return prediction

new_iris = {}
new_iris['petal length'] = 5.1
new_iris['sepal length'] = 7.2
new_iris['petal width'] = 1.5
new_iris['sepal width'] = 2.5

def z_score(columns, data):
    data_copy = data.copy()
    
    for col in columns:
        # Compute mean and standard deviation for the column
        col_mean = data_copy[col].mean()
        col_std = data_copy[col].std()
        
        # Replace the column with the z-score normalized values
        data_copy[col] = (data_copy[col] - col_mean) / col_std
    
    return data_copy

pred = knn(new_iris, iris, 15)
print(pred)
iris_norm = z_score(['sepal length', 'sepal width', 'petal width', 'petal length'], iris)
print(iris_norm.head())


#Weighted k-NNN Intuition:
#In weighted kNN, the nearest k points are given a weight, and the weights are grouped by the target variable. The class with the largest sum of weights will be the class that is predicted.

#1: Start by calculating the distance between the new example ('X'), and each of the other training examples:
#2: Then, __calculate the weight___ of each training example using the inverse distance squared.
#3: Find the k closest neighbors--let's assume k=3 for this example:


def weighted_knn(specimen, data, k):
    #calculate the distance
    data['distance_to_new'] = np.sqrt(
    (specimen['petal length'] - data['petal length'])**2 
    +(specimen['sepal length'] - data['sepal length'])**2 
    +(specimen['petal width'] - data['petal width'])**2
    +(specimen['sepal width'] - data['sepal width'])**2)

    
    # calculate the weights (remember, weights are 1/d^2)
    data['distance_to_new'] = np.sqrt(
        (specimen['petal length'] - data['petal length'])**2 +
        (specimen['sepal length'] - data['sepal length'])**2 +
        (specimen['petal width'] - data['petal width'])**2 +
        (specimen['sepal width'] - data['sepal width'])**2
    )
    
    # find the k closest neighbors
    data.sort_values(['distance_to_new'], inplace=True)
    neighbors = data.iloc[0:k]
    
    # use groupby to sum the weights of each species in the closest k
    weighted_sums = neighbors.groupby('species')['weight'].sum()
    # return the class that has the largest sum of weight.
    predicted_species = weighted_sums.idxmax()

    return predicted_species

new_iris = {}
new_iris['petal length'] = 5.1
new_iris['sepal length'] = 7.2
new_iris['petal width'] = 1.5
new_iris['sepal width'] = 2.5

weighted_knn(new_iris, iris, 15)


print("Not normalized:")
print('unweighted kNN, k=3:', knn(new_iris, iris, 3))
print('unweighted kNN, k=5:', knn(new_iris, iris, 5))
print('weighted kNN, k=3:', weighted_knn(new_iris, iris, 3))
print('weighted kNN, k=5:', weighted_knn(new_iris, iris, 5))

print("Normalized:")
print('unweighted kNN, k=3:', knn(norm_iris, iris_norm, 3))
print('unweighted kNN, k=5:', knn(norm_iris, iris_norm, 5))
print('weighted kNN, k=3:', weighted_knn(norm_iris, iris_norm, 3))
print('weighted kNN, k=5:', weighted_knn(norm_iris, iris_norm, 5))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from sklearn.metrics import accuracy_score


matplotlib.use('TkAgg')

def test_matplot():
    xvals = [1,2,3,4,5]
    series1 = [0.66,0.61,0.69,0.73,0.77]
    series2 = [0.8,0.83,0.77,0.81,0.79]
    series3 = [0.55,0.67,0.5,0.73,0.66]

    #add titles to axis and graph
    plt.suptitle('my rockin plot', fontsize=18)
    plt.xlabel('a very cool x axis')
    plt.ylabel('awesome y axis')

    #plot the data
    plt.plot(xvals, series1, 'ro--', label='1st series')
    plt.plot(xvals, series2, 'bs-.', label='2nd series')
    plt.plot(xvals, series3, 'g^-', label='3rd series')
    plt.axis([0,6,0,1]) #[x_min, x_max, y_min, y_max]
    plt.show()


def vehicles_test():
    data = pd.read_csv('machine_learning/datasets/vehicles.csv')
    pd.set_option('display.max_columns', 100)

    gas_vehicles = data[data['fuelType']=='Regular']

    # a silly function that returns the average MPG for the first k cars in the df
    def getAverageMPG(data, k):
        return data["comb08"].iloc[0:k].mean()

    number_of_points = 500

    #populate the series list
    series = []
    for i in range(1, number_of_points):
        val = getAverageMPG(gas_vehicles, i)
        series.append(val)

    #plot it!
    xvals = range(1, number_of_points)
    plt.suptitle('Average MPG', fontsize=18)
    plt.xlabel('cars used in average')
    plt.ylabel('average MPG')
    plt.plot(xvals, series, 'r,-', label='MPG')
    plt.legend(loc='lower right', shadow=True)
    plt.axis([1, number_of_points, 10,25])
    plt.show()

#test_matplot()
# vehicles_test()


# Evaluation of Machine Learning Algorithms:

# We want to know how good our model is at making predictions. How can we test it?

# Option 1: Deploy the model in a live setting and see how it does on new examples.
# Option 2: Run each of our training examples through the model and see how many it gets correct
# Option 3: Cross-Validation - set aside some of your training examples to be used for testing.


# Classification Metrics: Accuracy
# Accuracy: The fraction of test examples your model predicted correctly

# Classification Metrics: Confusion Matrix
# confusion matrix: A specific table layout that allows the visualiztion of the performance of an algorithm. Each row represents instances in an actual class while each column represents the instances in a predicted class.


# Classification v Regression:

# What's the difference?
# The output variable in regression is numerical (or continuous).
# The output variable in classification is categorical (or discrete).


#Regression Metrics: Mean Absolute Error (MAE): the average difference between the actual and predicted target values.
#Regression Metrics: Mean Squared Error (MSE): The average squared difference between the actual and predicted targets.

# MAE v MSE:


# Regression Metrics: R^2
# Things you should know:
# values fall between 0 and 1
# 1 means you perfectly fit the data
# 0 means you've done no better than average
# Negative numbers mean that the naive model that predicts the average is actually a better predictor--yours is really bad.
path = 'machine_learning/datasets/irisData.csv' #change this line to point to your data
data = pd.read_csv(path) 
#shuffle the data - "sampling" the full set in random order
shuffled_data = data.sample(frac=1, random_state=41)

#cross-validation
#use the first 20 rows in the shuffled set as testing data #train with the rest

test_data = shuffled_data.iloc[0:20]
train_data = shuffled_data.iloc[20:]

def classify_kNN(new_example,train_data,k):
    #getting a copy of the training set just so we don't
    #mess up the original
    train_data_copy = train_data.copy()
    train_data_copy['distance_to_new'] = numpy.sqrt(
        (new_example['petal length']-train_data_copy['petal length'])**2
        +(new_example['sepal length']-train_data_copy['sepal length'])**2
        +(new_example['petal width']-train_data_copy['petal width'])**2
        +(new_example['sepal width']-train_data_copy['sepal width'])**2)
    
    sorted_data = train_data_copy.sort_values(['distance_to_new'])
    #mode to get most common thing in the first k examples in the sorted dataframe
    #iloc to get the actual string, mode will return the string inside of a pandas Series
    prediction = sorted_data.iloc[0:k]['species'].mode().iloc[0] 
    return prediction

def accuracy(actual,predicted):
    #get the series comparing the two series
    compared = actual.equals(predicted)
    correct_predictions = compared[ compared == True ]
    num_correct = len(correct_predictions)
    frac_correct = num_correct/len(actual)
    return frac_correct


def classify_all_kNN(test_data,train_data,k):
    #apply the classify_kNN function to each item in the test data with the train
    #data and k passed as the other two arguments. The result will be a series of 
    #the individual results.
    
    results = []
    
    for i in range(len(test_data)):
        prediction = classify_kNN(test_data.iloc[i], train_data,k)
        results.append(prediction)
    
    return pd.Series(results)


predictions5NN = classify_all_kNN(test_data,train_data,11)


#this will print out our predictions so we can see:
print('ACTUAL\t\tPREDICTIONS')
for i in range(20):
    print(test_data['species'].iloc[i], "  ", predictions5NN.iloc[i] )

#acc = accuracy(test_data['species'],predictions5NN)

acc = accuracy_score(test_data['species'], predictions5NN)
print("accuracy:", acc)


path = 'machine_learning/datasets/irisData.csv'

#reload the data
data = pd.read_csv(path)

shuffled_data = data.sample(frac=1, random_state = 41)

test_data = shuffled_data.iloc[0:20]
train_data = shuffled_data.iloc[20:]


k_vals = [1,3,5,9,15,21,31,51,101,129]
kNN_accuracies = []

for k in k_vals:
    predictions = classify_all_kNN(test_data,train_data,k)
    current_accuracy = accuracy_score(test_data['species'],predictions)
    kNN_accuracies.append(current_accuracy)


plt.suptitle('Iris Data k-NN Experiment',fontsize=18)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_vals,kNN_accuracies,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,130,0,1])

plt.show()
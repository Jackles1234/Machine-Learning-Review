import pandas as pd
import numpy as np
path = 'machine_learning/datasets/irisData.csv' #'/content/drive/MyDrive/CS167/datasets/irisData.csv'
iris= pd.read_csv(path)

# ID3 Decision Tree Learning Algorithm.

# It goes like this:

# Main ID3 Loop:
#     Assign A to be the best decision feature for the next node.
#     Assign A as decision feature for the node
#     For each possible attribute of A, create a new descendant of node
#     Sort training examples to leaf nodes
#     If Training examples are perfectly classified, the STOP, Else, iterate over new leaf nodes.


# Entropy: measure of impurity/randomness
#     high entropy: more evenly split classes - highly unpredictable
#     low entropy: mostly one class - highly predictable

# Entropy Prior
# Prior Probability:aka the 'prior'
#     the split of the examples
#     if I have 9 positive examples and 5 negative examples, my prior is:

import math
# here's the syntax for a log(Base 2)
def entropy(percentage_list):
    #input: percentage_list consists of float values that sum to 1.0 
    #return: calculation of entropy for input percentages
    result = 0
    for percentage in percentage_list:
        if percentage != 0:
            result += -percentage*math.log2(percentage)
    return result

path = 'machine_learning/datasets/restaurant.csv'
restaurant = pd.read_csv(path)

print(restaurant.head())
#1: Start by calculating the entropy of the example before picking a feature:
restaurant[['pat', 'target']].sort_values(['pat','target'])
entropy_patrons_full = entropy([4/6,2/6]) # 4/6 was No; 2/6 was Yes
entropy_patrons_none = entropy([2/2,0/2])
entropy_patrons_some = entropy([0/4,4/4])
print(entropy_patrons_full, entropy_patrons_none, entropy_patrons_some)
#The expected entropy for a feature is defined as the weighted sum of the entropies multiplied by the fraction of samples that belong to each set:

expected_entropy_patrons = 6/12*entropy_patrons_full + 2/12*entropy_patrons_none + 4/12*entropy_patrons_some
print(expected_entropy_patrons)

#The information gain is difference between the entropy before the test and the expected entropy after the test.
#calculate information gain (prior entropy - expected entropy)
information_gain_patrons = 1.0 - expected_entropy_patrons
print(information_gain_patrons)

import pandas as pd
import numpy as np
path = 'machine_learning/datasets/titanic.csv' 
titanic = pd.read_csv(path)
titanic.head()

### Missing Data ###

# isna(), notna(), and any()

 # isna() will return a boolean series where it is True if the element is `NaN'.
 # notna() will return a bollean seires where it is True if the element is not NaN.
 # Use any() on the call to isna() we just did to let us know which columns have missing data:
 # value_counts() allows us to check to see how much data they are missing using 
print(titanic.loc[:4].isna())
print(titanic.isna().any())
print(titanic.deck.value_counts(dropna=False) )

#Option 1: Drop it using dropna()
print("before: ", titanic.shape)
clean = titanic.dropna()
print("after: " , clean.shape)
#add the parameter inplace=True to the function call, and it will drop the rows in the original dataset (be careful with this one)
##titanic.dropna(inplace=True)

#Option 2: Fill it using fillna()
print("before: ", titanic['age'].isna().any())
age_mean = titanic['age'].mean()
titanic['age'].fillna(age_mean, inplace=True)
print("after: ", titanic['age'].isna().any())
titanic.head(7)

#Option #3: Let it be
a = np.nan


### Normalization ###
    #Rescale attrbute values so they're about the same
    #Adjusting values measured on different scales to a common scale

#One simple method of normalizing data is to replace each value with a proportion relativeto the max value.
#Z-Score: Another Normalization Method

#Normalization Code:
#New function replace()
titanic['sex'] = titanic['sex'].replace(to_replace='female', value=1)
titanic['sex'] = titanic['sex'].replace(to_replace='male', value=0)
print(titanic.head())

#Calculating z-score:
s_mean = titanic.sex.mean()
s_std = titanic.sex.std()

#replace column with each entry's z-score
titanic.sex = (titanic.sex - s_mean)/s_std
print(titanic.head())
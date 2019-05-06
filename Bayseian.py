import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Importing dataset
data = pd.read_csv("#")  # Replace # with path of Dataset
print(data.head())
# Convert categorical variable to numeric
data["M/F_C"]=np.where(data["M/F"]=="M",0,1)
data["Group_C"]=np.where(data["Group"]=="Demented",0,1)
# Cleaning dataset of NaN
data=data[['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'Group_C']].dropna(axis=0, how='any')



def bayseianClassifier():
    data = pd.read_csv("#")  # Replace # with path of Dataset
    print(data.head())
    # Convert categorical variable to numeric
    data["M/F_C"]=np.where(data["M/F"]=="M",0,1)
    data["Group_C"]=np.where(data["Group"]=="Demented",0,1)
    # Cleaning dataset of NaN
    data=data[['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'Group_C']].dropna(axis=0, how='any')


    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))



    gnb = GaussianNB()
    used_features =['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

    # Train classifier
    gnb.fit(
        X_train[used_features].values,
        X_train["Group_C"]
    )
    y_pred = gnb.predict(X_test[used_features])
    
    # Print results
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}% ==> Naive Bayes, "
        .format(
            X_test.shape[0],
            (X_test["Group_C"] != y_pred).sum(),
            100*(1-(X_test["Group_C"] != y_pred).sum()/X_test.shape[0])
    ))


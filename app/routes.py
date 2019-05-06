from app import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')





@app.route('/bayseian', methods = ['GET', 'POST'])
def bayseianClassifier():
    
    data = pd.read_csv("app/oasis_longitudinal.csv")
    print(data.head())
    # Convert categorical variable to numeric
    data["M/F_C"] = np.where(data["M/F"] == "M", 0, 1)
    data["Group_C"] = np.where(data["Group"] == "Demented", 0, 1)
    # Cleaning dataset of NaN
    data = data[['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV',
                 'nWBV', 'ASF', 'Group_C']].dropna(axis=0, how='any')

    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(
        data, test_size=0.5, random_state=int(time.time()))

    gnb = GaussianNB()
    used_features = ['M/F_C', 'Age', 'EDUC',
                     'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

    # Train classifier
    gnb.fit(
        X_train[used_features].values,
        X_train["Group_C"]
    )
    y_pred = gnb.predict(X_test[used_features])

    tableData = []
    print(len(y_pred), len(X_train))
    i = 0
    for index, row in X_test.iterrows():
        dic = dict(row)
        # print(row[0])
        dic["Group"] = "Affected" if dic["Group_C"] else "Not Affected"
        dic["Prediction"] = "Affected" if y_pred[i] else "Not Affected"
        dic["Gender"] = "Male" if dic["M/F_C"] else "Female"
        dic["green"] = True if (dic['Group'] == dic["Prediction"]) else False
        print(dic["green"])
        i = i+1
        tableData.append(dic)

    return render_template("table.html", data=tableData, green=dic["green"], model="Bayesian Classifier", accur=100*(1-(X_test["Group_C"] != y_pred).sum()/X_test.shape[0]))


@app.route('/svm')
def svm():

    data = pd.read_csv("app/oasis_longitudinal.csv")
    print(data.head())
    # Convert categorical variable to numeric
    data["M/F_C"] = np.where(data["M/F"] == "M", 0, 1)
    data["Group_C"] = np.where(data["Group"] == "Demented", 0, 1)
    # Cleaning dataset of NaN
    data = data[['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV',
                 'nWBV', 'ASF', 'Group_C']].dropna(axis=0, how='any')

    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, random_state=int(time.time()))

    used_features = ['M/F_C', 'Age', 'EDUC',
                     'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

    svclassifier = SVC(kernel='poly', degree=1, gamma="auto")
    svclassifier.fit(X_train[used_features].values, X_train["Group_C"])
    y_pred = svclassifier.predict(X_test[used_features])

    tableData = []
    i = 0
    for index, row in X_test.iterrows():
        dic = dict(row)
        # print(row[0])
        dic["Group"] = "Affected" if dic["Group_C"] else "Not Affected"
        dic["Prediction"] = "Affected" if y_pred[i] else "Not Affected"
        dic["Gender"] = "Male" if dic["M/F_C"] else "Female"
        dic["green"] = True if (dic['Group'] == dic["Prediction"]) else False
        print(dic["green"])
        i = i+1
        tableData.append(dic)

    return render_template("table.html", data=tableData, green=dic["green"], model="Support Vector machine", accur=100*(1-(X_test["Group_C"] != y_pred).sum()/X_test.shape[0]))


@app.route('/ann')
def ann():
    dataset = pd.read_csv("app/oasis_longitudinal.csv")


    dataset['SES'].fillna((dataset['SES'].median()), inplace=True)
    dataset['MMSE'].fillna((dataset['MMSE'].median()), inplace=True)
    dataset['CDR'].fillna((dataset['CDR'].mean()), inplace=True)
    dataset['eTIV'].fillna((dataset['eTIV'].mean()), inplace=True)
    dataset['nWBV'].fillna((dataset['nWBV'].mean()), inplace=True)
    dataset['ASF'].fillna((dataset['ASF'].mean()), inplace=True)
    used_features = ['CDR', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

    x = dataset.iloc[:, 4:15].values
    y = dataset.iloc[:, 2:3].values
    labelencoder_X_1=LabelEncoder()
    x[:,1]=labelencoder_X_1.fit_transform(x[:,1])
    labelencoder_X_2=LabelEncoder()
    x[:,2]=labelencoder_X_2.fit_transform(x[:,2])
    onehotencoder=OneHotEncoder(categorical_features=[1])
    x=onehotencoder.fit_transform(x).toarray()
    x=x[:,1:]
    labelencoder_y_1=LabelEncoder()
    onehotencoder1=OneHotEncoder(categorical_features=[0])
    y[:,0]=labelencoder_y_1.fit_transform(y[:,0])
    y=onehotencoder1.fit_transform(y).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
    x,y,test_size=0.25, random_state=0)
    
    sc=StandardScaler()
    x_train=sc.fit_transform(X_train)
    x_test=sc.transform(X_test)
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=3,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

    y_pred=classifier.predict(x_test)
    # classifier.fit(x_test,y_test,batch_size=2,nb_epoch=100)
    tableData = []
    i = 0
    # print(y_pred,x_test)
    new_prediction=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
    print(new_prediction)
    # for index, row in X_test.iterrows():
    #     dic = dict(row)
    #     print(row[0])
    #     dic["Group"] = "Affected" if dic["Group_C"] else "Not Affected"
    #     dic["Prediction"] = "Affected" if y_pred[i] else "Not Affected"
    #     dic["Gender"] = "Male" if dic["M/F_C"] else "Female"
    #     dic["green"] = True if (dic['Group'] == dic["Prediction"]) else False
    #     print(dic["green"])
    #     i = i+1
    #     tableData.append(dic)
    return("Done")
    return render_template("table.html", data=tableData, green=dic["green"], model="Artificial Neural Network", accur=100*(1-(X_test["Group_C"] != y_pred).sum()/X_test.shape[0]))


@app.route('/table', methods = ['GET', 'POST'])
def showTab():
    return render_template("table.html")



	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return 'file uploaded successfully'+f.filename
    #return render_template('upload.html')

@app.route('/cat', methods = ['GET', 'POST'])
def showCat():
    return render_template("category.html")

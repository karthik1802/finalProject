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
from flask import Flask, render_template, request, url_for, redirect
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')





@app.route('/bayesian', methods = ['GET', 'POST'])
def bayesian():
    
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
    data = pd.read_csv("app/oasis_longitudinal.csv")
    print(data.head())
    # Convert categorical variable to numeric
    data["M/F_C"] = np.where(data["M/F"] == "M", 0, 1)
    data["Group_C"] = np.where(data["Group"] == "Demented", 0, 1)
    # Cleaning dataset of NaN
    data = data[['M/F_C', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV',
                 'nWBV', 'ASF', 'Group_C','CDR']].dropna(axis=0, how='any')

   
    used_features = ['CDR', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

    
    X_train, X_test = train_test_split(data,test_size=0.5, random_state=0)
    
    sc=StandardScaler()
    x_train=sc.fit_transform(X_train[used_features])
    x_test=sc.transform(X_test[used_features])

    # I already found out the tuned hyper parameters so commenting the code.

   
    

    # def build_classifier(optimizer):
    #     classifier = Sequential()
    #     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    #     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    #     classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    #     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    #     return classifier
    # classifier = KerasClassifier(build_fn = build_classifier)
    # parameters = {'batch_size': [1, 5],
    #             'epochs': [100, 120],
    #             'optimizer': ['adam', 'rmsprop']}
    # grid_search = GridSearchCV(estimator = classifier,
    #                         param_grid = parameters,
    #                         scoring = 'accuracy',
    #                         cv = 10)
    # grid_search = grid_search.fit(X_train[used_features].values, X_train['Group_C'])


    # best_parameters = grid_search.best_params_
    # best_accuracy = grid_search.best_score_
    # print("best_parameters: ")
    # print(best_parameters)
    # print("\nbest_accuracy: ")
    # print(best_accuracy)
    # return("Done!!")

    #####RESULTS######
    # best_parameters: {'batch_size': 1, 'epochs': 100, 'optimizer': 'rmsprop'}

    # best_accuracy: 0.978021978022


    classifier = Sequential() # Initialising the ANN

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

    classifier.fit(x_train, X_train["Group_C"], batch_size = 1, epochs = 100)

    y_pred=classifier.predict(x_test)
    # classifier.fit(x_test,y_test,batch_size=2,nb_epoch=100)
    tableData = []
    i = 0
    print(y_pred)
    tr=0
    a = False
    for index, row in X_test.iterrows():
        dic = dict(row)
        print(row[0])
        dic["Group"] = "Affected" if dic["Group_C"] else "Not Affected"
        dic["Prediction"] = "Affected" if y_pred[i]>=0.5 else "Not Affected"
        dic["Gender"] = "Male" if dic["M/F_C"] else "Female"
        dic["green"] = True if (dic['Group'] == dic["Prediction"]) else False
        print(dic["green"])
        if dic["Group"] == dic["Prediction"]:
            tr=tr+1
        i = i+1
        tableData.append(dic)
    # return("Done")
    return render_template("ann.html", data=tableData, green=dic["green"], model="Artificial Neural Network", accur=100*(tr/i))


@app.route('/table', methods = ['GET', 'POST'])
def showTab():
    return render_template("table.html")



	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return redirect(url_for('cat'))
    #return render_template('upload.html')

@app.route('/cat', methods = ['GET', 'POST'])
def cat():
    return render_template("category.html")

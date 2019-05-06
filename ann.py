import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"oasis_longitudinal.csv")

dataset['SES'].fillna((dataset['SES'].median()),inplace=True)
dataset['MMSE'].fillna((dataset['MMSE'].median()),inplace=True)
dataset['CDR'].fillna((dataset['CDR'].mean()),inplace=True)
dataset['eTIV'].fillna((dataset['eTIV'].mean()),inplace=True)
dataset['nWBV'].fillna((dataset['nWBV'].mean()),inplace=True)
dataset['ASF'].fillna((dataset['ASF'].mean()),inplace=True)

x=dataset.iloc[:, 4:15].values
y=dataset.iloc[:, 2:3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.25, random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=3,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

y_pred=classifier.predict(x_test)
classifier.fit(x_test,y_test,batch_size=2,nb_epoch=100)

new_prediction=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))

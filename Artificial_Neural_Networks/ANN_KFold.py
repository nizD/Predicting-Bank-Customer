import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

data=pd.read_csv('Churn_Modelling.csv')

X=data.iloc[:,3:13].values
y=data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X1=LabelEncoder()
X[:,1]=le_X1.fit_transform(X[:,1])
le_X2=LabelEncoder()
X[:,2]=le_X2.fit_transform(X[:,2])
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X)
X=X.toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit(X_test)

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_ANN():
    ANN=Sequential()
    ANN.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    ANN.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    ANN.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    ANN.compile(optimizer= 'adam',loss= 'binary_crossentropy',metrics = ['accuracy'])
    return ANN

classifier=KerasClassifier(build_fn=build_ANN,batch_size=10,epochs=100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)




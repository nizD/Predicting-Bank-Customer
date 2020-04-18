import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
#datapreprocessing
data=pd.read_csv('Churn_Modelling.csv')

X=data.iloc[:,3:13].values
y=data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X1=LabelEncoder()
X[:,1]=le_X1.fit_transform(X[:,1])
le_X2=LabelEncoder()
X[:,2]=le_X2.fit_transform(X[:,2])
X=np.array(X,dtype='float')
ohe_X1=OneHotEncoder(categorical_features=[1])
X=ohe_X1.fit_transform(X)
X=X.toarray()
X=X[:,1:]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#building ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense
#initialising ANN
ANN=Sequential()
#adding input layer and first hidden layer
ANN.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
#adding second hidden layer

ANN.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#building an output layer

ANN.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#compiling the ANN
ANN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
ANN.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#checking performance
Z=['France',600,'Male',40,3,60000,2,1,1,50000]
y_pred=ANN.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


'''
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000

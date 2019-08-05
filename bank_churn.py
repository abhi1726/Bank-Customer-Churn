
import pandas as pd
import numpy as np

bank = pd.read_csv('Bank Churn.csv')

bank.head()

bank.tail()

bank = bank.drop("RowNumber", axis=1)

bank = bank.drop("Surname", axis=1)
bank.head(5)

bank = bank.drop("CustomerId", axis=1)

feats = ['Geography','Gender']
bank_final = pd.get_dummies(bank,columns=feats,drop_first=True)
print(bank_final)

bank_final.head(5)

from sklearn.model_selection import train_test_split

x= bank_final.drop(['Exited'], axis=1).values
y= bank_final['Exited'].values

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

print(x_train)

print(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier= Sequential()

classifier.add(Dense(2, kernel_initializer = "uniform", activation= "relu", input_dim=11))

classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))

classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])

classifier.fit(x_train, y_train, batch_size=5, epochs = 15)

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from  keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(2, kernel_initializer = "uniform", activation= "relu", input_dim=11))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
    classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = make_classifier, batch_size=5, nb_epoch =1)

accuracies = cross_val_score(estimator = classifier, X = x_train, y= y_train, cv =5 )

mean = accuracies.mean()
mean

variance = accuracies.var()
variance

from keras.layers import Dropout
    
classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=11))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])

from sklearn.model_selection import GridSearchCV
def make_clasifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))
    classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = make_classifier)

def build_classifier(optimizer = 'adam'):
  
  classifier.compile(optimizer=optimizer , loss = 'binary_crossentropy' , 
  metrics=['accuracy'])
  
  return classifier

params = {
    'batch_size':[20,35],
    'epochs':[2,3],
    'optimizer':['adam','rmsprop']
}

grid_search = GridSearchCV(estimator=classifier,
                          param_grid=params,
                          scoring="accuracy",
                          cv=2)

!pip install scikit-learn --user

batch_size = [20, 35]
epochs = [2, 3]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x, y)

grid_result =grid_search.fit(x, y)

"""Applying RFE (Recursive feature elimination) and then building logistics regression model to predict which variable might impact the employees attrition rate."""

# %matplotlib inline
import matplotlib.pyplot as plt

bank_final.shape

#applying RFE to filter best suitable variables for our model

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, 5)
rfe = rfe.fit(x, y)
print(rfe.support_)
print(rfe.ranking_)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import model_from_json

# Importing the train dataset
train_set = pd.read_csv('train.csv')
X = train_set.iloc[:, 0:3].values
y = train_set.iloc[:, 3].values

test_set = pd.read_csv('test.csv')
X_ts = test_set.iloc[:, 0:].values

# Label encoding for the training set
from sklearn.preprocessing import LabelEncoder #OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])

# Label encoding for the test set
X_ts[:, 0] = labelencoder_X_1.fit_transform(X_ts[:, 0])
X_ts[:, 1] = labelencoder_X_2.fit_transform(X_ts[:, 1])
X_ts[:, 2] = labelencoder_X_3.fit_transform(X_ts[:, 2])

# Further Splitting the train dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_ts = sc.transform(X_ts)

#Import libraries for Evaluating improving and tuning the ANN 
from keras.wrappers.scikit_learn import KerasRegressor
#from keras.layers import Dropout
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
#Keras libraries
from keras.models  import Sequential
from keras.layers import Dense
def build_model():
   model = Sequential()
   model.add(Dense(activation = 'relu', input_dim = 3, units = 6, kernel_initializer = 'uniform',  ))
   #model.add(Dropout(rate = 0.2))
   model.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform', ))
   #model.add(Dropout(rate = 0.2))
   model.add(Dense(units = 1, kernel_initializer = 'uniform', ))
   model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])
   return model

estimator = KerasRegressor(build_fn=build_model, epochs=100, batch_size=10)
kfold = KFold(n_splits=10, shuffle=False, random_state= None)
estimator.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Cross validation on the test set
results = cross_val_score(estimator,X_train, y_train, cv=kfold)
prediction = cross_val_predict(estimator, X_test, y_test, cv=kfold, n_jobs=-1)

#Prediction on the split test set
prediction = estimator.predict(X_test)

#Variants of scoring
msle = mean_squared_log_error(y_test, prediction)
mse = mean_squared_error(y_test,prediction)
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)
from math import sqrt
rmse = sqrt(mse) #root mean --

#Prediction on the test.csv set given by DSN
prediction_Xts = estimator.predict(X_ts)
#format as type float
prediction_Xts2 = prediction_Xts.astype(float)
#write predictions to file
with open("predictions.txt", "a") as my_file:
    for i in prediction_Xts2:
        my_file.write('%.4f\n' % i)
    my_file.close()

#Visualizing keras model
from keras.utils import plot_model
plot_model(estimator.model, to_file='model.png')

# Save model to disk
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model 
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])


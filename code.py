from scipy.ndimage import label
import numpy as np
from sklearn import  datasets
from sklearn.model_selection import  train_test_split
from LineraRegression import LineraRegression
import matplotlib.pyplot as plt

## helper to calculate mean square error
def mean_square_error(y_true,y_predict):
    return  np.mean((y_true-y_predict)**2)

## create data
X, y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=.2,random_state=1234)

## train  model
regressor = LineraRegression(learning_rate=.01,n_iters=1000)
regressor.fit(X,y)

## predict using model
prediction = regressor.predict(X_test)

## test model
mse = mean_square_error(y_test,prediction)
print(f"Mes : {mse}")

## draw predictable with data
y_pred_line =  regressor.predict(X)
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,color="red",s=10)
m2 = plt.scatter(X_test,y_test,color="blue",s=10)
plt.plot(X,y_pred_line,color="black",linewidth=2,label="prediction")
plt.show()




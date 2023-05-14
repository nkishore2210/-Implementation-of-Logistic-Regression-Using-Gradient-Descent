# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: N.Kishore
RegisterNumber: 212222240049

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=",")
X=data[:,[0,1]]
Y=data[:,2]

X[:5]

Y[:5]

plt.figure()
plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0], 1)), X))
theta= np.array([0,0,0])
J, grad= costFunction(theta, X_train, Y)
print(J)
print(grad)  

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, Y)
print(J)
print(grad)

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun = cost, x0 = theta, args = (X_train, Y), method = "Newton-CG", jac = gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[Y == 1][:, 0], X[Y ==1][:, 1], label="Admitted")
    plt.scatter(X[Y == 0][:, 0], X[Y ==0][:, 1], label=" Not Admitted")
    plt.contour(xx,yy,Y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
    
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
prob

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X)==Y)
*/
```

## Output:

![Screenshot 2023-05-14 104825](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/8d77fd79-3070-4e2d-abcf-3385416281c9)

![Screenshot 2023-05-14 104903](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/8cb09ba8-23a7-44f2-b1c6-b74fc78a9059)

![Screenshot 2023-05-14 104926](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/3fc121ef-8d86-414a-ab00-8d863f0b26dd)

![Screenshot 2023-05-14 104949](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/f3454b49-ef37-427b-8a90-a44d7e96d211)

![Screenshot 2023-05-14 105010](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/48203f98-2ae8-4c40-8135-f2d58bdb2933)

![Screenshot 2023-05-14 105026](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/117082c4-e521-4206-8c19-67f7ff9b0a2e)

![Screenshot 2023-05-14 105138](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/ad8c53dd-2237-4f5b-b01e-67971c0447ac)

![Screenshot 2023-05-14 105200](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/fa82db44-2f65-49f9-9223-465f3e076660)

![Screenshot 2023-05-14 105226](https://github.com/nkishore2210/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707090/2e917aac-9c58-452b-985c-e21261ee36ec)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


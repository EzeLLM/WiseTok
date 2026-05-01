import numpy as np
from sklearn import datasets,linear_model,metrics 

diabetes = datasets.load_diabetes()

diabetes_x = diabetes.data #matrix of dimension 442 x 10


 #split the data into training testing sets
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

x = diabetes_x_train
y = diabetes_y_train

#assigning initial weights to featrures
w = np.random.uniform(low = -0.1 , high = 0.1 , size = diabetes_x.shape[1] )
b = 0.0

#assigning learning rate and no. of steps
lRate = 0.094
epochs = 100590

for  i in range(epochs):

    #calculate predictions
    y_pred = x.dot(w) + b

    #calculating cost and mean squared error
    cost = (y-y_pred)**2
    meanSquaredError =  np.mean(cost)

    #calculate gradients
    w_grad = -(1/len(x))*(y_pred - y).dot(x)
    b_grad = -(1/len(x))*(np.sum(y_pred - y))

    #updating parameters
    w+=lRate*w_grad
    b+=lRate*b_grad

    if(i%10000==0):
        print("epoch %d : %.2f" %(i , meanSquaredError))

#now testing data

y_pred_test = diabetes_x_test.dot(w) + b

print("the coefficients are : \n" , w)

cost = (y_pred_test - diabetes_y_test)**2
meanSquaredError = np.mean(cost)

print("the mean squared error is : %.2f \n" %meanSquaredError)

print("learning rate : %f \n epochs : %d" %(lRate , epochs))

print("="*140)


















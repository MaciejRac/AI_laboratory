import numpy as np
import matplotlib.pyplot as plt
import random

from data import get_data, inspect_data, split_data

def calculate_closed_form_solution(x, y):
    # Rozwiązanie jawne (optymalny wektor parametrów): theta = (X^T * X)^(-1) * X^T * y
    X = np.c_[np.ones((len(x), 1)), x]
    theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta_best

def calculate_mse(y_true, y_pred):
    #blad modelu na podstawie mean square error
    return np.mean((y_true - y_pred) ** 2)

def z_score_normalization(x):
    mean = np.mean(x)
    std_dev = np.std(x)
    return (x - mean) / std_dev, mean, std_dev

def gradient_mse(x, y, theta):
    X = np.c_[np.ones((len(x), 1)), x]
    m = len(y)  # Liczba obserwacji
    gradient = (2/m) * np.dot(X.T, np.dot(X,theta)-y)  # Obliczanie gradientu
    #print(np(X.T))
    #print(np.dot(X,theta))
    return gradient

def prediction(x,theta):
    return np.dot(np.c_[np.ones((len(x), 1)), x], theta)
    

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

theta_best = [0, 0]

# TODO: calculate closed-form solution
theta_best = calculate_closed_form_solution(x_train, y_train)
#print("theta closed-form solution:", theta_best)


# TODO: calculate error
y_pred = prediction(x_train,theta_best)
mse_closed_form = calculate_mse(y_train, y_pred)
print("MSE of train using Closed-form Solution:", mse_closed_form)

y_pred = prediction(x_test,theta_best)
mse_closed_form = calculate_mse(y_test, y_pred)
print("MSE of test using Closed-form Solution:", mse_closed_form)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: standardization
x_train_normalized, mean, std_dev = z_score_normalization(x_train)
x_test_normalized = (x_test - mean) / std_dev

y_train_normalized, mean, std_dev = z_score_normalization(y_train)
y_test_normalized = (y_test - mean) / std_dev

# TODO: calculate theta using Batch Gradient Descent
theta = [np.random.random(),np.random.random()]
learning_rate = 0.01
max_iter = 100000000
tol = 1e-6
prev_cost = float('inf')

for i in range(max_iter):
    y_pred = prediction(x_train_normalized,theta)
    #predictions = np.dot(np.c_[np.ones((len(x_train_normalized), 1)), x_train_normalized], theta)
    gradients = gradient_mse(x_train_normalized, y_train_normalized, theta)
    theta -= learning_rate * gradients
    cost = calculate_mse(y_train_normalized, y_pred)
    if abs(prev_cost - cost) < tol:
        print(f"Converged after {i} iterations.")
        break
    
    prev_cost = cost


# TODO: calculate error
y_pred =  prediction(x_train_normalized,theta)
mse_standarization = calculate_mse(y_train_normalized, y_pred)
print("MSE of train using standarization:", mse_standarization)

y_pred = prediction(x_test_normalized,theta)
mse_standarization= calculate_mse(y_test_normalized, y_pred)
print("MSE of test using standarization:", mse_standarization)

#odstandaryzpwanie thety
#sx= np.std(x_train)
#sy=np.std(y_train)
#theta[1]=theta[1]*(sy/sx)
#theta[0]=np.mean(y_train)-theta[1]*np.mean(x_train)
#print(theta[1], theta[0])

# plot the regression line
x = np.linspace(min(x_test_normalized), max(x_test_normalized), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_normalized, y_test_normalized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

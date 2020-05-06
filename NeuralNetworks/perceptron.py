import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[1,0,1],[0,0,1],[1,1,0],[1,1,1]]).T
y_train = np.array([1,0,1,1]).T

def sigmoid(x):
  return 1/(1+np.exp(-x))

def initialize(input_dim):
  w = np.random.random(size=input_dim)*0.01
  return w

def forward(x_input, w):
  return sigmoid(np.dot(w,x_input))

def compute_derivative(x):
  return x*(1-x)


def compute_error_backprop(current_output, actual_output):
  return (actual_output-current_output)*compute_derivative(current_output)


def update_wts(w, error, x_train, learning_rate):
  w += learning_rate*np.dot(error, x_train.T)

def train_model(x_train, y_train, learning_rate, nepochs):
  w = initialize((1,3))
  for i in range(nepochs):
    pred_out = forward(x_train, w)
    error = compute_error_backprop(pred_out, y_train)
    update_wts(w,error, x_train,learning_rate)
  return w, pred_out


def test(x_test, w):
  return forward(x_test,w)


x1 = int(input("First number"))
x2 = int(input("Second Number"))
x3 = int(input("Third Number"))

if x1 in range(2) and x2 in range(2) and x3 in range(2):
  w, pred_out = train_model(x_train, y_train, 1, 400)
  x_test = np.array([x1, x2, x3]).T
  ans = forward(x_test,w)
  print("Predicted output is")
  print(1 if answer[0]>=0.5 else 0)

  

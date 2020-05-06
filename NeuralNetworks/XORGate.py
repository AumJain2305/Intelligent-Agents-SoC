# this file implements a XOR gate for two binary values
import numpy as np
import matplotlib.pyplot as plt
import random

x_train = np.array([[1,1],[1,0],[0,1],[0,0]]).T
y_train = np.array([[0,1,1,0]])

def sigmoid(x):
	return 1/(1+np.exp(-x))

def initialize():
	random.seed(1)
	w1 = np.random.random(size=(2,2))
	b1 = np.zeros((2,1),dtype=float)
	w2 = np.random.random((1,2))
	b2 = np.array([0.])
	parameters = {
			'w1':w1,
			'b1':b1,
			'w2':w2,
			'b2':b2
	}
	return parameters


def forward(x_input, parameters):
	w1=parameters['w1']
	w2=parameters['w2']
	b1=parameters['b1']
	b2=parameters['b2']
	Z1 = np.dot(w1,x_input) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(w2, A1) + b2
	A2 = sigmoid(Z2)
	# print(A2.shape, "A2",np.log(A2).shape)
	# print(A2)
	return Z1,Z2,A1,A2


def get_cost(A2, Y):
	# I am using the logarithmic loss function
	# print(np.multiply(Y, np.log(A2)).shape)
	# print(np.dot(1-Y, np.log(1-A2)).shape)
	cost = np.sum(-1*(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))))/4
	return cost


def backpropagation(X, parameters, Z1, A1, Z2, A2, Y):
	# d(variable) stands for dCost/dVariable here
	w1=parameters['w1']
	b1=parameters['b1']
	w2=parameters['w2']
	b2=parameters['b2']
	dA2 = (1-Y)/(1-A2) - Y/A2
	dZ2 = A2 - Y
	dW2 = np.dot(dZ2, A1.T)
	db2 = np.sum(dZ2, axis=1).reshape(b2.shape)
	dZ1 = np.multiply(np.dot(w2.T, dZ2),(1-np.power(A1,2)))
	dW1 = np.dot(dZ1,X.T)
	db1 = (np.sum(dZ1,axis=1)/4).reshape(2,1)
	# print(db1.shape)
	return dW2, db2, dW1, db1

def update(parameters, dW2,db2,dW1,db1,lr):
  w1 = parameters['w1']
  w2=parameters['w2']
  b1=parameters['b1']
  b2=parameters['b2']
  db1 = db1.reshape(2,1)
  # print(b1.shape,db1.shape)
  w1 -= lr*dW1
  w2-= lr*dW2
  b1 -= lr*db1
  b2 -= float(lr*db2)
  parameters = {
      'w1':w1,
      'w2':w2,
      'b1':b1,
      'b2':b2
  }
  return parameters


def train(X,Y,n_epochs=400,lr=1):
  parameters = initialize()
  costs = []
  for i in range(n_epochs):
    Z1,Z2,A1,A2 = forward(X, parameters)
    cost = get_cost(A2,Y)
    dW2,db2,dW1,db1 = backpropagation(X, parameters, Z1,A1,Z2,A2,Y)
    parameters = update(parameters, dW2,db2,dW1,db1,lr)
    costs.append(cost)
    # print("haha")
  return costs, parameters


def main():
	num1 = int(input("First bit"))
	num2 = int(input("Second bit"))
	costs, parameters = train(x_train,y_train)
	inp_array = np.array([[num1,num2]]).T
	Z1,Z2,A1,A2 = forward(inp_array,parameters)
	print("The xor of numbers is ", 1 if A2>0.5 else 0)
	plt.plot(costs)
	# print("haha")


if __name__ == "__main__":
	main()

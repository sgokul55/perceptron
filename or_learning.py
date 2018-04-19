from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1

training_data = [ (array([0,0,1]), 0), (array([0,1,1]), 1), (array([1,0,1]), 1), (array([1,1,1]), 1), ]
#training_data = [ (array([0,0,1]), 0), (array([0,1,1]), 1), (array([1,0,1]), 0), (array([1,1,1]), 1), ]
#training_data = [ (array([0,0,1]), 0), (array([0,1,1]), 0), (array([1,0,1]), 1), (array([1,1,1]), 1), ]

w = random.rand(3)

errors = [] 
eta = 0.2 
n = 100
print("initial weight {}".format(w))
for i in range(n): 
	x, expected = choice(training_data) 
	print("input {}, Expected {}".format(x[:], expected))
	result = dot(w, x) 
	print("W * X = {}".format(result))
	# L1 Regularization
	error = expected - unit_step(result)
	print("Error {}".format(error))
	# errors are lot of 0's knows as sparsity (property of the L1 regu)
	errors.append(error) 
	w += eta * error * x
	print("Modified Weight {}".format( w ))
print("final weight {}".format(w))
#print(errors)
for x, _ in training_data: 
	result = dot(x, w) 
	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
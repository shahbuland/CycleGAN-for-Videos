import numpy as np

# Both functions below work under assumption that
# Input is image data on [0,255]
def sigmoid_normalize(x):
	return x/255

def tanh_normalize(x):
	return x/127.5 - 1

# Applies a function to a video
# Applying directly may not be feasible due to size of video array
def video_apply(A, f):
	total_size = A.shape[0]
	for i in range(total_size):
		A[i] = f(A[i])
	return A
	

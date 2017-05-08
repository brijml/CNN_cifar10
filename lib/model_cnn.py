import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def softmax(var):
	exponent = np.exp(var)
	return exponent/sum(exponent)

class Conv():

	def __init__(self,stride=1,pad=1,F=3,depth=3,N=6,fanin=1):
		self.stride = stride
		self.pad = pad
		self.F = F
		self.depth = depth
		self.N = N
		self.filters = []
		self.fanin = fanin
		self.bias = 0.01 * np.random.randn(1)
		for i in range(self.N):
			fil = np.random.randn(self.F,self.F,self.depth)/np.sqrt(self.fanin/2.0)
			self.filters.append(fil)
		
	def forward(self,activations_below):

		m,n = activations_below.shape[:2]
		self.out = np.zeros((m,n,self.N))
		for i in range(self.N):
			a = ndimage.convolve(activations_below,self.filters[i],mode = 'constant',cval = 0.0) + self.bias
			for j in range(a.shape[2]):
				self.out[:,:,i] += a[:,:,j]

		return self.out

	def backward(self,error_derivatives_above):

		self.error_derivatives = []
		for i in range(self.N):
			error_derivatives_y += ndimage.convolve(error_derivatives_above,self.filters[i],mode = 'constant',cval = 0.0)
			self.error_derivatives_w[i] = activations_below * error_derivatives_above
		self.error_derivatives_bias = error_derivatives_above
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		for i in range(self.N):
			self.filters -= learning_rate * self.error_derivatives_w[i]

class ReLU():

	def __init__(self):
		pass

	def rectify(self,activations_below):
		mask = activations_below < 0
		activations_below[mask] = 0
		self.out = activations_below
		self.local_grad = np.array(mask,dtype = np.uint8)
		return self.out

	def backward(self,error_derivatives_above):
		error_derivatives_y = error_derivatives_above * self.local_grad
		return error_derivatives_y

class Pool(object):
	"""docstring for Pool"""
	def __init__(self, stride=1,F=2):
		super(Pool, self).__init__()
		self.stride = stride
		self.F = F

	def max_pooling(self,activations_below):
		m,n,p = activations_below.shape
		# print m,n,p
		out = np.zeros(((m-self.F)/self.stride+1,(n-self.F)/self.stride+1,p))
		# print out.shape
		self.local_grad = np.zeros((m,n,p))
		# print activations_below[0:0+self.F,0:0+self.F,0].shape

		for k in range(p):
			# print k	
			for i in range(0,m,self.stride):
				for j in range(0,n,self.stride):
					# print i,j
					# print activations_below[i:i+self.F,j:j+self.F,k].shape
					t = activations_below[i:i+self.F,j:j+self.F,k].reshape(self.F*self.F)
					out[(i-self.F)/self.stride+1,(j-self.F)/self.stride+1,k] = max(t)
					self.local_grad[i:i+self.F,j:j+self.F,k] = np.array(t == max(t),dtype = np.uint8).reshape(self.F,self.F)

		return out

	def backward(self,error_derivatives_above):
		error_derivatives_y = error_derivatives_above * self.local_grad
		return error_derivatives_y
		
class FC(object):

	def __init__(self,H,fanin):
		self.H = H
		self.fanin = fanin
		self.weights = np.random.randn(self.H,self.fanin)/np.sqrt(self.fanin)
		self.bias = 0.01 * np.random.randn(1)
		return

	def forward(self,activations_below):
		print activations_below.shape,self.weights.shape,self.bias.shape
		self.out = np.matmul(self.weights,activations_below) + self.bias#softmax(activations_below * self.weights + self.bias)
		self.local_grad = self.weights
		return self.out

	def backward(self,error_derivatives_above):
		error_derivatives_y = error_derivatives_above * self.local_grad
		self.error_derivatives_w = error_derivatives_above * self.out
		self.error_derivatives_bias = error_derivatives_above
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		self.weights -= learning_rate * self.error_derivatives_w
		self.bias -= learning_rate * self.error_derivatives_bias
		return

class Softmax(object):

	def __init__(self,H,fanin):
		self.H = H
		self.fanin = fanin
		self.weights = np.random.randn(self.H,self.fanin)/np.sqrt(self.fanin)
		# self.bias = 0.01 * np.random.randn(1)
		return

	def forward(self,activations_below):

		self.out = softmax(np.matmul(self.weights,activations_below))# + self.bias)
		print self.out.shape
		self.local_grad = np.zeros(len(self.out))
		for i,value in enumerate(self.out):
			self.local_grad[i] = self.out[i] * (1 - self.out[i])
			for j in range(len(self.out)):
				self.local_grad[j] += -1 * self.out[i] * self.out[j]

		return self.out	

	def backward(self,target):
		error_derivatives_y = target - self.out
		self.error_derivatives_w = activations_below * self.local_grad
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		self.weights -= learning_rate * self.error_derivatives_w
		return


if __name__ == '__main__':
	pass
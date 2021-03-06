import numpy as np
import matplotlib.pyplot as plt
import os,time
from scipy import ndimage

def softmax(var):
	exponent = np.exp(var)
	return exponent/sum(exponent)

def zero_padding(input_array,pad=1):
	m,n,p = input_array.shape
	out_array = np.zeros((m+2*pad,n+2*pad,p))
	for i in range(p):
		out_array[pad:m+pad,pad:n+pad,i] = input_array[:,:,i]
	return out_array

def convolve(image,filter_,pad):
	out = np.zeros(image.shape[:2])
	image = zero_padding(image,pad)
	m_i,n_i,p_i = image.shape
	m_f,n_f,p_f = filter_.shape

	filter_ = rot180(filter_)
	
	for i in range(m_i-m_f+1):
		for j in range(n_i-n_f+1):
			for ch in range(p_i):
				out[i,j] += np.sum(image[i:i+m_f,j:j+n_f,ch]*filter_[:,:,ch])

	return out

def rot180(arr):
	return np.rot90(np.rot90(arr))

class Conv():

	"""This is the convolutional layer of the net.

	Attributes
		attr1(int) : Defines the stride with which the filter progresses on input volume
		attr2(int) : Defines the amount of padding is to be done to the length and width of the input volume.
		attr3(int) : Size of the input filter generally a single number i.e. the filter is square in shape.
		attr4(int) : Defines the depth of the filter generally equal to the depth of input volume.
		attr5(int) : The number of filter for the layer.
		attr6(int) : The number of input size received by the layer.
	"""


	def __init__(self,stride=1,pad=1,F=3,depth=3,N=6,fanin=1, filter_param = None, bias = None):
		self.stride = stride
		self.pad = pad
		self.F = F
		self.depth = depth
		self.N = N
		self.filters = []
		self.fanin = fanin
		self.v = np.zeros((self.N,self.F,self.F,self.depth))
		if filter_param == None:
			self.bias = 0.01 * np.random.randn(1)
			self.filters = np.random.randn(N,F,F,depth)/np.sqrt(self.fanin/2.0)
		else:
			self.filters = filter_param
			self.bias = bias

		self.error_derivatives_w = np.zeros_like(self.filters)
		self.error_derivatives_bias = np.zeros_like(self.bias)

	def forward(self,activations_below):

		"""
		The Forward progpogation for the convolutional layer. The output is the convolution of all the filters defined for the layer
		If the size of the input volume is m*n*p, the size of filter is F*F*p, there are N such filters then the ouput size is m*n*N.

		Args:
		    param1: The input image for the first convolutional layer and the output of the max-pool layer for subsequent layer.

		Returns:
		    param2: The ouput volume after the convolution of the input with the filters.

		"""
		self.input_to_conv = activations_below
		self.m,self.n = activations_below.shape[:2]
		out = np.zeros((self.m,self.n,self.N))
		for i in range(self.N):
			out[:,:,i] = convolve(activations_below,self.filters[i],self.pad) + self.bias

		return out

	def backward(self,error_derivatives_above):
		
		"""
		The backpropogation for the layer performed similar to that of the input.
		"""
		error_derivatives_above =  rot180(error_derivatives_above)
		self.input_to_conv = rot180(self.input_to_conv)
		error_derivatives_y = np.zeros((self.m,self.n,self.depth))
		for i in range(self.depth):
			for j in range(self.N):
				error_derivatives_y[:,:,i] += ndimage.convolve(error_derivatives_above[:,:,j],self.filters[j][:,:,i],mode = 'constant',cval = 0.0)

		t = self.input_to_conv
		row, column = t.shape[:2]
		for result_depth in range(self.N): # depth of result 
			one_filter_derivative = np.zeros((self.F,self.F,self.depth))
			for d in range(self.depth):
				for u in range(self.F): # Filter width
					for v in range(self.F): # Filter height
						for x in range(self.m - self.F):  # Input image width
							for y in range(self.n - self.F): #  Input image height
					
								one_filter_derivative[u,v,d] += t[x+u,y+v,d]*error_derivatives_above[x,y,result_depth]						

			self.error_derivatives_w[result_depth] += one_filter_derivative
			self.error_derivatives_bias += np.mean(error_derivatives_above)

		return error_derivatives_y


	def update(self,learning_rate,momentum):


		self.v = momentum * self.v - learning_rate * self.error_derivatives_w
		self.filters += self.v
		self.bias -= learning_rate * self.error_derivatives_bias
		self.error_derivatives_w = np.zeros_like(self.filters)
		self.error_derivatives_bias = np.zeros_like(self.bias)

		return

class ReLU():

	
	def __init__(self):
		pass

	def rectify(self,activations_below):

		mask = activations_below < 0
		mask1 = activations_below > 0
		activations_below[mask] = 0
		out = activations_below
		self.local_grad = np.array(mask1,dtype = np.uint8)

		return out

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
		self.m,self.n,self.p = activations_below.shape
		out = np.zeros(((self.m-self.F)/self.stride+1,(self.n-self.F)/self.stride+1,self.p))
		self.local_grad = np.zeros((self.m,self.n,self.p))

		for k in range(self.p):	
			for i in range(0,self.m,self.stride):
				for j in range(0,self.n,self.stride):
					t = activations_below[i:i+self.F,j:j+self.F,k].reshape(self.F*self.F) #Reshaping 2*2 to 4
					max_ = max(t)
					if max_ != 0:
						out[(i-self.F)/self.stride+1,(j-self.F)/self.stride+1,k] = max_
						self.local_grad[i:i+self.F,j:j+self.F,k] = np.array(t == max(t),dtype = np.uint8).reshape(self.F,self.F)

		return out

	def backward(self,error_derivatives_above):

		error_derivatives_y = np.zeros((self.m,self.n,self.p))		
		for k in range(self.p):	
			for i in range(0,self.m,self.stride):
				for j in range(0,self.n,self.stride):
					t1 = self.local_grad[i:i+self.F,j:j+self.F,k].reshape(self.F*self.F)\
					 * error_derivatives_above[(i-self.F)/self.stride+1,(j-self.F)/self.stride+1,k]
					error_derivatives_y[i:i+self.F,j:j+self.F,k] = t1.reshape(self.F,self.F)
	
		return error_derivatives_y
		
class FC(object):

	def __init__(self,H,fanin, weights = None,bias = None):
	
		self.H = H
		self.fanin = fanin
		self.v = 0
		if weights == None:
			self.weights = np.random.randn(self.fanin,self.H)/np.sqrt(self.fanin)
			self.bias = 0.01 * np.random.randn(H,1)
		else:
			self.weights = weights
			self.bias = bias

		self.error_derivatives_w = np.zeros_like(self.weights)
		self.error_derivatives_bias = np.zeros_like(self.bias)

		return

	def forward(self,activations_below):
	
		out = np.matmul(self.weights.T,activations_below)# + self.bias
		self.activations_MP = activations_below
		return out

	def backward(self,error_derivatives_above):
	
		error_derivatives_y = np.matmul(self.weights,error_derivatives_above)
		self.error_derivatives_w += np.matmul(self.activations_MP,error_derivatives_above.T)
		self.error_derivatives_bias += error_derivatives_above
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		self.v = momentum * self.v - learning_rate * self.error_derivatives_w
		self.weights += self.v
		self.bias -= learning_rate * self.error_derivatives_bias
		self.error_derivatives_w = np.zeros_like(self.weights)
		self.error_derivatives_bias = np.zeros_like(self.bias)

		return

class Softmax(object):

	def __init__(self,H,fanin, weights, bias):
		self.H = H
		self.fanin = fanin
		self.v = 0
		if weights == None:
			self.weights = np.random.randn(self.fanin,self.H)/np.sqrt(self.fanin)
			self.bias = 0.01 * np.random.randn(H,1)
		else:
			self.weights = weights
			self.bias = bias
		self.error_derivatives_w = np.zeros_like(self.weights)
		self.error_derivatives_bias = np.zeros_like(self.bias)

		return

	def forward(self,activations_below, weight_decay = None):

		self.out = softmax(np.matmul(self.weights.T,activations_below) + self.bias) #+ weight_decay * np.atleast_2d(np.sum(self.weights,axis=1)).T
		self.activations_FC = activations_below
		return self.out	

	def backward(self,target):

		error_derivatives_ISM = self.out - np.atleast_2d(target).T
		self.error_derivatives_w += np.matmul(self.activations_FC,error_derivatives_ISM.T)
		self.error_derivatives_bias += error_derivatives_ISM
		delta_FC = np.matmul(self.weights,error_derivatives_ISM)

		return delta_FC

	def update(self,learning_rate,momentum):

		self.v = momentum * self.v - learning_rate * self.error_derivatives_w
		self.weights += self.v
		self.bias -= learning_rate*self.error_derivatives_bias
		self.error_derivatives_w = np.zeros_like(self.weights)
		self.error_derivatives_bias = np.zeros_like(self.bias)

		return


if __name__ == '__main__':
	pass
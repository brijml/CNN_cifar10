import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def softmax(var):
	#print var
	exponent = np.exp(var)
	#print exponent
	return exponent/sum(exponent)

def zero_padding(input_array,pad=1):
	m,n,p = input_array.shape
	out_array = np.zeros((m+2*pad,n+2*pad,p))
	for i in range(p):
		out_array[pad:m+pad,pad:n+pad,i] = input_array[:,:,i]
	return out_array

class Conv():

	def __init__(self,stride=1,pad=1,F=3,depth=3,N=6,fanin=1):
		self.stride = stride
		self.pad = pad
		self.F = F
		self.depth = depth
		self.N = N
		self.filters = []
		self.fanin = fanin
		# self.bias = 0.01 * np.random.randn(1)
		self.error_derivatives_w = []
		for i in range(self.N):
			fil = np.random.randn(self.F,self.F,self.depth)/np.sqrt(self.fanin/2.0)
			self.filters.append(fil)
		
	def forward(self,activations_below):

		# self.input_to_conv = activations_below
		self.m,self.n = activations_below.shape[:2]
		self.out = np.zeros((self.m,self.n,self.N))
		a = np.zeros((self.m,self.n,self.depth))
		for i in range(self.N):
			for j in range(self.depth):
				a[:,:,j] = ndimage.convolve(activations_below[:,:,j],self.filters[i][:,:,j],mode = 'constant',cval = 0.0) #+ self.bias
			for k in range(a.shape[2]):
				self.out[:,:,i] += a[:,:,k]
		return self.out

	def backward(self,error_derivatives_above):
		error_derivatives_y = np.zeros((self.m,self.n,self.N))
		for i in range(self.N):
			error_derivatives_y += ndimage.convolve(error_derivatives_above,self.filters[i],mode = 'constant',cval = 0.0)

		t = zero_padding(self.out,pad = self.pad)
		row, column = t.shape[:2]
		# print 't_size',t.shape,error_derivatives_above.shape

		# error_derivatives_above, t, 
		print 'backprop'
		for result_depth in range(self.N):
			one_filter_derivative = np.zeros((self.F,self.F,self.depth))
			
			for x in range(self.m):
				for y in range(self.n):
					for u in range(self.F):
						for v in range(self.F):

							R_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),0]
							G_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),1]
							B_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),2]

							one_filter_derivative[u,v,0] = np.sum(R_slice) * error_derivatives_above[x,y,result_depth]
							one_filter_derivative[u,v,1] = np.sum(G_slice) * error_derivatives_above[x,y,result_depth]
							one_filter_derivative[u,v,2] = np.sum(B_slice) * error_derivatives_above[x,y,result_depth]


			self.error_derivatives_w.append(one_filter_derivative)
			# print len(self.error_derivatives_w)

		return error_derivatives_y


	def update(self,learning_rate,momentum):
		print 'update', len(self.error_derivatives_w), len(self.filters)
		for i in range(self.N):
			self.filters[i] -= learning_rate * self.error_derivatives_w[i]

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
		self.m,self.n,self.p = activations_below.shape
		out = np.zeros(((self.m-self.F)/self.stride+1,(self.n-self.F)/self.stride+1,self.p))
		self.local_grad = np.zeros((self.m,self.n,self.p))

		for k in range(self.p):	
			for i in range(0,self.m,self.stride):
				for j in range(0,self.n,self.stride):
					t = activations_below[i:i+self.F,j:j+self.F,k].reshape(self.F*self.F)
					out[(i-self.F)/self.stride+1,(j-self.F)/self.stride+1,k] = max(t)
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

	def __init__(self,H,fanin):
		self.H = H
		self.fanin = fanin
		self.weights = np.random.randn(self.fanin,self.H)/np.sqrt(self.fanin)
		self.bias = 0.01 * np.random.randn(1)
		return

	def forward(self,activations_below):
		self.out = np.matmul(self.weights.T,activations_below)# + self.bias
		self.local_grad = self.weights
		return self.out

	def backward(self,error_derivatives_above):
		error_derivatives_y = np.matmul(self.local_grad,error_derivatives_above)
		self.error_derivatives_w = error_derivatives_above * self.out
		# self.error_derivatives_bias = error_derivatives_above
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		self.weights -= learning_rate * self.error_derivatives_w
		# self.bias -= learning_rate * self.error_derivatives_bias
		return

class Softmax(object):

	def __init__(self,H,fanin):
		self.H = H
		self.fanin = fanin
		self.weights = np.random.randn(self.H,self.fanin)/np.sqrt(self.fanin)
		# self.bias = 0.01 * np.random.randn(1)
		return

	def forward(self,activations_below,weight_decay):

		self.out = softmax(np.matmul(self.weights,activations_below)) + weight_decay * np.atleast_2d(np.sum(self.weights,axis=1)).T
		self.local_grad = np.zeros(len(self.out))
		for i,value in enumerate(self.out):
			self.local_grad[i] = self.out[i] * (1 - self.out[i])
			for j in range(len(self.out)):
				if i == j:
					continue
				self.local_grad[j] += -1 * self.out[i] * self.out[j]

		self.input_to_softmax = activations_below
		return self.out	

	def backward(self,target):
		error_derivatives_y = np.atleast_2d(target).T - self.out
		self.error_derivatives_w = self.input_to_softmax * self.local_grad
		return error_derivatives_y

	def update(self,learning_rate,momentum):
		self.weights -= learning_rate * self.error_derivatives_w
		return	


if __name__ == '__main__':
	pass
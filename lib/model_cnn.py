import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage

class Conv():

	def __init__(self,stride=1,pad=1,F=3,depth=3,N=6):
		self.stride = stride
		self.pad = pad
		self.F = F
		self.depth = depth
		self.N = np
		self.filters = []

	def init_filters(self,fanin):

		#filters = []
		for i in range(self.N):
			fil = np.random.randn((self.F,self.F,self.depth),dtype = 'float')/np.sqrt(fanin/2.0)
			self.filters.append(fil)

		return
		

	def forward(self,input_prev):

		self.out = []
		for i in range(self.N):
			self.out.append(ndimage.convolve(input_prev,self.filters[i],mode = 'constant',cval = 0.0))

		return self.out

	def backward(self,error_derivatives_above):

		self.error_derivatives = []
		for i in range(self.N):
			error_derivatives = #Need to figure it out


class ReLU():

	def __init__(self):
		pass

	def rectify(self,input_to_layer):
		input_to_layer[input_to_layer < 0] = 0

		return input_to_layer

class Pool(object):
	"""docstring for Pool"""
	def __init__(self, stride=1,F=2):
		super(Pool, self).__init__()
		self.stride = stride
		self.F = F

	def max_pooling(self,input_to_layer):
		m,n,p = input_to_layer.shape

		out = np.zeros(((m-self.F)/self.stride,(n-self.F)/self.stride,p),dtype = np.uint8)
		
		for k in range(p):	
			for i in range(0,m,self.stride):
				for j in range(0,n,self.stride):
					t = input_to_layer[i:i+self.F,j:j+self.F,p].reshape(self.F*self.F)
					out[i,j,p] = max(t)

		return out
		



if __name__ == '__main__':
	pass
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
	# print 'hi'
	out = np.zeros(image.shape[:2])
	# print out.shape
	# print filter_.shape
	image = zero_padding(image,pad)
	m_i,n_i,p_i = image.shape
	m_f,n_f,p_f = filter_.shape

	filter_ = np.rot90(np.rot90(filter_))
	# print filter_.shape
	for ch in range(p_i):
		for i in range(m_i-m_f+1):
			for j in range(n_i-n_f+1):
			#for i_f in range(m_f/2,m_f/2+1):
				#for j_f in range(n_f/2,n_f/2+1):
				out[i,j] = np.sum(image[i:i+m_f,j:j+n_f,ch]*filter_[:,:,ch])
				# print out[i,j]

	return out

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


	def __init__(self,stride=1,pad=1,F=3,depth=3,N=6,fanin=1):
		self.stride = stride
		self.pad = pad
		self.F = F
		self.depth = depth
		self.N = N
		self.filters = []
		self.fanin = fanin
		# self.bias = 0.01 * np.random.randn(1)
		for i in range(self.N):
			fil = np.random.randn(self.F,self.F,self.depth)/np.sqrt(self.fanin/2.0)
			self.filters.append(fil)
		
		# print len(self.filters)
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
		self.out = np.zeros((self.m,self.n,self.N))
		print self.out.shape
		# a = np.zeros((self.m,self.n,self.depth))
		for i in range(self.N):
			self.out[:,:,i] = convolve(activations_below,self.filters[i],self.pad)
		# 	for j in range(self.depth):
		# 		a[:,:,j] = ndimage.convolve(activations_below[:,:,j],self.filters[i][:,:,j],mode = 'constant',cval = 0.0) #+ self.bias
		# 	for k in range(a.shape[2]):
		# 		self.out[:,:,i] += a[:,:,k]
			# print self.out[0:5,0:5,i]
			# tajfak = raw_input()
		return self.out

	def backward(self,error_derivatives_above):
		
		"""
		The backpropogation for the layer performed similar to that of the input.
		"""

		self.error_derivatives_w = []
		error_derivatives_y = np.zeros((self.m,self.n,self.N))
		for i in range(self.N):
			error_derivatives_y += ndimage.convolve(error_derivatives_above,self.filters[i],mode = 'constant',cval = 0.0)

		t = zero_padding(self.out,pad = self.pad)
		row, column = t.shape[:2]
		# print 't_size',t.shape,error_derivatives_above.shape

		# error_derivatives_above, t, 
		print 'backprop'
		for result_depth in range(self.N): # depth of result 
			one_filter_derivative = np.zeros((self.F,self.F,self.depth))
			for x in range(self.m):  # Input image width
				for y in range(self.n): #  Input image height
					for u in range(self.F): # Filter width
						for v in range(self.F): # Filter height
							for d in range(self.depth):								
								_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),d]
								# print 'slice', _slice.shape
								# G_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),1]
								# B_slice = t[u:row-(self.F-1-u), v:column-(self.F-1-v),2]

								one_filter_derivative[u,v,d] = np.sum(_slice) * error_derivatives_above[x,y,result_depth]
								# one_filter_derivative[u,v,1] = np.sum(G_slice) * error_derivatives_above[x,y,result_depth]
								# one_filter_derivative[u,v,2] = np.sum(B_slice) * error_derivatives_above[x,y,result_depth]

			# print 'filter',result_depth, '\n',  one_filter_derivative
			# xxxxx = raw_input()
			self.error_derivatives_w.append(one_filter_derivative)
			# print len(self.error_derivatives_w)

		return error_derivatives_y


	def update(self,learning_rate,momentum):
		print 'update', len(self.error_derivatives_w), len(self.filters)
		for i in range(self.N):
			print learning_rate , self.error_derivatives_w[i]
			self.filters[i] -= learning_rate * self.error_derivatives_w[i]

class ReLU():

	
	def __init__(self):
		pass

	def rectify(self,activations_below):
		mask = activations_below < 0
		mask1 = activations_below > 0
		activations_below[mask] = 0
		self.out = activations_below
		self.local_grad = np.array(mask1,dtype = np.uint8)
		for k in range(activations_below.shape[2]):

			print activations_below[0:5,0:5,k],'\n\n',self.local_grad[0:5,0:5,k],'\n\n',self.out[0:5,0:5,k]
			k = raw_input()	
		# time.sleep(1)
		return self.out

	def backward(self,error_derivatives_above):
		# print error_derivatives_above
		# time.sleep(1)
		error_derivatives_y = error_derivatives_above * self.local_grad
		# print error_derivatives_y
		# kkk = raw_input()
		# del kkk
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

			# print activations_below[0:6,0:6,k],'\n\n',out[0:3,0:3,k],'\n\n',self.local_grad[0:6,0:6,k]
			# r = raw_input()
		return out

	def backward(self,error_derivatives_above):

		error_derivatives_y = np.zeros((self.m,self.n,self.p))		
		for k in range(self.p):	
			for i in range(0,self.m,self.stride):
				for j in range(0,self.n,self.stride):
					t1 = self.local_grad[i:i+self.F,j:j+self.F,k].reshape(self.F*self.F)\
					 * error_derivatives_above[(i-self.F)/self.stride+1,(j-self.F)/self.stride+1,k]
					error_derivatives_y[i:i+self.F,j:j+self.F,k] = t1.reshape(self.F,self.F)
		# print 'error_derivatives_y\n',error_derivatives_y
		# kkk = raw_input()
		# print 'error_derivatives_above\n',error_derivatives_above
		# kkk = raw_input()
			# print error_derivatives_above[0:3,0:3,k],'\n\n',error_derivatives_y[0:6,0:6,k],'\n\n',self.local_grad[0:6,0:6,k]
			# r = raw_input()
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
		self.activations_MP = activations_below
		return self.out

	def backward(self,error_derivatives_above):
		error_derivatives_y = np.matmul(self.local_grad,error_derivatives_above)
		# print self.activations_MP
		# r = raw_input()
		self.error_derivatives_w = np.matmul(self.activations_MP,error_derivatives_above.T)
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
		self.weights = np.random.randn(self.fanin,self.H)/np.sqrt(self.fanin)
		# self.bias = 0.01 * np.random.randn(1)
		return

	def forward(self,activations_below,weight_decay):

		# print activations_below
		self.out = softmax(np.matmul(self.weights.T,activations_below)) #+ weight_decay * np.atleast_2d(np.sum(self.weights,axis=1)).T
		# self.local_grad = np.zeros(len(self.out))
		# for i,value in enumerate(self.out):
		# 	self.local_grad[i] = self.out[i] * (1 - self.out[i])
		# 	for j in range(len(self.out)):
		# 		if i == j:
		# 			continue
		# 		self.local_grad[j] += -1 * self.out[i] * self.out[j]

		self.activations_FC = activations_below
		return self.out	

	def backward(self,target):
		error_derivatives_ISM = np.atleast_2d(target).T - self.out
		self.error_derivatives_w = np.matmul(self.activations_FC,error_derivatives_ISM.T)
		delta_FC = np.matmul(self.weights,error_derivatives_ISM)
		# print delta_FC.shape,'hi',error_derivatives_ISM.shape,self.weights.shape
		print delta_FC.shape
		return delta_FC

	def update(self,learning_rate,momentum):
		self.weights -= learning_rate * self.error_derivatives_w
		return	


if __name__ == '__main__':
	pass
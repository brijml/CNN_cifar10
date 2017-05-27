import numpy as np
from lib import *
import os
import matplotlib.pyplot as plt
import cPickle
import time

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_path,'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
del X_train
del Y_train
image = X_test[0]
m,n,p = image.shape
nc = 10 #number of classes
# plt.ion()


def test(**kwargs):
	initialise = kwargs['initialise']

	parameters_ = {"conv1":None, "conv2":None, 	"conv3":None, 	"fc":None, "softmax":None, 	"bias1":None , "bias2":None,"bias3":None }
	if initialise==0:
		read_path = os.path.join(file_path, 'network_parameters')
		read_path = os.path.join(read_path, 'network_parameters_2.pickle')
		with open(read_path, "rb") as input_file:
			parameters_ = cPickle.load(input_file)
	# print  parameters_['conv3'][0][:,:,0]

	# temp = parameters_['conv1']
	# for i in range(6):
	# 	for j in range(3):
	# 		print temp[i][:,:,j]
	# for var in temp:
	# 	print var[var>=0].shape
	# 	print var[var<0].shape
	# print np.sum(np.uint8(var[var>=0]))

	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=6,fanin=m*n*6, filter_param = parameters_['conv1'], bias = parameters_['bias1'])
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=6,N=6,fanin=m*n*10, filter_param = parameters_['conv2'], bias = parameters_['bias2'])
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=8,N=8,fanin=m*n*10, filter_param = parameters_['conv3'], bias =  parameters_['bias3'])
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(H =50,fanin = 128, weights = parameters_['fc'])
	softmax = Softmax(H=10,fanin = 50, weights = parameters_['softmax'])





	
	for i,img in enumerate(X_test):
		# plt.imshow(image)
		# plt.show()
		out_conv1 = conv1.forward(img)
		# print out_conv1[:,:,0]	
		out_relu1 = relu1.rectify(out_conv1)
		# print out_relu1[:,:,0]
		out_pool1 = pool1.max_pooling(out_relu1)
		out_conv2 = conv2.forward(out_pool1)
		out_relu2 = relu2.rectify(out_conv2)
		out_pool2 = pool2.max_pooling(out_relu2)
		out_conv3 = conv3.forward(out_pool2)
		out_relu3 = relu3.rectify(out_conv3)
		out_pool3 = pool3.max_pooling(out_relu3)
		out_pool3 = out_pool3.reshape(128,1)
		out_full = full.forward(out_pool3)
		out_softmax = softmax.forward(out_full)
		# print out_softmax

		plt.bar(range(nc), out_softmax)
		plt.title(str(Y_test[i]))
		plt.pause(1)
		plt.clf()
		time.sleep(3)

if __name__ == '__main__':
	test(initialise = 0)
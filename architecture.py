import numpy as np
from lib import *
import os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
image = X_train[0]
m,n,p = image.shape
nc = 10 #number of classes

def one_hot(index):
	probability = np.zeros(nc)
	probability[index] = 1
	return probability

def train(**kwargs):

	learning_rate,momentum,batch,epoch,wd = kwargs['learning_rate'],kwargs['momentum'],kwargs['batch'],\
											kwargs['epoch'],kwargs['weight_decay']

	number_samples = X_train.shape[0]
	number_samples_batch = number_samples/batch

	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=16,fanin=m*n*16)
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=16,N=20,fanin=m*n*16)
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=20,N=20,fanin=m*n*20)
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(H =50,fanin = 320)
	softmax = Softmax(H=10,fanin = 50)

	iters = 0
	while iters < epoch:
		for i,image in enumerate(X_train):
			out_conv1 = conv1.forward(image)
			out_relu1 = relu1.rectify(out_conv1)
			out_pool1 = pool1.max_pooling(out_relu1)
			out_conv2 = conv2.forward(out_pool1)
			out_relu2 = relu2.rectify(out_conv2)
			out_pool2 = pool2.max_pooling(out_relu2)
			out_conv3 = conv3.forward(out_pool2)
			out_relu3 = relu3.rectify(out_conv3)
			out_pool3 = pool3.max_pooling(out_relu3)
			# print out_pool1.shape
			out_pool3 = out_pool3.reshape(320,1)
			out_full = full.forward(out_pool3)
			out_softmax = softmax.forward(out_full,wd)
			target = one_hot(Y_train[i])
			print out_softmax#.shape,np.atleast_2d(target).T.shape
			error = np.sum(abs(np.atleast_2d(target).T - out_softmax))
			print error


			grad_softmax = softmax.backward(target)
			# print grad_softmax.shape
			grad_full = full.backward(grad_softmax)
			grad_full = grad_full.reshape(4,4,20)
			# print grad_full.shape
			grad_pool3 = pool3.backward(grad_full)
			# # print grad_pool3.shape
			grad_relu3 = relu3.backward(grad_pool3)
			# # print grad_relu3.shape
			grad_conv3 = conv3.backward(grad_relu3)
			# # print grad_conv3.shape
			grad_pool2 = pool2.backward(grad_conv3)
			# # print grad_pool2.shape
			grad_relu2 = relu2.backward(grad_pool2)
			# # print grad_relu2.shape
			grad_conv2 = conv2.backward(grad_relu2)
			# print grad_conv2.shape
			grad_pool1 = pool1.backward(grad_conv2)
			# print grad_pool1.shape
			grad_relu1 = relu1.backward(grad_pool1)
			# print grad_relu1.shape
			grad_conv1 = conv1.backward(grad_relu1)
			# print grad_conv1.shape

			conv1.update(learning_rate,momentum)
			# conv2.update(learning_rate,momentum)
			# conv3.update(learning_rate,momentum)
			full.update(learning_rate,momentum)
			softmax.update(learning_rate,momentum)

			# print np.atleast_2d(target).T - out_softmax
		iters+=1

if __name__ == '__main__':
	# model = init_model()
	train(epoch = 1,learning_rate = 0.00001,momentum = 0.9,weight_decay = 0.001,batch=5)
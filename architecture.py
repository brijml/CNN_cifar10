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

	learning_rate,momentum,batch,epoch = kwargs['learning_rate'],kwargs['momentum'],kwargs['batch'],\
											kwargs['epoch']

	number_samples = X_train.shape[0]
	number_samples_batch = number_samples/batch

	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=16,fanin=m*n)
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=3,N=20,fanin=m/2*n/2)
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=3,N=20,fanin=m/4*n/4)
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(H =10,fanin = 320)
	softmax = Softmax(10,10)

	iters = 0
	while iters < epoch:
		for i,image in enumerate(X_train):
			print image.shape
			out_conv1 = conv1.forward(image)
			print out_conv1.shape
			out_relu1 = relu1.rectify(out_conv1)
			print out_relu1.shape
			out_pool1 = pool1.max_pooling(out_relu1)
			print out_pool1.shape
			out_conv2 = conv2.forward(out_pool1)
			print out_conv2.shape
			out_relu2 = relu2.rectify(out_conv2)
			print out_relu2.shape
			out_pool2 = pool2.max_pooling(out_relu2)
			print out_pool2.shape
			out_conv3 = conv3.forward(out_pool2)
			print out_conv3.shape
			out_relu3 = relu3.rectify(out_conv3)
			print out_relu3.shape
			out_pool3 = pool3.max_pooling(out_relu3)
			print out_pool3.shape
			out_pool3 = out_pool3.reshape(320,1)
			out_full = full.forward(out_pool3)
			print out_full.shape
			out_softmax = softmax.forward(out_full)
			print out_softmax.shape
			target = one_hot(Y_train[i])

			grad_softmax = softmax.backward(target)
			grad_full = full.backward(grad_softmax)
			grad_pool3 = pool3.backward(grad_full)
			grad_relu3 = relu3.backward(grad_pool3)
			grad_conv3 = conv3.backward(grad_relu3)
			grad_pool2 = pool3.backward(grad_conv3)
			grad_relu2 = relu3.backward(grad_pool2)
			grad_conv2 = conv3.backward(grad_relu2)
			grad_pool1 = pool3.backward(grad_conv2)
			grad_relu1 = relu3.backward(grad_pool1)
			grad_conv1 = conv3.backward(grad_relu1)

			conv1.update(learning_rate,momentum)
			conv2.update(learning_rate,momentum)
			conv3.update(learning_rate,momentum)
			full.update(learning_rate,momentum)
			softmax.update(learning_rate,momentum)

			error = np.sum(target - out_softmax)
		
		iters+=1

if __name__ == '__main__':
	# model = init_model()
	train(epoch = 1,learning_rate = 0.01,momentum = 0.9,weight_decay = 0.001,batch=5)
import numpy as np
from lib import *
import os
import matplotlib.pyplot as plt
import cPickle

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_path,'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
image = X_train[0]
m,n,p = image.shape
nc = 10 #number of classes
# X_train = X_train[45:46] #Overfit on 50 training examples to validate the implementation

def one_hot(index):
	probability = np.zeros(nc)
	probability[index] = 1
	return probability

def dump_parameters(conv1, conv2, conv3, full, softmax, iters, i):

	save_path = os.path.join(file_path, 'network_parameters')
	dict_ = {"conv1":conv1.filters, "conv2":conv2.filters,
	"conv3":conv3.filters, 	"fc":full.weights, "softmax":softmax.weights}

	filename = os.path.join(save_path, 'parameters_epoch_'+str(iters) + '_sample_' + str(i) + '.pickle')

	with open(filename, 'wb"') as output_file:
		cPickle.dump(dict_, output_file)
	print 'parameters saved for : ', filename

def train(**kwargs):

	learning_rate,momentum,batch,epoch,wd = kwargs['learning_rate'],kwargs['momentum'],kwargs['batch'],\
											kwargs['epoch'],kwargs['weight_decay']

	number_samples = X_train.shape[0]
	number_samples_batch = number_samples/batch

	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=6,fanin=m*n*6)
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=6,N=10,fanin=m*n*10)
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=10,N=10,fanin=m*n*10)
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(H =50,fanin = 160)
	softmax = Softmax(H=10,fanin = 50)
	plt.ion()
	iters = 0
	val = []
	while iters < epoch:
		error = []
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
			out_pool3 = out_pool3.reshape(160,1)
			out_full = full.forward(out_pool3)
			out_softmax = softmax.forward(out_full,wd)
			target = one_hot(Y_train[i])
			error.append(np.sum(abs(np.atleast_2d(target).T - out_softmax)))
			# print out_softmax

			grad_softmax = softmax.backward(target)
			grad_full = full.backward(grad_softmax)
			grad_full = grad_full.reshape(4,4,10)
			grad_pool3 = pool3.backward(grad_full)
			grad_relu3 = relu3.backward(grad_pool3)
			grad_conv3 = conv3.backward(grad_relu3)
			grad_pool2 = pool2.backward(grad_conv3)
			grad_relu2 = relu2.backward(grad_pool2)
			grad_conv2 = conv2.backward(grad_relu2)
			grad_pool1 = pool1.backward(grad_conv2)
			grad_relu1 = relu1.backward(grad_pool1)
			grad_conv1 = conv1.backward(grad_relu1)

			conv1.update(learning_rate,momentum)
			conv2.update(learning_rate,momentum)
			conv3.update(learning_rate,momentum)
			full.update(learning_rate,momentum)
			softmax.update(learning_rate,momentum)

			if i%500 == 0:
				dump_parameters(conv1, conv2, conv3, full, softmax, iters, i)		
			# print np.atleast_2d(target).T - out_softmax
		dump_parameters(conv1, conv2, conv3, full, softmax, iters, i)
		
		val.append(sum(error)/len(error))
		plt.plot(val)
		plt.pause(0.01)
		iters+=1


if __name__ == '__main__':
	train(epoch = 20,learning_rate = 0.001,momentum = 0.9,weight_decay = 0.001,batch=5)
import numpy as np
from lib import *
import os
import matplotlib.pyplot as plt
import cPickle

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_path,'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
del X_test; del Y_test
image = X_train[0]
m,n,p = image.shape
nc = 10 #number of classes
offset = 0 #Let this be your first trianing example
X_train = X_train[offset:] #Overfit on 50 training examples to validate the implementation
Y_train = Y_train[offset:]

def one_hot(index):
	probability = np.zeros(nc)
	probability[index] = 1
	return probability

def dump_parameters(conv1,  conv2,  conv3,  full, softmax, iters, i, param):

	dict_ = {"conv1":conv1.filters, "conv2":conv2.filters,
	"conv3":conv3.filters, 	"fc":full.weights, "softmax":softmax.weights,
	"bias1":conv1.bias , "bias2":conv2.bias,"bias3":conv3.bias,"bias4":full.bias,"bias5":softmax.bias}

	if param == 1:
		save_path1 = os.path.join(file_path, 'temp_parameters')
		filename1 = os.path.join(save_path1, 'parameters_epoch_'+str(iters) + '_sample_' + str(i) + '.pickle')
		with open(filename1, 'wb"') as output_file:
			cPickle.dump(dict_, output_file)
		print 'parameters saved for : ', filename1

	if param == 2:	
		save_path2 = os.path.join(file_path, 'network_parameters')
		filename2 = os.path.join(save_path2, 'network_parameters_' + str(iters) + '.pickle')
		with open(filename2, 'wb"') as output_file:
			cPickle.dump(dict_, output_file)
		print 'parameters saved for : ', filename2


def train(**kwargs):

	learning_rate,momentum,batch,epoch,wd, initialise = kwargs['learning_rate'],kwargs['momentum'],kwargs['batch'],\
											kwargs['epoch'],kwargs['weight_decay'], kwargs['initialise']

	number_samples = X_train.shape[0]
	number_samples_batch = number_samples/batch

	parameters_ = {"conv1":None, "conv2":None, 	"conv3":None, 	"fc":None, "softmax":None, 	"bias1":None , "bias2":None,"bias3":None,"bias4":None,"bias5":None }

	if initialise==0:
		print "\n\n*******Reading from pickle file***********\n\n"
		read_path = os.path.join(file_path, 'network_parameters')
		read_path = os.path.join(read_path, 'network_parameters_5.pickle')
		with open(read_path, "rb") as input_file:
			parameters_ = cPickle.load(input_file)

	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=6,fanin=m*n*6, filter_param = parameters_['conv1'], bias = parameters_['bias1'])
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=6,N=6,fanin=m*n*6, filter_param = parameters_['conv2'], bias = parameters_['bias2'])
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=6,N=8,fanin=m*n*8, filter_param = parameters_['conv3'], bias =  parameters_['bias3'])
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(H =50,fanin = 128, weights = parameters_['fc'],bias = parameters_['bias4'])
	softmax = Softmax(H=10,fanin = 50, weights = parameters_['softmax'],bias = parameters_['bias5'])
	plt.ion()
	iters = 0
	val = []
	for iters in range(2, epoch,1):
		error,loss = [],[]
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
			out_pool3 = out_pool3.reshape(128,1)
			out_full = full.forward(out_pool3)
			out_softmax = softmax.forward(out_full,wd)


			target = one_hot(Y_train[i])

			#Negative log likelihood
			loss.append(-np.log(out_softmax[Y_train[i]]))
			
			grad_softmax = softmax.backward(target)
			grad_full = full.backward(grad_softmax)
			grad_full = grad_full.reshape(4,4,8)
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

			#Plot the cumulative loss for the 500 training samples and save the parameters
			if i%500 == 0:
				dump_parameters(conv1, conv2, conv3, full, softmax, iters, i + offset, 1)
				val.append(sum(loss)/len(loss))
				plt.plot(val)
				plt.pause(0.01)
				error = []

		#Save the parameters after every iteration through training data
		dump_parameters(conv1, conv2, conv3, full, softmax, iters, i + offset, 2)		


if __name__ == '__main__':
	initialise = 0

	train(epoch = 6,learning_rate = 1e-5,momentum = 0.9,weight_decay = 0.001,batch=5, initialise = initialise)

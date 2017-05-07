import numpy as np
import lib,os

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
image = X_train[0][0]
m,n,p = image.shape
nc = 10 #number of classes

def one_hot(index):
	probability = np.zeros(nc)
	probability[index] = 1
	return probability

def train(**kwargs):
	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=16,fanin=m*n)
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	conv2 = Conv(F=5,stride=1,pad=2,depth=3,N=20,fanin=m/2*n/2)
	relu2 = ReLU()
	pool2 = Pool(stride=2,F=2)
	conv3 = Conv(F=5,stride=1,pad=2,depth=3,N=20,fanin=m/4*n/4)
	relu3 = ReLU()
	pool3 = Pool(stride=2,F=2)
	full = FC(10)
	softmax = Softmax(10)

	while iters < epoch:
		for i,batch in enumerate(X_train):
			for j,image in enumerate(batch):
				out_conv1 = conv1.forward(image)
				out_relu1 = relu1.forward(out_conv1)
				out_pool1 = pool1.forward(out_relu1)
				out_conv2 = conv1.forward(out_pool1)
				out_relu2 = relu1.forward(out_conv2)
				out_pool2 = pool1.forward(out_relu2)
				out_conv3 = conv1.forward(out_pool2)
				out_relu3 = relu1.forward(out_conv3)
				out_pool3 = pool1.forward(out_relu3)
				out_full = full.forward(out_pool3)
				out_softmax = softmax.forward(out_full)
				target = one_hot(X_test[i][j])

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

				conv1.update()
				conv2.update()
				conv3.update()
				full.update()
				softmax.update()

				error = np.sum(target - out_softmax)
		
		iters+=1

if __name__ == '__main__':
	# model = init_model()
	train(epoch = 1,learning_rate = 0.01,momentum = 0.9,weight_decay = 0.001,batch=4)
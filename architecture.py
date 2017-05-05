import numpy as np
import lib

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'cifar-10-batches-py')
X_train,Y_train,X_test,Y_test = load_CIFAR10(data_dir)
image = X_train[0][0]
m,n,p = image.shape

def train(model,**kwargs):
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


if __name__ == '__main__':
	# model = init_model()
	train(epoch = 1,learning_rate = 0.01,momentum = 0.9,weight_decay = 0.001,batch=4)
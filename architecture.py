import numpy as np
import lib

if __name__ == '__main__':
	conv1 = Conv(F=5,stride=1,pad=2,depth=3,N=16)
	relu1 = ReLU()
	pool1 = Pool(stride=2,F=2)
	
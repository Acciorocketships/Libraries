import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *


def test_stereo():

	stream = Stream(mode='img',src='stereotest')
	img1 = stream.get()
	img2 = stream.get()

	f = 1000
	K = np.zeros((3,3))
	K[0,0] = f
	K[1,1] = f
	K[2,2] = 1
	K[0,2] = img1.shape[0]
	K[1,2] = img1.shape[1]

	R = np.eye(3)
	T = np.array([0,-1,0])

	depth = stereoDepth(img1, img2, R, T, K)

	print(depth)


if __name__ == '__main__':
	test_stereo()
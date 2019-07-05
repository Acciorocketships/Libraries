import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *


# NOT VERIFIED


def test_homography():

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

	H, Rs, Ts, Ns = homography(img1, img2, K)

	print("H")
	print(H)
	print("Rs")
	print(Rs)
	print("Ts")
	print(Ts)
	print("Ns")
	print(Ns)

	pt1 = np.array([50,110,1])
	pt2 = np.linalg.inv(H) @ pt1
	pt2 = pt2 / pt2[2]

	print("pt1")
	print(pt1)
	print("pt2")
	print(pt2)

	img1 = Stream.mark(img1, (pt1[1], pt1[0]), xyaxis=True)
	img2 = Stream.mark(img2, (pt2[1], pt2[0]), xyaxis=True)

	Stream.show(img1, "img1", shape=(600,300))
	Stream.show(img2, "img2", shape=(600,300), pause=True)

if __name__ == '__main__':
	test_homography()
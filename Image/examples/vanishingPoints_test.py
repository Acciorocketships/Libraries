import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *

K = np.array([[981.7,      0, 335.7],
			  [    0, 1043.7, 636.4],
			  [    0,      0,     1]])

def test_vanishingPoints():
    stream = Stream(mode='img',src='house.jpg')
    img = stream.get()
    pts = vanishingPoints(img, K)
    print(pts)

if __name__ == '__main__':
    test_vanishingPoints()



    
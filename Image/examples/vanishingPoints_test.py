import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *

def test_vanishingPoints():
    stream = Stream(mode='img',src='house.jpg')
    img = stream.get()
    pts = vanishingPoints(img, show=True)
    print(pts)

if __name__ == '__main__':
    test_vanishingPoints()



    
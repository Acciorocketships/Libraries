import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

import numpy as np
import time

from image import *
from imgstream import *

K = np.array([[981.7,      0, 335.7],
			  [    0, 1043.7, 636.4],
			  [    0,      0,     1]])
d = np.array([0.0977, 0.0012])


stream = Stream(mode='webcam') 
for img in stream: 
	undistorted_img = undistort(img, K, d)
	Stream.show(img,'Original/Undistorted',pause=False,shape=(720,1280))
	time.sleep(0.5)
	Stream.show(undistorted_img,'Original/Undistorted',pause=True,shape=(720,1280))
from imgstream import *
import cv2

stream = Stream(mode='webcam', src=0)

for i, img in enumerate(stream):
	cv2.imwrite("pics/" + str(i) + ".png", img)
	Stream.show(img, "img", pause=True, shape=(400,600))
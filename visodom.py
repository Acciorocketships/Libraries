import numpy as np
import cv2

class VisOdom:

	def __init__(self,K=None,distcoeff=None,scale=1):
		if K is None:
			self.K = np.identity(3,np.float64)
		else:
			self.K = K
		if distcoeff is None:
			self.distcoeff = np.array([[0,0,0,0]],np.float64) # [radial[0],radial[1],tangential[0],tangential[1]]
		else:
			self.distcoeff = distcoeff
		self.scale = 0.1
		self.detector = self.featureDetection()
		self.disparity = cv2.StereoSGBM_create(0, 128, 21) # mindisparity, maxdisparity (mult of 16), matched block size (odd num)
		self.Q = np.identity(4,np.float64)
		self.imglast = None
		self.imgcurr = None
		self.ptslast = None
		self.traj = None

		self.pos = np.array([[0,0,0]],np.float64).T
		self.rot = np.identity(3,np.float64)
		self.t = np.array([[0,0,0]],np.float64).T
		self.R = np.identity(3,np.float64)
		self.img3d = None
		self.disp = None

	def update(self,img,mapping=False):
		# If first image
		if self.imglast is None:
			self.imglast = self.rectify(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
			self.ptslast = np.array([element.pt for element in self.detector.detect(self.imglast)], dtype='float32')
		# If there is a previous image
		else:
			try:
				self.calcOdom(img)
			except Exception as err:
				print(err)
			if mapping:
				try:
					self.calcMap()
				except Exception as err:
					print(err)
			# Update other class variables
			self.imglast = self.imgcurr

	def calcOdom(self,img):
		# Find and match points
		self.imgcurr = self.rectify(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		self.ptscurr = np.array([element.pt for element in self.detector.detect(self.imgcurr)], dtype='float32')
		self.ptslast, self.ptscurr = self.featureTracking(self.imglast, self.imgcurr, self.ptscurr)
		# Camera np to get translation, rotation
		f = self.K[0,0]
		pp = (self.K[0,2],self.K[1,2])
		E, mask = cv2.findEssentialMat(self.ptscurr, self.ptslast, f, pp, cv2.RANSAC, 0.999, 1.0)
		_, self.R, self.t, mask = cv2.recoverPose(E, self.ptscurr, self.ptslast, focal=f, pp=pp)
		# Filter
			# t, R = smoothing(t,self.rot2euler(R))
			# R = self.euler2rot(R)
		# Update absolute position
		self.pos = self.pos + self.scale * self.rot.dot(self.t)
		self.rot = self.R.dot(self.rot)

	def calcMap(self):
		self.disp = self.disparity.compute(self.imglast,self.imgcurr)
		self.img3d = cv2.reprojectImageTo3D(self.disp,self.Q)
		# self.img3d[self.img3d==-np.inf] = np.inf
		self.img3d = np.abs(self.img3d)
		mindepth =  np.min(self.img3d)
		self.img3d[self.img3d==np.inf] = mindepth

	# Callable / Getter Methods

	def getRelativeT(self):
		return self.t

	def getRelativeR(self):
		return self.rot2euler(self.R)

	def getAbsT(self):
		return self.pos

	def getAbsR(self):
		return self.rot2euler(self.rot)

	# Used to show the path so far
	def odomvis(self,pos=None,scale=30):
		if self.traj is None:
			self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
		if pos is None:
			pos = self.pos
		cv2.circle(self.traj, (int(pos[0]*scale) + 300, int(pos[2]*scale) + 300) ,1, (255,0,0), 2)
		cv2.rectangle(self.traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED)
		text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(pos[0]), float(pos[1]), float(pos[2]))
		cv2.putText(self.traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
		cv2.imshow("Trajectory", self.traj)
		k = cv2.waitKey(1) & 0xFF
		return (k != ord('q'))

	def mapvis(self):
		if self.disp is not None:
			maxdepth = np.max(self.disp)
			mindepth = np.min(self.disp)
			depthmap = np.round((self.disp - mindepth) * 255 / (maxdepth - mindepth)).astype(np.uint8)
			cv2.imshow("Disparity", self.disp)
			k = cv2.waitKey(1) & 0xFF
			return (k != ord('q'))
		return True

	# Helper Functions

	# Rectifies the image and calculates self.Q for mapping
	# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
	# https://github.com/tobybreckon/python-examples-cv/blob/master/stereo_sgbm.py
	def rectify(self,img):
		_, RR, _, PR, self.Q, _, _ = cv2.stereoRectify(self.K, self.distcoeff, self.K, self.distcoeff, img.shape[::-1], self.R, self.t)
		# map1, map2 = cv2.initUndistortRectifyMap(self.K, self.distcoeff, RR, PR, img.shape[::-1], cv2.CV_32FC1)
		# img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
		return img

	def featureTracking(self,img_1, img_2, p1):
		lk_params = dict( winSize  = (21,21),
						  maxLevel = 3,
						  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
		p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
		st = st.reshape(st.shape[0])
		## find good one
		p1 = p1[st==1]
		p2 = p2[st==1]
		return p1,p2

	def featureDetection(self):
		thresh = dict(threshold=25, nonmaxSuppression=True);
		fast = cv2.FastFeatureDetector_create(**thresh)
		return fast

	def rot2euler(self,R):
		sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
		singular = (sy < 1E-6)
		if  not singular :
			x = np.arctan2(R[2,1] , R[2,2])
			y = np.arctan2(-R[2,0], sy)
			z = np.arctan2(R[1,0], R[0,0])
		else :
			x = np.arctan2(-R[1,2], R[1,1])
			y = np.arctan2(-R[2,0], sy)
			z = 0
		return np.array([x,y,z]) # [roll, pitch, yaw]

	def euler2rot(self,theta): # theta = [roll, pitch, yaw]
		R_x = np.array([[1,         0,                  0                   ],
						[0,         np.cos(theta[0]), -np.sin(theta[0]) ],
						[0,         np.sin(theta[0]), np.cos(theta[0])  ]]) 
		R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
						[0,                     1,      0                   ],
						[-np.sin(theta[1]),   0,      np.cos(theta[1])  ]]) 
		R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
						[np.sin(theta[2]),    np.cos(theta[2]),     0],
						[0,                     0,                      1]])	 
		R = np.dot(R_z, np.dot( R_y, R_x ))
		return R



if __name__ == '__main__':
	from imgstream import Stream
	# Video Stream
	stream = Stream(mode='cam',src='0')
	odom = VisOdom()
	# Main Loop
	for img in stream:
		# Compute
		odom.update(img,mapping=False)
		# Get Results
		pos = odom.getAbsT()
		rot = odom.getAbsR()
		# Visualization
		if not odom.odomvis(pos) or not odom.mapvis():
			cv2.destroyAllWindows()
			break

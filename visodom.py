import numpy as np
import cv2

class VisOdom:

	def __init__(self,K=None,scale=1):
		if K is None:
			self.K = np.identity(3)
		else:
			self.K = K
		self.scale = 0.1
		self.pos = np.array([[0,0,0]]).T
		self.rot = np.identity(3)
		self.detector = self.featureDetection()
		self.imglast = None
		self.ptslast = None
		self.traj = np.zeros((600, 600, 3), dtype=np.uint8)

	# Main logic. Gets translation and rotation from last photo
	def update(self,img,smoothing=None):
		t = np.array([[0,0,0]]).T
		R = np.identity(3)
		# If first image
		if self.imglast is None:
			self.imglast = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			self.ptslast = np.array([element.pt for element in self.detector.detect(img)], dtype='float32')
		else:
		# If there is a previous image
			try:
				# Find and match points
				imgcurr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				ptscurr = np.array([element.pt for element in self.detector.detect(img)], dtype='float32')
				pts1, pts2 = self.featureTracking(self.imglast, imgcurr, ptscurr)
				# Camera np to get translation, rotation
				f = self.K[0,0]
				pp = (self.K[0,2],self.K[1,2])
				E, mask = cv2.findEssentialMat(pts2, pts1, f, pp, cv2.RANSAC, 0.999, 1.0)
				_, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=f, pp=pp)
				# Update absolute position
				if smoothing is not None:
					t, R = smoothing(t,self.rot2euler(R))
					R = self.euler2rot(R)
				self.pos = self.pos + self.scale * self.rot.dot(t)
				self.rot = R.dot(self.rot)
				# Update other class variables
				self.imglast = imgcurr
				self.ptslast = ptscurr
			except Exception as err:
				print(err)
		return t, R

	# Callable / Getter Methods

	def getRelativeT(self,img):
		t, _ = self.update(img)

	def getRelativeR(self,img):
		_, R = self.update(img)
		return self.rot2euler(R)

	def getAbsT(self,img=None):
		if img is not None:
			self.update(img)
		return self.pos

	def getAbsR(self,img=None):
		if img is not None:
			self.update(img)
		return self.rot2euler(self.rot)

	# Used to show the path so far
	def visualization(self,pos=None,scale=30):
		if pos is None:
			pos = self.pos
		cv2.circle(self.traj, (int(pos[0]*scale) + 300, int(pos[2]*scale) + 300) ,1, (255,0,0), 2)
		cv2.rectangle(self.traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED)
		text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(pos[0]), float(pos[1]), float(pos[2]))
		cv2.putText(self.traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
		cv2.imshow("Trajectory", self.traj)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			return False
		return True

	# Helper Functions

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
	# Main Loop
	for img in stream:
		# Compute
		odom.update(img)
		# Get Results
		pos = odom.getAbsT()
		rot = odom.getAbsR()
		# Visualization
		if not odom.visualization(pos):
			cv2.destroyAllWindows()
			break

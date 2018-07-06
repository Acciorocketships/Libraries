from imgstream import Stream
from kalman import Kalman

# Video Stream
stream = Stream(mode='cam',src='0')
# Odometry
K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
			  [0, 7.188560000000e+02, 1.852157000000e+02],
			  [0,                  0,                  1]])
odom = VisOdom(K=K)
# H
H = np.array([[1,0,0,0,0,0],
			  [0,0,1,0,0,0],
			  [0,0,0,0,1,0]])
# F
def F(dt):
	return np.array([[1,dt,0,0,0,0],
					 [0,1,0,0,0,0],
					 [0,0,1,dt,0,0],
					 [0,0,0,1,0,0],
					 [0,0,0,0,1,dt],
					 [0,0,0,0,0,1]])
# Q
Q = np.zeros((6,6))
Q[0,1]=1; Q[1,0]=1; Q[2,3]=1; Q[3,2]=1; Q[4,5]=1; Q[5,4]=1
Q *= 0.5
Q[1,1]=1; Q[3,3]=1; Q[5,5]=1
Q *= 0.3
# Kalman Filter
kfpos = Kalman(F=F,dt=0,H=H,Q=Q,R=0.2,numsensors=3)
kfrot = Kalman(F=F,dt=0,H=H,Q=Q,R=0.2,numsensors=3)
# Main Loop
for img in stream:
	odom.update(img)
	pos = odom.getAbsT()
	rot = odom.getAbsR()
	try:
		pos = kfpos.predict(pos)
		rot = kfrot.predict(rot)
	except ValueError:
		pass
	if not odom.visualization(pos):
		cv2.destroyAllWindows()
		break
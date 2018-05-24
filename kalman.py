import numpy as np
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py
# https://www.cs.utexas.edu/~teammco/misc/kalman_filter/

class Kalman:

	def __init__(self,F='pv',H=None,dt=0,numsensors=1,Q=None,R=None,P=None,B=None,**kwargs):
		# Filter
		Fstates = F
		if type(F) == type(lambda x: 0):
			try:
				Fstates = F(0)
			except TypeError:
				Fstates = F()
		if type(Fstates) == list:
			self.states = len(F)
		elif type(Fstates) == np.ndarray:
			self.states = Fstates.shape[0]
		elif type(Fstates) == str:
			self.states = len(Fstates) * numsensors
		self.filter = KalmanFilter(dim_x=self.states,dim_z=numsensors)
		self.time = 0
		# P Covariance Initialization (changes over time), size:(states,states)
		if P is None:
			self.filter.P *= 1000
		elif type(P)==int or type(P)==float:
			self.filter.P *= P
		else:
			self.filter.P = np.array(P)
		# Time
		self.autotime = (dt == 0)
		self.dt = dt
		self.tic = time.time()
		# F Next State Prediction, size:(states,states)
		self.F = None
		if (type(F) == list) or (type(F) == np.ndarray):
			self.filter.F = np.array(F)
		else:
			self.F = F
			self.calcF()
		# H Measurement, size:(numsensors,states) (Give matrix, or tuple (0,3) to provide Hs for only states 0 and 3)
		# x_measured = H * x_predicted (0s in x_measured where no measurement is taken) 
		if (type(H) == list) or (type(H) == np.ndarray):
			self.filter.H = np.array(H)
		else:
			self.filter.H = np.identity(self.states)
			if type(H)==int or type(H)==tuple:
				self.filter.H = self.filter.H[H,:]
			if len(self.filter.H.shape)==1:
				self.filter.H = np.array([self.filter.H])
		# R Sensor Uncertainty Init size:(numsensors,numsensors)
		if (type(R) == list) or (type(R) == np.ndarray):
			self.filter.R = np.array(R)
		elif type(R)==int or type(R)==float:
			self.filter.R = R*np.identity(numsensors)
		else:
			self.filter.R = np.identity(numsensors)
		# Q Actual Volatility Init size:(states,states)
		self.Qmag = None
		if (type(Q) == list) or (type(Q) == np.ndarray):
			self.filter.Q = np.array(Q)
		else:
			if (type(Q)==int) or (type(Q)==float):
				self.Qmag = Q
			elif type(self.filter.R)==np.ndarray:
				self.Qmag = np.linalg.norm(self.filter.R)
			else:
				self.Qmag = self.filter.R
			self.calcQ()
		# B Control Init (Extra Forces)
		if (type(B) == list) or (type(B) == np.ndarray):
			self.filter.B = np.array(B)
		# Other Init
		for name, val in kwargs.items():
			if name in self.filter.__dict__:
				self.filter.__dict__[name] = val

	def calcDt(self):
		self.dt = time.time() - self.tic
		self.tic = time.time()

	def calcF(self):
		if self.F == 'pva':
			self.filter.F = np.array([[1, self.dt, 0.5*self.dt**2], [0, 1, self.dt],[0, 0, 1]])
		elif self.F == 'pv' or self.F == 'va':
			self.filter.F = np.array([[1, self.dt], [0, 1]])
		elif type(self.F) == str and len(self.F) == 1:
			self.filter.F = np.array([1])
		elif type(self.F) == type(lambda x: 0):
			try:
				self.filter.F = np.array(self.F(self.dt))
			except TypeError:
				self.filter.F = np.array(self.F())

	def calcQ(self):
		if self.Qmag is not None:
			self.filter.Q = Q_discrete_white_noise(self.states, self.dt, self.Qmag)

	# xex: x expected, xme: m measured, xf: x filtered, P: covariance, H: measurement, F: prediction matrix K: kalman constant
	# xex = (F * xf) + (B * u)
	# P = (F * P * F') + R
	# K = P * H' * inv(H*P*H' + Q)
	# xf = xex + K*(xme - H*xex)
	# P = (I - K*H) * P
	def predict(self,x=None):
		self.time += self.dt
		if self.autotime:
			self.calcDt()
			self.calcF()
			self.calcQ()
		if x is not None:
			x = np.array([x])
			x = x.reshape((x.size,1))
		if (self.time <= self.dt) and (x is not None):
			self.filter.x = np.matmul(np.transpose(self.filter.H),x)
		self.filter.predict()
		if x is not None:
			self.filter.update(x)
		return self.filter.x



from gaussnd import *
import numpy as np
from math import log

class Regressor:

	def __init__(self):
		self.restop = 100
		self.resstep = 0.1
		self.levels = 5
		self.map = {}
		self.dim = 3
		self.offset = 0

	# TODO: add fit(x,y) command that uses svm with rbf kernel to find gaussians
	# https://stackoverflow.com/questions/31300131/extracting-coefficients-with-a-nonlinear-kernel-using-sklearn-svm-python

	def min(self,x0=None,eqcons=[],ieqcons=[]):
		return self.equation(x0).min(x0=x0,eqcons=eqcons,ieqcons=ieqcons)


	def max(self,x0=None,eqcons=[],ieqcons=[]):
		return self.equation(x0).max(x0=x0,eqcons=eqcons,ieqcons=ieqcons)


	def set(self,pos,val,radius=1):
		pos = np.array(pos)
		self.dim = pos.size
		currval = self.eval(pos)
		cov = radius * np.eye(pos.size)
		gaussian = GaussND(numN=(pos,cov))
		newval = gaussian[pos]
		gaussian *= (val-currval) / newval
		self.add(gaussian)


	def add(self,pos,val,radius=1):
		pos = np.array(pos)
		self.dim = pos.size
		cov = radius * np.eye(pos.size)
		gaussian = GaussND(numN=(pos,cov))
		newval = gaussian[pos]
		gaussian *= val / newval
		self.addg(gaussian)
			

	def addg(self,gaussian):
		if gaussian.numN[0] is None:
			self.offset += gaussian.numC[0]
			return
		self.dim = gaussian.numN[0].mean.shape[0]
		pos = gaussian.numN[0].mean
		radius = gaussian.numN[0].cov[0][0]
		deepestlevel = min(round(log(1/radius * self.restop) / log(1/self.resstep))-1, self.levels-1)
		# Go to the largest level at which the gaussian spills into surrounding boxes
		curr = self.map
		for level in range(0,deepestlevel):
			key = self.tohash(self.discretize(pos, self.stepsize(level)))
			if key not in curr:
				curr[key] = ([],{})
			curr = curr[key][1]
		# Add the gaussian to the middle and surrounding boxes
		discpos = self.discretize(pos, self.stepsize(deepestlevel))
		for corner in self.corners(discpos, self.stepsize(deepestlevel)):
			key = self.tohash(corner)
			if key not in curr:
				curr[key] = ([],{})
			curr[key][0].append(gaussian)


	def fit(self, x, y, gamma='scale', C=1.0):
		# x numpy shape: (nsamples, ndim), y numpy shape: (nsamples,)
		model = SVR(gamma=gamma, C=C)
		model.fit(x,y)
		numC = np.reshape(model.dual_coef_, (-1,)).tolist()
		numN = [ (np.reshape(model.support_vectors_[i,:],(-1,)), 1/np.sqrt(2*model._gamma)*np.eye(model.support_vectors_.shape[1]))
				for i in range(model.support_vectors_.shape[0])]
		numC.append(np.asscalar(model.intercept_))
		numN.append(None)
		for i in range(len(numC)):
			gaussian = GaussND(numC=numC[i],numN=numN[i])
			self.addg(gaussian)


	def eval(self,pos):
		if not isinstance(pos,np.ndarray):
			pos = np.array(pos)
		if len(pos.shape) == 2:
			outputs = []
			for i in range(pos.shape[1]):
				outputs.append(self.eval(pos[:,i]))
			outputs = np.array(outputs)
			return outputs
		gaussians = []
		curr = self.map
		for level in range(0,self.levels):
			key = self.tohash(self.discretize(pos, self.stepsize(level)))
			if key not in curr:
				break
			gaussians = gaussians + curr[key][0]
			curr = curr[key][1]
		return sum(list(map(lambda f: f[pos], gaussians))) + self.offset


	def equation(self, pos):
		# TODO: get analytical solution for the entire area if pos is not given
		pos = np.array(pos)
		eqn = 0
		curr = self.map
		for level in range(0,self.levels):
			key = self.tohash(self.discretize(pos, self.stepsize(level)))
			if key not in curr:
				break
			for term in curr[key][0]:
				eqn += term
			curr = curr[key][1]
		eqn += self.offset
		return eqn


	# Finds the corners that circumscribe a position at a given level
	# Given a corner, this will return the centers of the adjacent cells
	def corners(self,middle,h):
		return self.cornershelper(middle, h, 0)
	def cornershelper(self,pos,h,i):
		if i == len(pos):
			return [pos]
		pos0 = pos.copy()
		pos1 = pos.copy()
		pos2 = pos.copy()
		pos1[i] += h
		pos2[i] -= h
		return self.cornershelper(pos0, h, i+1) + self.cornershelper(pos1, h, i+1) + self.cornershelper(pos2, h, i+1)


	def stepsize(self,level):
		return self.restop * (self.resstep ** level)


	def discretize(self,val,step):
		if isinstance(val,np.ndarray):
			val = np.round(val / step) * step
		else:
			val = round(val / step) * step
		return val


	def tohash(self,val):
		if isinstance(val,np.ndarray):
			return tuple(np.reshape(val,(-1,)))
		else:
			return val


	def __getitem__(self,pos):
		return self.eval(pos)


	def __add__(self,term):	
		pos = term[0]
		val = term[1]
		if len(term) > 2:
			self.add(pos,val,radius=term[2])
		else:
			self.add(pos,val)
		return self


	def __radd__(self,term):
		return self.__add__(term)


	def __mul__(self,term):	
		pos = term[0]
		val = term[1]
		if len(term) > 2:
			self.set(pos,val,radius=term[2])
		else:
			self.set(pos,val)
		return self


	def __rmul__(self,term):
		return self.__mul__(term)


	def plot(self,lim=[[-5,5],[-5,5],[-5,5]]):
		if self.dim > 3:
			print("Too many dimensions to plot")
		elif self.dim == 3:
			from mayavi import mlab
			# Evaluate
			xi,yi,zi = np.mgrid[lim[0][0]:lim[0][1]:50j, lim[1][0]:lim[1][1]:50j, lim[2][0]:lim[2][1]:50j]
			coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
			density = self.eval(coords).reshape(xi.shape)
			# Plot scatter with mayavi
			mlab.figure('DensityPlot',fgcolor=(0.0,0.0,0.0),bgcolor=(0.85,0.85,0.85),size=(600, 480))
			grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
			minval = 0
			maxval = density.max()
			mlab.pipeline.volume(grid, vmin=minval, vmax=minval + .5*(maxval-minval))
			mlab.axes(xlabel="x1",ylabel="x2",zlabel="x3")
			mlab.show()
		elif self.dim == 2:
			from matplotlib import cm
			import matplotlib.pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			fig = plt.figure()
			ax = fig.add_axes([0,0,1,1], projection='3d')
			# Evaluate
			xi,yi = np.mgrid[lim[0][0]:lim[0][1]:100j, lim[1][0]:lim[1][1]:100j]
			coords = np.vstack([item.ravel() for item in [xi, yi]])
			density = self.eval(coords).reshape(xi.shape)
			# Plot surface with matplotlib
			surf = ax.plot_surface(xi, yi, density, cmap=cm.coolwarm, linewidth=0, antialiased=False, rcount=100, ccount=100, alpha=0.7)
			ax.contour(xi, yi, density, linewidth=5)
			ax.view_init(90, -90)
			plt.xlabel("$x_1$")
			plt.ylabel("$x_2$")
			fig.colorbar(surf, orientation='horizontal', pad=0.01, fraction=0.12, shrink=0.9, aspect=18)
			plt.show()
		elif self.dim == 1:
			import matplotlib.pyplot as plt
			fig = plt.figure()
			# Evaluate
			xi = np.mgrid[lim[0][0]:lim[0][1]:100j]
			density = self.eval(np.array([xi])).reshape(xi.shape)
			# Plot surface with matplotlib
			plt.plot(xi, density)
			plt.xlabel("$x$")
			plt.show()


def test_fit():
	from sklearn.datasets.samples_generator import make_blobs
	x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
	r = Regressor()
	r.fit(x,y)
	r.plot()


def test_add():
	r = Regressor()
	r += ([2,2,-1],1,1)           # x, f(x), radius
	r += ([-3,0,1],-2,0.5)		 # x, f(x), radius (negative will not be displayed on plot)
	r += ([-2,1,2],1,0.5)		 # x, f(x), radius
	eq = r.equation([0,0,0])	 # analytical equation in the vicinity of [0,0,0]
	xopt = r.min([0,0,0])
	print("argmin(r) = ", xopt)			# Sure engouh, xopt = [-3,0,0], the location where we set the function to -2
	print("min(r) = ", eq[xopt])        # The analytical equation can be evaluated
	print("r([0,0,0]) = ", r[[0,0,0]])  # Or the regressor can be evaluated directly
	r.plot()	


if __name__ == '__main__':
	test_fit()

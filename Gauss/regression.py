from gauss import *
import numpy as np
from rtree import index as kdtree

class Regressor:

	def __init__(self):
		#https://rtree.readthedocs.io/en/latest/tutorial.html#insert-records-into-the-index
		self.init = False
		self.dim = None
		self.index = None


	def initialise(self, dim):
		if not self.init:
			self.dim = dim
			prop = kdtree.Property()
			prop.dimension = dim
			self.index = kdtree.Index(interleaved=False, properties=prop)
			self.init = True


	def set(self,pos,val,radius=1):
		self.initialise(len(pos))
		if isinstance(radius,np.ndarray):
			cov = radius
		else:
			cov = (radius**2) * np.eye(self.dim)
		gaussian = Gauss(numN=(pos,cov))
		currval = self.eval(pos)
		gaussval = gaussian[pos]
		C = (currval-val) / gaussval
		gaussian = C * gaussian
		self.addg(gaussian)


	def add(self,pos,val,radius=1):
		self.initialise(len(pos))
		if isinstance(radius,np.ndarray):
			cov = radius
		else:
			cov = (radius**2) * np.eye(self.dim)
		gaussian = Gauss(numN=(pos,cov), numC=val)
		gaussval = gaussian[pos]
		C = val / gaussval
		gaussian = C * gaussian
		self.addg(gaussian)
			

	def addg(self,gaussian):
		self.initialise(gaussian.dim)
		bounds = self.get_bounding_box(gaussian, astuple=True)
		uid = hash(gaussian)
		self.index.insert(uid, bounds, obj=gaussian)


	def eval(self, pos=None):
		# x is either shape (dim,) or (nsamples,dim)
		if pos is None:
			eqn = self.equation()
			return eqn.eval()
		if isinstance(pos,np.ndarray) and len(pos.shape)>1 and pos.shape[0]>1:
			return np.array([self.eval(pos[i,:]) for i in range(pos.shape[0])])
		box = [pos[i//2] for i in range(2*len(pos))]
		nearby_gaussians = [g.object for g in self.index.intersection(box, objects=True)]
		return sum(map(lambda g: g[pos], nearby_gaussians))

	def grad(self, pos=None):
		# x is either shape (dim,) or (nsamples,dim)
		if pos is None:
			eqn = self.equation()
			return eqn.eval()
		if isinstance(pos,np.ndarray) and len(pos.shape)>1 and pos.shape[0]>1:
			return np.array([self.eval(pos[i,:]) for i in range(pos.shape[0])])
		box = [pos[i//2] for i in range(2*len(pos))]
		nearby_gaussians = [g.object for g in self.index.intersection(box, objects=True)]
		return sum(map(lambda g: g.grad(pos), nearby_gaussians))


	def equation(self, pos=None):
		# get the analytical equation (as a gaussian) in the vicinity of pos. if pos is None, then it returns the global equation
		if pos is None:
			box = [-np.inf if i%2==0 else np.inf for i in range(2*self.dim)]
		else:
			box = [pos[i//2] for i in range(2*len(pos))]
		nearby_gaussians = [g.object for g in self.index.intersection(box, objects=True)]
		return (sum(nearby_gaussians) if len(nearby_gaussians)!=0 else lambda x: 0)


	def min(self,x0=None,eqcons=[],ieqcons=[]):
		return self.equation(x0).min(x0=x0,eqcons=eqcons,ieqcons=ieqcons)


	def max(self,x0=None,eqcons=[],ieqcons=[]):
		return self.equation(x0).max(x0=x0,eqcons=eqcons,ieqcons=ieqcons)


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
			gaussian = Gauss(numC=numC[i],numN=numN[i])
			self.addg(gaussian)


	def __getitem__(self,pos):
		return self.eval(pos)


	def __add__(self,term):
		if isinstance(term,tuple):
			pos = term[0]
			val = term[1]
			if len(term) == 3:
				self.add(pos,val,radius=term[2])
			else:
				self.add(pos,val)
		else:
			self.addg(term)
		return self


	def __radd__(self,term):
		return self.__add__(term)


	def get_bounding_box(self, gaussian, astuple=False):
		nsigma = 4
		cov = gaussian.cov
		mu = gaussian.mu
		eigvals, eigvecs = np.linalg.eig(cov)
		axes = eigvecs @ np.diag(np.sqrt(eigvals) * nsigma)
		extents = np.amax(np.abs(axes), axis=1)
		bounds = np.concatenate((np.reshape(mu-extents,(self.dim,1)),np.reshape(mu+extents,(self.dim,1))), axis=1)
		if not astuple:
			return bounds
		else:
			return tuple(np.reshape(bounds,(-1,)))


	def plot(self,lim=[[-5,5],[-5,5],[-5,5]]):
		if self.dim > 3:
			print("Too many dimensions to plot")
		elif self.dim == 3:
			from mayavi import mlab
			# Evaluate
			xi,yi,zi = np.mgrid[lim[0][0]:lim[0][1]:50j, lim[1][0]:lim[1][1]:50j, lim[2][0]:lim[2][1]:50j]
			coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
			density = self.eval(coords.T).reshape(xi.shape)
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
			density = self.eval(coords.T).reshape(xi.shape)
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
from scipy.stats import multivariate_normal, _multivariate
from scipy.optimize import fmin, fmin_slsqp
from time import time
import numpy as np


class GaussND:

	def __init__(self,numN=[None],numC=[1],denN=[None],denC=[1]):
		# N is a list of tuples (mu, cov)
		self.numC = numC if isinstance(numC,list) else [numC]
		self.numN = numN if isinstance(numN,list) else [numN]
		self.denC = denC if isinstance(denC,list) else [denC]
		self.denN = denN if isinstance(denN,list) else [denN]
		for i, term in enumerate(self.numN):
			if term is not None and not isinstance(term,_multivariate.multivariate_normal_frozen):
				self.numN[i] = multivariate_normal(mean=np.reshape(term[0],(-1,)),cov=term[1])
		for i, term in enumerate(self.denN):
			if term is not None and not isinstance(term,_multivariate.multivariate_normal_frozen):
				self.denN[i] = multivariate_normal(mean=np.reshape(term[0],(-1,)),cov=term[1])


	def min(self,x0=None,eqcons=[],ieqcons=[]):
		# eqcons (equality constrains) is a list of functions such that f(x)=0
		# ieqcons (inequality constraints) is a list of fucntions such that f(x)>=0
		# x0 is the starting point, default [0,0,0]
		if x0 is None:
			dim = 0
			for term in self.numN + self.denN:
				if term is not None:
					dim = term.mean.shape[0]
					break
			x0 = np.zeros((dim,))
		if len(eqcons) != 0 or len(ieqcons) != 0:
			m = fmin_slsqp(self.evaluate,x0,eqcons=eqcons,ieqcons=ieqcons,iprint=0)
		else:
			m = fmin(self.evaluate,x0,iprint=0)
		return m


	def __mul__(self,other):
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		# (a / b) * (c / d) = ac / bd
		numN, numC = self.multiplyPoly(self.numN,other.numN,self.numC,other.numC)
		denN, denC = self.multiplyPoly(self.denN,other.denN,self.denC,other.denC)
		return GaussND(numC=numC,numN=numN,denC=denC,denN=denN)


	def __add__(self,other):
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		# a/b + c/d = (ad + bc) / (bd), where a/b is self and c/d is other
		numN0, numC0 = self.multiplyPoly(self.numN,other.denN,self.numC,other.denC)
		numN1, numC1 = self.multiplyPoly(self.denN,other.numN,self.denC,other.numC)
		denN, denC = self.multiplyPoly(self.denN,other.denN,self.denC,other.denC)
		return GaussND(numC=numC0+numC1,numN=numN0+numN1,denC=denC,denN=denN)


	def multiplyPoly(self,N0,N1,C0=None,C1=None):
		# Inputs: two lists of gaussians (N0 and N1), and their coefficients (C0 and C1)
		# Output: tuple of list of gaussian terms and their coefficients (N, C)
		N = [None for i in range(len(N0)*len(N1))]
		C = [None for i in range(len(N0)*len(N1))]
		if C0 is None:
			C0 = [1 for i in range(len(N0))]
		if C1 is None:
			C1 = [1 for i in range(len(N1))]
		for i0 in range(len(N0)):
			for i1 in range(len(N1)):
				i = i0*len(N1) + i1
				N[i] = self.multiply(N0[i0],N1[i1])
				C[i] = C0[i0] * C1[i1]
		return (N, C)


	def multiply(self,N0,N1):
		if N0 is None and N1 is None:
			N = None
		elif N0 is None:
			N = N1
		elif N1 is None:
			N = N0
		else: # Multiply 2 Gausians
		# https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
			inv = np.linalg.inv(N0.cov + N1.cov)
			cov = N0.cov @ inv @ N1.cov
			mu = (N1.cov @ inv @ N0.mean) + (N0.cov @ inv @ N1.mean)
			N = multivariate_normal(mean=mu,cov=cov)
		return N


	def evaluate(self,x):
		x = np.array(x)
		num = sum(map(lambda term: term[0]*term[1].pdf(x) if term[1] is not None else term[0], zip(self.numC,self.numN)))
		den = sum(map(lambda term: term[0]*term[1].pdf(x) if term[1] is not None else term[0], zip(self.denC,self.denN)))
		return num / den


	def equal(self,N0,N1):
		if N0 is None and N1 is None:
			return True
		elif N0 is None or N1 is None:
			return False
		else: 
			return (np.all(N0.mean==N1.mean) and np.all(N0.cov==N1.cov))


	def plot(self,lim=None):
		from mayavi import mlab
		if lim is None:
			# Calculate limits
			mus = np.concatenate(tuple([[N.mean] for N in filter(lambda N: N is not None, self.numN+self.denN)]), axis=0)
			maxs = np.amax(mus,axis=0) + 1
			mins = np.amin(mus,axis=0) - 1
			s = np.amax(maxs-mins) / 2
			mid = (maxs + mins) / 2
			lim = [[mid[i]-1.2*s, mid[i]+1.2*s] for i in range(maxs.shape[0])]
		if len(lim) == 3:
			# Evaluate
			xi,yi,zi = np.mgrid[lim[0][0]:lim[0][1]:50j, lim[1][0]:lim[1][1]:50j, lim[2][0]:lim[2][1]:50j]
			coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
			density = self.evaluate(coords.T).reshape(xi.shape)
			# Plot scatter with mayavi
			figure = mlab.figure('DensityPlot')
			grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
			minval = 0
			maxval = density.max()
			mlab.pipeline.volume(grid, vmin=minval, vmax=minval + .5*(maxval-minval))
			mlab.axes()
			mlab.show()
		pass


	def __getitem__(self,x):
		return self.evaluate(x)

	def __eq__(self,other):
		if not isinstance(other,GaussND):
			return False
		if len(other.numC) != len(self.numC) or len(other.denC) != len(self.denC):
			return False
		for i in range(len(self.numC)):
			if self.numC[i] != other.numC[i] or not self.equal(self.numN[i],other.numN[i]):
				return False
		for i in range(len(self.denC)):
			if self.denC[i] != other.denC[i] or not self.equal(self.denN[i],other.denN[i]):
				return False
		return True

	def __ne__(self,other):
		return not self.__eq__(other)

	def __neg__(self):
		return GaussND(numC=list(map(lambda x: -x, self.numC)), numN=list(self.numN), denC=list(self.denC), denN=list(self.denN))

	def __sub__(self,other):
		# self - other
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		return self.__add__(other.__neg__())

	def __rsub__(self,other):
		# other - self
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		return other.__add__(self.copy().__neg__())

	def __radd__(self,other):
		return self.__add__(other)

	def __rmul__(self,other):
		return self.__mul__(other)

	def __truediv__(self,other):
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		numC = list(other.numC)
		numN = list(other.numN)
		denC = list(other.denC)
		denN = list(other.denN)
		invother = GaussND(numC=denC,numN=denN,denC=numC,denN=numN)
		return self.__mul__(invother)

	def __rtruediv__(self,other):
		if not isinstance(other,GaussND):
			other = GaussND(numC=other)
		numC = list(self.numC)
		numN = list(self.numN)
		denC = list(self.denC)
		denN = list(self.denN)
		invself = GaussND(numC=denC,numN=denN,denC=numC,denN=numN)
		return invself.__mul__(other)

		

		


if __name__ == '__main__':
	# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
	# https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html

	mu0 = np.array([[0,0,0]]).T
	cov0 = 0.2*np.identity(3)
	g0 = GaussND(numN=(mu0,cov0))

	mu1 = np.array([[1,0,0]]).T
	cov1 = 0.1*np.array([[1,0,0],[0,1,0],[0,0,1]])
	g1 = GaussND(numN=(mu1,cov1))

	mu2 = np.array([[-0.5,0,0]]).T
	cov2 = 0.1*np.identity(3)
	g2 = 10*GaussND(numN=(mu2,cov2))

	g3 = (g2 + g1) / g0
	x = np.array([0,0,0])
	# g3.plot()

	import code; code.interact(local=locals())


	
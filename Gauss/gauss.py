from mpmath import *
from sympy import *
from sympy.stats import *

# also make a 3D version

class Gauss:

	x = Symbol('x')

	c1 = Symbol('c_1')
	mu1 = Symbol('mu_1')
	sig1 = Symbol('sigma_1')
	c2 = Symbol('c_2')
	mu2 = Symbol('mu_2')
	sig2 = Symbol('sigma_2')

	minGaussRatio = lambdify( [mu1,mu2,sig1,sig2], (mu2*sig1**2 - mu1*sig2**2) / (sig1**2 - sig2**2) ) # catch sig1==sig2. perhaps with limit?

	gauss = lambdify( [c1,mu1,sig1,x], c1*density(Normal("N",mu1,sig1))(x) )

	def __init__(self,c=[1],mu=[None],sigma=[None],invc=[1],invmu=[None],invsigma=[None]):
		# ( ∑ c * N(mu,sigma) ) / ( ∑ invc * N(invmu,invsigma) )
		# for just a constant, do Gauss(c=[const],mu=[None],sigma=[None])
		self.numC = c if isinstance(c,list) else [c]
		self.numMu = mu if isinstance(mu,list) else [mu]
		self.numSig = sigma if isinstance(sigma,list) else [sigma]
		self.denC = invc if isinstance(invc,list) else [invc]
		self.denMu = invmu if isinstance(invmu,list) else [invmu]
		self.denSig = invsigma if isinstance(invsigma,list) else [invsigma]


	def min(self,approx=False,check=False):
		
		# the approximation equals the min if there is one term in the numerator and denominator
		guess = 0
		totalweighting = 0
		maxDen = (0, 0, 0)
		for i in range(len(self.denSig)):
			# choose term in the denominator with the highest sigma to represent the entire denominator
			# if there is a constant value, use that
			if (self.denSig[i] == None and self.denC[i] > maxDen[0]) or \
			   (maxDen[2] != None and self.denSig[i] > maxDen[2]):
				maxDen = (self.denC[i], self.denMu[i], self.denSig[i])
		for i in range(len(self.numSig)):
			if maxDen[2] != None and self.numSig[i] != None: # if Gauss / Gauss
				weighting = self.numC[i] * (self.numSig[i] / maxDen[2]) ** 2
			elif maxDen[2] != None: # if Constant / Gauss
				weighting = self.numC[i] * 30
			else: # if Gauss / Constant
				weighting = self.numC[i] * exp(-5*self.numSig[i]) # 1 when sigma=0, 0 when sigma=inf
			if self.numSig[i] != maxDen[2] and maxDen[2] != None:
				if self.numSig[i] == None: # if Constant / Gauss
					localmin = maxDen[1]
				else: # if Gauss / Gauss
					localmin = Gauss.minGaussRatio(self.numMu[i],maxDen[1],self.numSig[i],maxDen[2])
			elif maxDen[2] == None: # if Gauss / Constant
				localmin = self.numMu[i]
			else: # if Constant (from cancellation)
				localmin = guess / totalweighting # no effect
			guess += weighting * localmin
			totalweighting += weighting
		guess = guess / totalweighting
		guess = float(guess)

		if approx:
			return guess
		return self.localmin(guess,check)


	def localmin(self,x,check=False):
		# The critical point closest to x
		# This doesn't guarantee a minimum
		# import time; starttime = time.time()
		eqn = lambdify([Gauss.x], self.evaluate().diff(Gauss.x), "mpmath")
		if check:
			# returns None if no minimum is found
			# Specify range to verify derivative going from negative to positive
			offset = 1
			factor = 2
			maxiter = 5
			if eqn(x) < 0:
				iterations = 0
				while eqn(x + offset) < 0:
					if iterations > maxiter:
						return None
					offset *= factor
					iterations += 1
				root = findroot(eqn,(x,x+offset),solver="pegasus")
			else:
				iterations = 0
				while eqn(x - offset) > 0:
					if iterations > maxiter:
						return None
					offset *= factor
					iterations += 1
				root = findroot(eqn,(x-offset,x),solver="pegasus")
			# Sanity check that it worked
			if eqn(root) > eqn(root+0.0001):
				return None
		else:
			root = findroot(eqn,x,solver="anewton")
		# print (time.time() - starttime)
		return float(root)


	def evaluate(self,x=None):
		# evaluates at a given x, or symbolically if given Symbol('x'). f[x] calls evaluate.
		if x is None:
			x = Gauss.x
		if len(self.numC) > 0:
			if isinstance(x,Expr):
				num = sum(map(lambda N: N[0]*density(Normal("N",N[1],N[2]))(x) if (N[1] is not None) else N[0], zip(self.numC,self.numMu,self.numSig)))
			else:
				num = sum(map(lambda N: Gauss.gauss(N[0],N[1],N[2],x) if (N[1] is not None) else N[0], zip(self.numC,self.numMu,self.numSig)))
		else:
			num = 1
		if len(self.denC) > 0:
			if isinstance(x,Expr):
				den = sum(map(lambda N: N[0]*density(Normal("N",N[1],N[2]))(x) if (N[1] is not None) else N[0], zip(self.denC,self.denMu,self.denSig)))
			else:
				den = sum(map(lambda N: Gauss.gauss(N[0],N[1],N[2],x) if (N[1] is not None) else N[0], zip(self.denC,self.denMu,self.denSig)))
		else:
			den = 1
		return num / den


	def __add__(self,other):

		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		if self.denC == other.denC and self.denMu==other.denMu and self.denSig==other.denSig:
			# If the denominators match, just add the numerators
			numC = self.numC + other.numC
			numMu = self.numMu + other.numMu
			numSig = self.numSig + other.numSig
			denC = list(self.denC)
			denMu = list(self.denMu)
			denSig = list(self.denSig)
		else:
			# a/b + c/d = (ad + bc) / (bd), where a/b is self and c/d is other
			ad = self.multiplyPoly((self.numC,self.numMu,self.numSig),(other.denC,other.denMu,other.denSig))
			bc = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.numC,other.numMu,other.numSig))
			bd = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.denC,other.denMu,other.denSig))
			numC = ad[0] + bc[0]
			numMu = ad[1] + bc[1]
			numSig = ad[2] + bc[2]
			denC = bd[0]
			denMu = bd[1]
			denSig = bd[2]

		return Gauss(c=numC,mu=numMu,sigma=numSig,invc=denC,invmu=denMu,invsigma=denSig)


	def __radd__(self,other):
		return self.__add__(other)


	def __mul__(self,other):
		# us multiplyPoly on num and den a/b * c/d = (ac) / (bd)
		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		ac = self.multiplyPoly((self.numC,self.numMu,self.numSig),(other.numC,other.numMu,other.numSig))
		bd = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.denC,other.denMu,other.denSig))

		numC = ac[0]
		numMu = ac[1]
		numSig = ac[2]
		denC = bd[0]
		denMu = bd[1]
		denSig = bd[2]

		return Gauss(c=numC,mu=numMu,sigma=numSig,invc=denC,invmu=denMu,invsigma=denSig)

	def __rmul__(self,other):
		return self.__mul__(other)


	def multiplyPoly(self,poly0,poly1):
		# input is of the form ([c0, c1, ...],[mu0, m1, ...],[sigma0, sigma1, ...])
		c = [None for i in range(len(poly0[0])*len(poly1[0]))]
		mu = [None for i in range(len(poly0[0])*len(poly1[0]))]
		sigma = [None for i in range(len(poly0[0])*len(poly1[0]))]

		for i0 in range(len(poly0[0])):
			for i1 in range(len(poly1[0])):
				ci, mui, sigi = self.multiply((poly0[0][i0],poly0[1][i0],poly0[2][i0]),
									          (poly1[0][i1],poly1[1][i1],poly1[2][i1]))
				i = i0*len(poly1[0]) + i1
				c[i] = ci
				mu[i] = mui
				sigma[i] = sigi

		return (c,mu,sigma)


	def multiply(self,gauss0,gauss1):
		# inputs are of the form (c,mu,sigma)
		c0, mu0, sig0 = gauss0
		c1, mu1, sig1 = gauss1

		c = c0 * c1
		if (mu0 is None) and (mu1 is not None):
			mu = mu1
			sig = sig1
		elif (mu1 is None) and (mu0 is not None):
			mu = mu0
			sig = sig0
		elif (mu1 is None) and (mu0 is None):
			mu = None
			sig = None
		else:
			mu = (mu0 * sig1**2 + mu1 * sig0**2) / (sig0**2 + sig1**2)
			sig = sqrt( (sig0**2 * sig1**2) / (sig0**2 + sig1**2) )

		return (c,mu,sig)


	def __rtruediv__(self,other):
		# other / self

		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		selfNumC = self.numC
		selfNumMu = self.numMu
		selfNumSig = self.numSig
		selfDenC = self.denC
		selfDenMu = self.denMu
		selfDenSig = self.denSig
		selfcopy = Gauss(c=list(selfDenC),mu=list(selfDenMu),sigma=list(selfDenSig),
						 invc=list(selfNumC), invmu=list(selfNumMu), invsigma=list(selfNumSig))
		return selfcopy.__mul__(other)


	def __truediv__(self,other):
		# self / other

		if not isinstance(other,Gauss):
			other = Gauss(c=other)
		else:
			other = Gauss(c=list(other.numC),mu=list(other.numMu),sigma=list(other.numSig),
						  invc=list(other.denC), invmu=list(other.denMu), invsigma=list(other.denSig))
		otherNumC = other.numC
		otherNumMu = other.numMu
		otherNumSig = other.numSig
		otherDenC = other.denC
		otherDenMu = other.denMu
		otherDenSig = other.denSig
		other.numC = otherDenC
		other.numMu = otherDenMu
		other.numSig = otherDenSig
		other.denC = otherNumC
		other.denMu = otherNumMu
		other.denSig = otherNumSig
		return self.__mul__(other)


	def __neg__(self):
		return Gauss(c=list(map(lambda x: -x, self.numC)),
					 mu=list(self.numMu),
					 sigma=list(self.numSig),
					 invc=list(self.denC),
					 invmu=list(self.denMu),
					 invsigma=list(self.denSig))


	def __sub__(self,other):
		# self - other
		if not isinstance(other,Gauss):
			other = Gauss(c=other)
		return self.__add__(other.__neg__())

	def __rsub__(self,other):
		# other - self
		if not isinstance(other,Gauss):
			other = Gauss(c=other)
		return other.__add__(self.copy().__neg__())

	def copy(self):
		return Gauss(c=list(self.numC),mu=list(self.numMu),sigma=list(self.numSig),
					 invc=list(self.denC), invmu=list(self.denMu), invsigma=list(self.denSig))


	def __eq__(self,other):
		return (self.numC==other.numC and self.numMu==other.numMu and self.numSig==other.numSig) and \
			   (self.denC==other.denC and self.denMu==other.denMu and self.denSig==other.denSig)


	def __ne__(self,other):
		return not self.__eq__(other)


	def __getitem__(self,x):
		return self.evaluate(x)


	def plot(self,ylim=None):
		plot(self.evaluate(Gauss.x), ylim=ylim)



if __name__ == '__main__':
	# g1 = Gauss(mu=[0],sigma=[1])
	# g2 = Gauss(mu=[1],sigma=[2])
	# gratio = g2 / g1
	init_printing()

	gadd = Gauss(mu=0,sigma=1) + Gauss(mu=3,sigma=1) + Gauss(mu=6,sigma=2)
	x = Gauss.x
	# print("Equation: ", pretty(gadd.evaluate(), use_unicode=True))
	print("Min: ", gadd.min())
	print("Min Approx", gadd.min(approx=True))
	print("Min Checking", gadd.min(check=True))

	import code; code.interact(local=locals())

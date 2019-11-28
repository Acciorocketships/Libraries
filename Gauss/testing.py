from regression import Regressor
from gauss import Gauss
import numpy as np

def test_regressor_approx_and_min():
	r = Regressor()
	r += ([0,1], 1, 0.5)
	r += ([1,0],-0.5, 0.75)
	r += ([-2,2], -1, 1)
	r += ([-0.8, -0.2], 2, 0.5)
	r += ([-1,-1],-3, 0.25)
	g = r.equation()
	x0 = np.array([0.0,0.0])
	hstep = np.array([2.0,0.0])
	vstep = np.array([0.0,2.0])
	# Test approximation
	print("r(%s)=%g, g(%s)=%g" % (x0, r[x0], x0, g[x0]))
	print("r(%s)=%g, g(%s)=%g" % (x0+hstep, r[x0+hstep], x0+hstep, g[x0+hstep]))
	print("r(%s)=%g, g(%s)=%g" % (x0-hstep, r[x0-hstep], x0-hstep, g[x0-hstep]))
	print("r(%s)=%g, g(%s)=%g" % (x0+vstep, r[x0+vstep], x0+vstep, g[x0+vstep]))
	print("r(%s)=%g, g(%s)=%g" % (x0-vstep, r[x0-vstep], x0-vstep, g[x0-vstep]))
	# Test local min
	min1 = r.min(x0)
	print("x0=%s, argmin=%s, r(%s)=%g" % (x0, min1, min1, r[min1]))
	min2 = r.min(x0+hstep)
	print("x0=%s, argmin=%s, r(%s)=%g" % (x0+hstep, min2, min2, r[min2]))
	min3 = r.min(x0-hstep)
	print("x0=%s, argmin=%s, r(%s)=%g" % (x0-hstep, min3, min3, r[min3]))
	min4 = r.min(x0+vstep)
	print("x0=%s, argmin=%s, r(%s)=%g" % (x0+vstep, min4, min4, r[min4]))
	min5 = r.min(x0-vstep)
	print("x0=%s, argmin=%s, r(%s)=%g" % (x0-vstep, min5, min5, r[min5]))

def test_regressor_grad():
	r = Regressor()
	r += ([0,1], 1, 0.5)
	r += ([1,0],-0.5, 0.75)
	r += ([-2,2], -1, 1)
	r += ([-0.8, -0.2], 2, 0.5)
	r += ([-1,-1],-3, 0.25)
	x = np.array([0.0,0.0])
	print(r.grad(x))
	print(r.grad(x.reshape((1,2))))
	r.plot()

def test_gradient():
	cov = np.array([[2,1],[1,1]])
	mu = np.array([0.1,0.2])
	g = 2 * Gauss(numN=[(mu,cov)])
	x = np.array([0.2,0.4])
	dx = 0.000001
	dgdx = np.array([ (g[x+np.array([dx,0])] - g[x]) / dx, (g[x+np.array([0,dx])] - g[x]) / dx ])
	dgdx2 = -g[x] * np.linalg.inv(g.numN[0].cov) @ (x.reshape((-1,)) - g.numN[0].mean)
	print(np.linalg.inv(g.numN[0].cov))
	print("%s - %s" % (x.reshape((-1,)), g.numN[0].mean))
	print("computed dgdx: %s" % dgdx)
	print("calculated dgdx: %s" % dgdx2)
	import code; code.interact(local=locals())


def test_gradient_implementation():
	cov = np.array([[2,1],[1,1]])
	mu = np.array([0.1,0.2])
	g = 2 * Gauss(numN=[(mu,cov)])
	x = np.array([0.2,0.4])
	print(g.grad(x))
	g.single_gaussian = False
	print(g.grad(x))
	print(g.grad())


def test_regressor_fit():
	from sklearn.datasets.samples_generator import make_blobs
	x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
	r = Regressor()
	r.fit(x,y)
	r.plot()


def test_regressor_min():
	r = Regressor()
	r += ([2,2,-1],1,1)           # x, f(x), radius
	r += ([-3,0,1],-2,2)		 # x, f(x), radius (negative will not be displayed on plot)
	r += ([0.1,-0.2,0],1,0.5)
	eq = r.equation([0,0,0])	 # analytical equation in the vicinity of [0,0,0]
	xopt = r.min([0,0,0])
	print("argmin(r) = ", xopt)			# Sure engouh, xopt = [-3,0,0], the location where we set the function to -2
	print("min(r) = ", eq[xopt])        # The analytical equation can be evaluated
	print("r([0,0,0]) = ", r[[0,0,0]])  # Or the regressor can be evaluated directly
	f, x = r.eval()
	import code; code.interact(local=locals())


def test_gauss():
	mu0 = [0,0]
	cov0 = (0.8**2)*np.identity(2)
	g0 = 50*Gauss(numN=(mu0,cov0))

	mu1 = [1,0]
	cov1 = (0.1**2)*np.array([[2,1],[1,1]])
	g1 = Gauss(numN=(mu1,cov1))

	mu2 = [0,1.5]
	cov2 = (0.3**2)*np.identity(2)
	g2 = 8*Gauss(numN=(mu2,cov2))

	g3 = g2 + g1 + g0
	import code; code.interact(local=locals())


if __name__ == '__main__':
	test_regressor_grad()
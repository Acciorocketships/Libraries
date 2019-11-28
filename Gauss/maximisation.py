from gauss import *
from regression import *
from matplotlib import pyplot as plt

class Maximiser:

	def __init__(self, function, dim):
		# function must be an instance of a class that has implemented .eval(x) and .grad(x) methods
		self.f = function
		self.dim = dim

	def max_k_grads(self, x0=None, k=30, sigma=2, plotter=None):
		if x0 is None:
			x0 = np.zeros((self.dim,))
		step_dist=0.1
		slow_step_C=0.1
		slow_step_thres = np.zeros((k,))
		stop_thres = 0.1
		x = sigma * np.random.randn(k,self.dim) + x0
		max_x = x0
		max_y = -np.inf
		for i in range(1000):
			# Break loop if there are no more active searches
			if x.shape[0]==0:
				break
			# Step toward gradient
			dx = self.f.grad(x)
			mag = np.linalg.norm(dx, axis=1)
			magnitudes = np.copy(mag)
			slow_step_thres = np.maximum(slow_step_thres, magnitudes/4)
			mag[mag<=slow_step_thres] = step_dist / slow_step_C
			dx_step = dx / mag[:,np.newaxis]
			x += step_dist*dx_step
			y = self.f[x]
			# Plot
			if plotter is not None:
				plotter.update(x,y)
			# Update maximum with searches that have finished
			done = np.where(magnitudes<=stop_thres)[0]
			if done.size!=0:
				max_x_local = x[done,:]
				max_y_local = y[done]
				idx = np.argmax(max_y_local)
				max_x_local = max_x_local[idx,:]
				max_y_local = max_y_local[idx]
				if max_y_local > max_y:
					max_y = max_y_local
					max_x = max_x_local
				x = np.delete(x,done,axis=0)
				slow_step_thres = np.delete(slow_step_thres,done,axis=0)
		return max_x


class Plotter:

	def __init__(self, ax):
		self.ax = ax

	def update(self, x, y):
		self.ax.scatter(x[:,0],x[:,1],y,color='k',marker='.')
		plt.draw()
		plt.pause(0.0001)


def test():
	r = Regressor()
	r += ([0,1], 1, 0.5)
	r += ([1,0],-0.5, 0.75)
	r += ([-2,2], -1, 1)
	r += ([-0.8, -0.2], 2, 0.5)
	r += ([-1,-1], 3, 0.5)
	plotter = Plotter(r.plot(pause=False))
	maximiser = Maximiser(r, 2)
	max_x = maximiser.max_k_grads(x0=np.zeros((2,)), k=10, sigma=4, plotter=plotter)
	max_y = r[max_x]
	print("min: r(%s) = %g" % (max_x, max_y))
	# x = np.random.randn(5,2)
	# for i in range(10000):
	# 	dx = r.grad(x)
	# 	mag = np.linalg.norm(dx, axis=1)[:,np.newaxis] + 1E-10
	# 	dx_unit = dx / mag
	# 	x += 0.05*dx_unit
	# 	y = r[x]
	# 	plot.update(x,y)



if __name__ == '__main__':
	test()
import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from kalman import *
import numpy as np
from matplotlib import pyplot as plt
from filterpy.common import Q_discrete_white_noise

dt = 0.01
x = np.arange(0,30,dt)
y = np.sin(0.5*x)
vnoisey = 0.5*np.cos(0.5*x) + 0.05*np.cos(25*x)+0.1*np.sin(12*x)
ynoisey = y + 0.05*np.sin(20*x)+0.1*np.cos(8*x)

kf = Kalman(F='pv',dt=dt,H=(0,1),Q=1e-5,R=0.1)
output = []
for i in range(x.size):
	output.append(kf.predict([ynoisey[i],vnoisey[i]])[0])

plt.plot(x,ynoisey,x,vnoisey,x,output)
plt.show()
import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from network import Network
import numpy as np
import time

# Test data setup
class Test:
	a = np.array([1,2,3])
	def __init__(self):
		self.b = np.identity(3)

var = Test()

def gotit(var):
	print('In Callback:')
	print(var)

receiver = Network(host='localhost',portin=8080)
sender = Network(host='localhost',portout=8080)


receiver.run(callbacks={'a':gotit})
sender.send(var,"a")

time.sleep(1)

print('In Get:')
a = receiver.get("a")
print(a)

receiver.close()
sender.close()
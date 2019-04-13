from network import *
import threading
from datetime import datetime

class Server:

	def __init__(self):
		self.node = Network(host=None, port=8080, otherport=8080)
		self.node.connectCallback = self.newConnection
		self.connections = set()
		self.running = False
		self.adding = None
		self.run()

	def newConnection(self, address):
		self.adding = address
		self.node.send(self.connections, name='connections', host=address) # Send to new connection first
		self.connections.add(address)
		print('New Connection:', address)
		self.sendConnections() # Update the connections for everyone else
		self.adding = None
		
	def pruneConnections(self):
		while self.running:
			remove = set()
			for address in self.connections.copy():
				if address == self.adding:
					continue # Don't remove a connection while adding it
				try:
					self.node.send(datetime.now(), name='ConnectionTest', host=address)
				except:
					print('Disconnected:', address)
					remove.add(address)
			if len(remove) > 0:
				self.connections -= remove
				self.sendConnections() # Update everyone's connection list
		self.node.close()

	def sendConnections(self):
		for address in self.connections.copy():
			self.node.send(self.connections, name='connections', host=address)

	def run(self):
		self.running = True
		self.prunethread = threading.Thread(target=self.pruneConnections)
		self.prunethread.start()

	def close(self):
		self.running = False
		self.prunethread.join()



if __name__ == '__main__':
	server = Server()
	print('Host:', server.node.host, "|", "Port:", server.node.port)
	import code; code.interact(local=locals())

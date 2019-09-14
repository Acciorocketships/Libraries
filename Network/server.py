from network import *
import threading
from datetime import datetime

class Server:

	SERVER_PORT = 8888

	def __init__(self):
		self.node = Network(host=None, port=Server.SERVER_PORT)
		self.node.connectCallback = self.newConnection
		self.connections = {}
		self.running = False
		self.adding = None
		self.run()

	def newConnection(self, address):
		self.adding = address
		while self.node.data.get('last connection info', None) is None:
			time.sleep(0.01)
		ip = {'address': address, 'port': self.node.data['last connection info']['port']}
		print('New Connection:', ip)
		self.connections[self.node.data['last connection info']['name']] = ip
		self.node.send(self.connections, name='connections', host=ip['address'], port=ip['port'])
		self.sendConnections()
		del self.node.data['last connection info']
		self.adding = None
		
	def pruneConnections(self):
		while self.running:
			remove = set()
			for name, ip in self.connections.copy().items():
				if ip['address'] == self.adding:
					continue # Don't remove a connection while adding it
				try:
					self.node.send(datetime.now(), name='last communication time', host=ip['address'])
				except:
					print('Disconnected: ' + str(ip['address']) + " : " + str(ip['port']))
					remove.add(name)
			if len(remove) > 0:
				for name in remove:
					del self.connections[name]
				self.sendConnections() # Update everyone's connection list
		self.node.close()

	def sendConnections(self):
		for name, ip in self.connections.copy().items():
			self.node.send(self.connections, name='connections', host=ip['address'], port=ip['port'])

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

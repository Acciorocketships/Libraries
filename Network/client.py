
from network import *

class Client:

	SERVER_ADDRESS = '168.235.109.146'
	SERVER_PORT = 8888

	def __init__(self, name, port):
		self.name = name
		self.port = port
		self.node = Network(port=port, otherport=Client.SERVER_PORT, otherhost=Client.SERVER_ADDRESS) # local: Neutrino.local   server: 168.235.109.146
		self.node.send({'name': self.name, 'port': port}, name='last connection info', host=Client.SERVER_ADDRESS, port=Client.SERVER_PORT)
		while self.node.data.get('connections', None) is None:
			time.sleep(0.01)
		self.data = self.node.data
		self.connections = self.node.data['connections']
		if self.name in self.connections:
			del self.connections[self.name]

	def get_connections(self):
		self.connections = self.node.data['connections']
		if self.name in self.connections:
			del self.connections[self.name]
		return self.connections

	def close(self):
		self.node.close()

	def get(self, data_name):
		return self.data.get(data_name, None)

	# if a node name is not specified, it will send to all connections
	def send(self, data, node_name=None, data_name='var'):
		self.get_connections()
		if node_name is None:
			self.node.send(data, name=data_name, host=self.connections[node_name]['address'], port=self.connections[node_name]['port'])
		else:
			for name, ip in self.connections.items():
				self.node.send(data, name=data_name, host=ip['address'], port=ip['port'])



if __name__ == '__main__':

	client = Client('node0', 8000)

	import code; code.interact(local=locals())
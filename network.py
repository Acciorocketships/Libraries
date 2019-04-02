import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))
import socket
import pickle
import threading
import time

class Network:

	def __init__(self,port=8080,host=None,otherport=None,otherhost=None):
		self.s = None # Socket Sender
		self.r = None # Socket Listener
		self.host = host if (host is not None) else socket.gethostname() # The host of this node
		self.otherhost = otherhost if (otherhost is not None) else self.host # The host that this node sends data to
		self.port = port # The port that this node binds to
		self.otherport = otherport # The port that this node sends data to
		self.data = {} # Data that has been recieved, lookup by name of variable
		self.initreceiver = False # Internal Flag
		self.initsender = False # Internal Flag
		self.callbacks = {} # Functions called with the variable as an input whenever new data arrives
		self.connections = {} # Lookup of connections by address
		self.listener = None # Listening Thread
		self.run()

	# Listens to a specific connection, updates the variable, calls the callback, and exits when done
	def clientThread(self,connection,address):
		buf = []
		while address in self.connections:
			try:
				databits = connection.recv(4096)
				buf = buf + databits.split(b'|!|')
				buf = buf[:-1]
			except socket.timeout:
				pass
			if len(buf) > 0:
				msg = buf[0].split(b'|~|')
				buf = buf[1:]
				name = msg[0].decode()
				var = pickle.loads(msg[1])
				self.data[name] = var
				if name in self.callbacks:
					self.callbacks[name](var)

	# Closes the socket listening to a specific connection
	def removeConnection(self,address):
		connection = self.connections[address]
		connection.close()
		del self.connections[address]

	# Closes the send socket
	def close(self):
		try:
			if self.initsender:
				self.s.close()
				self.initsender = False
			if self.initreceiver:
				self.r.close()
				self.initreceiver = False
		except:
			pass

	# Listening thread called by run()
	def listen(self):
		while self.initreceiver:
			try:
				connection, address = self.r.accept()
			except ConnectionAbortedError:
				break
			self.connections[address] = connection
			connection.settimeout(5.0)
			newclient = threading.Thread(target=self.clientThread, args=[connection,address])
			newclient.start()
		# Close threads
		for address in list(self.connections.keys()):
			self.removeConnection(address)

	# Listens for incoming messages and updates self.data
	def run(self,callbacks={}):
		# Initialization
		self.callbacks = callbacks
		if not self.initreceiver:
			self.r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			for i in range(100):
				try:
					self.r.bind((self.host, self.port))
					break
				except:
					print('Trying to connect to ' + str(self.host) + " : " + str(self.port))
					time.sleep(1)
			self.r.listen(5)
			self.initreceiver = True
		# Spin off new listener threads for each new connection
		self.listener = threading.Thread(target=self.listen)
		self.listener.start()

	# Sends a message
	def send(self,var,name='var',host=None,port=None):
		# Initialization
		if (host is not None and host != self.otherhost) or (port is not None and port != self.otherport) or (not self.initsender):
			if host is not None:
				self.otherhost = host
			if port is not None:
				self.portout = port
			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.s.connect((self.otherhost, self.otherport))
			self.initsender = True
		# Set up data to send
		databits = pickle.dumps(var)
		msg = name.encode()  + b'|~|' + databits + b'|!|'
		self.s.send(msg)
		# TODO: handshake verification

	# Gets the latest updated version of variable
	def get(self,name='var'):
		return self.data.get(name, None)




if __name__ == '__main__':
	import code
	n1 = Network(port=8000,otherport=8080)
	n2 = Network(port=8080,otherport=8000)
	code.interact(local=locals())



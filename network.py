import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))
import socket
import miniupnpc
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
		self.threads = []
		self.dataCallback = {} # {'varname': fucntion} Function called (with the variable as an input) whenever new data arrives.
		self.connectCallback = None # Function called (with the address as an input) whenever theres a new connection
		self.connections = {} # Lookup of connections by address
		self.listener = None # Listening Thread
		self.delim = b'|!endmsg!|'
		self.openPort(port)
		self.run()


	# Initiates port forwarding for a port (int) or list of ports, allowing incoming connections
	def openPort(self,ports):
		if type(port) == int:
			port = [port]
		upnp = miniupnpc.UPnP()
		upnp.discoverdelay = 10
		upnp.discover()
		upnp.selectigd()
		for port in ports:
			upnp.addportmapping(port, 'TCP', upnp.lanaddr, port, 'p2p', '')


	# Listens to a specific connection, updates the variable, calls the callback, and exits when done
	def clientThread(self,connection,address):
		buf = b''
		while address in self.connections:
			try:
				# Receive message
				buf += connection.recv(4096)
				# Split the message. Only take messages that end in delimiters
				s = buf.split(self.delim)
				i = 0
				msgs = []
				# Buffer must end with delimiter, so don't bother checking the last element
				for i in range(len(s)-1):
					# It cant be length 0 (['', e1, e2, ...] happens if buf starts with a delimiter)
					# If it's the second to last element, it must be followed by a delimiter
					if len(s[i])!=0 and (i<len(s)-2 or len(s[i+1])==0):
						msgs.append(s[i])
				# Interpret the message
				for msg in msgs:
					name, var = pickle.loads(msg)
					self.data[name] = var
					buf = buf[len(msg):]
					if name in self.dataCallback:
						self.dataCallback[name](var)
			except (socket.timeout, OSError, Exception) as err:
				pass


	# Closes the socket listening to a specific connection
	def removeConnection(self,address):
		connection = self.connections[address]
		connection.close()
		del self.connections[address]


	# Stops listening. use run() to start listening again.
	def close(self):
		# Stop loops and close sockets
		try:
			if self.initsender:
				self.s.close()
				self.initsender = False
			if self.initreceiver:
				self.r.close()
				self.initreceiver = False
		except:
			pass
		# Remove connections
		for address in list(self.connections.keys()):
			self.removeConnection(address)
		# Close threads
		for thread in self.threads:
			thread.join()
		self.listener.join()

	# Listening thread called by run()
	def listen(self):
		while self.initreceiver:
			try:
				connection, address = self.r.accept()
				address = address[0]
				self.connections[address] = connection
				connection.settimeout(5.0)
				newclient = threading.Thread(target=self.clientThread, args=[connection,address])
				newclient.start()
				self.threads.append(newclient)
				if self.connectCallback is not None:
					self.connectCallback(address)
			except (ConnectionAbortedError, OSError):
				pass

	# Listens for incoming messages and updates self.data
	def run(self):
		# Initialization
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
				self.otherport = port
			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.s.connect((self.otherhost, self.otherport))
			self.initsender = True
		# Set up data to send
		databits = pickle.dumps((name,var))
		msg = databits + self.delim
		self.s.send(msg)
		# TODO: handshake verification

	# Gets the latest updated version of variable
	def get(self,name='var'):
		return self.data.get(name, None)





if __name__ == '__main__':
	import code
	n1 = Network(port=8001,otherport=8000)
	n2 = Network(port=8000,otherport=8001)
	code.interact(local=locals())



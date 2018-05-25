import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

import socket
import select
import pickle
import threading
import time
import sys

class Network:

	def __init__(self,host='localhost',portin=8000,portout=8080):
		self.s = None
		self.r = None
		self.host = host
		if host == 'localhost':
			self.host = socket.gethostname()
		self.portin = portin
		self.portout = portout
		self.data = {}
		self.receiver = False
		self.sender = False
		self.callbacks = {}
		self.connections = {}
		self.listener = None

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
		if self.sender:
			self.s.close()
			self.sender = False
		if self.receiver:
			self.r.close()
			self.receiver = False

	# Gets the latest updated version of variable
	def get(self,name):
		return self.data.get(name, None)

	def listen(self):
		while self.receiver:
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
		if not self.receiver:
			self.r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			for i in range(100):
				try:
					self.r.bind((self.host, self.portin))
					break
				except:
					print('Trying to connect to ' + str(self.host) + " : " + str(self.portin))
					time.sleep(1)
			self.r.listen(5)
			self.receiver = True
		# Spin off new listener threads for each new connection
		self.listener = threading.Thread(target=self.listen)
		self.listener.start()

	# Sends a message
	def send(self,var,name='var'):
		# Initialization
		if not self.sender:
			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.s.connect((self.host, self.portout))
			self.sender = True
		# Set up data to send
		databits = pickle.dumps(var)
		msg = name.encode()  + b'|~|' + databits + b'|!|'
		self.s.send(msg)
		# TODO: handshake verification
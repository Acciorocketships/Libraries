
from network import *

if __name__ == '__main__':

	n = Network(port=8888, otherport=8888, otherhost='168.235.109.146')

	n.send('connecting')

	print(n.get('connections'))

	import code; code.interact(local=locals())

	# lastupdate variable, updated in callback for ConnectionTest from server

	# send port when connecting

	# no server mode:
	# if client loses connection to another client AND the server, it should update other nodes with the connection list
	# there should be a member function to request the connection list, in case you want to get it from another client if the server is down
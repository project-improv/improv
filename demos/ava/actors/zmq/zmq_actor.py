# import asyncio
import time
import traceback

import zmq as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, ETERM, EAGAIN
from zmq.log.handlers import PUBHandler

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZMQActor(Actor):
    """Base class for an actor that uses ZMQ.

    TODO:
    - ONE ACTOR, ONE CONTEXT, TWO DIFF SOCKETS 
    - MULTIPLE ACTORS (ACQ AND STIM) = SHARED CONTEXT, TWO DIFF SOCKETS
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info("Instantiating a global ZMQ context.")

        # self.context = zmq.Context()
        # zmq.Context.instance() returns a global Context instance — best practice for single-threaded applications instead of passing around Context instances
        self.context = zmq.Context.instance() 

    def setSocket(self, socket_type, socket_class=None):
        """Sets one socket for the ZMQ context.

        Args:
            socket_type (int): The socket type, for example:
                REQ, REP, PUB, SUB, PAIR, DEALER, ROUTER, PULL, PUSH, etc.
                
                NOTE: for socket type int values:
                https://pyzmq.readthedocs.io/en/latest/api/zmq.html#zmq.SocketType
            
            socket_class (zmq.Socket, optional): The socket class to instantiate, if different from the default for this Context, e.g., for creating an asyncio socket attached to a default Context or vice versa.
                Defaults to None.

        Returns:
            socket (zmq.Socket): ZMQ socket
        """
        
        # default is to use socket type int values
        # if socket_type == "PUB":
        #     socket_type = 1
        
        # if socket_type == "SUB":
        #     socket_type = 2
        logger.info(f"Setting up {socket_type} socket.")
        socket = self.context.socket(socket_type, socket_class)
        return socket

    def bindSocket(self, socket, ip, port):
        """Wrapper for zmq.Socket.bind — bind the socket to an address.
        
        NOTE: specifically for a TCP address

        TODO: add functionality for more address types?

        Args:
            socket (zmq.Socket): ZMQ socket
            ip (str): IP address
            port (str): Port number
        """

        address = f"tcp://{ip}:{port}"
        socket.bind(address)
        logger.info(f"Socket bound: {address}")

    def connectSocket(self, socket, ip, port):
        """Wrapper for zmq.Socket.connect — connect the socket to an address.

        Args:
            socket (zmq.Socket): ZMQ socket
            ip (str): IP address
            port (str): Port number
        """
        
        address = f"tcp://{ip}:{port}"
        socket.connect(address)
        logger.info(f"Socket connected: {address}")
        # TODO: options, choice?
        # PUB=1, SUB = 2, always connect with SUB, and more, but not PUB
        if socket.socket_type == 2:
            socket.setsockopt(SUBSCRIBE, b"")

    def sendMsg(self, socket, msg, msg_type=None):
        """Wrapper for zmq.Socket.send and zmq.Socket.send_multipart
        
        TODO: add functionality for specific message types, objects?

        Args:
            socket (zmq.Socket): ZMQ socket
            msg (any): Message to send, single  message frame or sequence of buffers
            msg_type (str): Single or multi-part
        """

        if msg_type == "multipart":
            socket.send_multipart(msg)
        if msg_type == "pyobj":
            socket.send_pyobj(msg)
        elif msg_type == None: 
            socket.send(msg)

    def recvMsg(self, socket, msg_type=None, flags=0):
        """Wrapper for zmq.Socket.recv and zmq.Socket.recv_multipart.
        
        TODO: add functionality for specific message types, objects?

        Args:
            socket (zmq.Socket): ZMQ socket
            msg (any): Message to send, single  message frame or sequence of buffers
            msg_type (str): Single or multi-part. Defaults to None.
            flags(int): 0 or NOBLOCK. Defaults to 0. When using NOBLOCK, this a ZMQError if no messages have arrived or waits until a message arrives

        Returns:
            msg (List[Frame] or List[Bytes]): Message
        """
        
        try:
            if msg_type == "multipart":
                msg = socket.recv_multipart(flags=flags)
            if msg_type == "pyobj":
                msg = socket.recv_pyobj(flags=flags)
            elif msg_type == None: 
                msg = socket.recv(flags=flags)
        except ZMQError as e:
            logger.info(f"ZMQ error: {e.msg}")
            if e.errno == ETERM:
                pass  # interrupted  - pass or break if in try loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise  # raise real error
        
        return msg

    def closeSocket(self, socket):
        """Wrapper for zmq.Socket.close and zmq.Socket.term — closes and terminates the socket.

        NOTE: unnecessary since these are native methods?
        TODO: info regard what socket is closed, what actor using the socket?

        Args:
            socket (zmq.Socket): ZMQ socket
        """

        logger.info("Closing and terminating socket.")
        socket.close()
        socket.term()

        if socket.closed is True:
            logger.info("Socket is closed.")

    def termContext(self, context, linger=None):
        """Wrapper for zmq.context.destroy — closes and terminated the context.
        
        NOTE: closes all sockets associated with this context, then terminates

        TODO: info regard what socket and context are closed, what actor using the socket?

        Args:
            socket (zmq.Socket): ZMQ socket
        """

        logger.info("Closing and terminating context.")

        try:
            # context.term(linger=linger)
            context.destroy(linger=linger)
        except ZMQError as e:
            logger.info(f"ZMQ error: {e.msg}")
            if e.errno == ETERM:
                logger.info("Blocking operations in progress on sockets open within context — close sockets before context.")
        
        if context.closed is True:
            logger.info("Context is closed.")
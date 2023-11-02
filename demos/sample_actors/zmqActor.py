import asyncio
import time
import zmq.asyncio
import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, ETERM, EAGAIN

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZmqActor(Actor):
    """
    ZMQ actor using the Pub-Sub or Request-Reply pattern.
    """
    def __init__(self, *args, pub_sub=True, rep_req=False, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Constructed ZMQ Actor")
        self.pub_sub_flag = pub_sub  # default
        self.rep_req_flag = rep_req

        self.send_socket = None
        self.recv_socket = None

        self.req_socket = None
        self.rep_socket = None

        self.address = None
        self.context = zmq.Context.instance()

    def setSendSocket(self, ip, port, timeout=0.001):
        """Publish: wrapper for zmq.Socket.bind — bind the socket to an address.
        
        NOTE: specifically for a TCP address

        Args:
            socket (zmq.Socket): ZMQ socket.
            ip (str): IP address.
            port (str): Port number.
        """

        self.send_socket = self.context.socket(PUB)
        # bind to the socket according to the ip and port
        self.address = f"tcp://{ip}:{port}"
        self.send_socket.bind(self.address)
        time.sleep(timeout)

    def setRecvSocket(self, ip, port, timeout=0.001):
        """Subscribe: wrapper for zmq.Socket.connect — connect the socket to an address.

        NOTE: specifically for a TCP address

        Args:
            socket (zmq.Socket): ZMQ socket.
            ip (str): IP address.
            port (str): Port number.
        """
         
        self.recv_socket = self.context.socket(SUB)
        self.address = f"tcp://{ip}:{port}"
        self.recv_socket.connect(self.address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)

    def sendMsg(self, msg, msg_type="pyobj"):
        """Sends a message to the controller: wrapper for zmq.Socket.send, send_pyobj, send_multipart.

        Args:
            socket (zmq.Socket): ZMQ socket
            msg (any): Message to send, single  message frame or sequence of buffers
            msg_type (str): Single or multi-part. Defualts to pyobj, Python object.
        """
        try:
            if msg_type == "multipart":
                self.socket.send_multipart(msg)
            if msg_type == "pyobj":
                self.socket.send_pyobj(msg)
            elif msg_type == "single": 
                self.socket.send(msg)
        except ZMQError as e:
            logger.info(f"ZMQ error: {e.msg}")
            if e.errno == ETERM:
                pass  # interrupted  - pass or break if in try loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise  # raise real error     
        
    def recvMsg(self, msg_type="pyobj", flags=NOBLOCK):
        """Receive message: wrapper for zmq.Socket.recv, recv_pyobj, recv_multipart.

        Args:
            socket (zmq.Socket): ZMQ socket
            msg (any): Message to receive, single  message frame or sequence of buffers.
            msg_type (str): Single or multi-part. Defualts to pyobj, Python object.
            flags(int): 0 or NOBLOCK. Defaults to NOBLOCK. When using NOBLOCK or 0, this raises a ZMQError if no messages have arrived or waits until a message arrives, respectively.

        Returns:
            msg (List[Frame], List[Bytes]; Python object; bytes or Frame): Message received.
        """

        try:
            if msg_type == "multipart":
                msg = self.socket.recv_multipart(flags=flags)
            if msg_type == "pyobj":
                msg = self.socket.recv_pyobj(flags=flags)
            elif msg_type == None: 
                msg = self.socket.recv(flags=flags)
        except ZMQError as e:
            logger.info(f"ZMQ error: {e.msg}")
            if e.errno == ETERM:
                pass  # interrupted  - pass or break if in try loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise  # raise real error
        return msg
    
    def setReqSocket(self, ip, port, timeout=0.001):
        """
        Sets up the request socket for the actor.

        NOTE: specifically for a TCP address

        Args:
            socket (zmq.Socket): ZMQ socket.
            ip (str): IP address.
            port (str): Port number.
        """

        self.req_socket = self.context.socket(REQ)
        # bind to the socket according to the ip and port
        self.address = f"tcp://{ip}:{port}"
        time.sleep(timeout)

    def setRepSocket(self, ip, port, timeout=0.001):
        """
        Sets up the reply socket for the actor.

        NOTE: specifically for a TCP address

        Args:
            socket (zmq.Socket): ZMQ socket.
            ip (str): IP address.
            port (str): Port number.
        """

        self.rep_socket = self.context.socket(REP)
        self.address = f"tcp://{ip}:{port}"
        self.rep_socket.bind(self.address)
        time.sleep(timeout)

    def requestMsg(self, msg):
        """Safe version of send/receive with controller.
        Based on the Lazy Pirate pattern [here]
        (https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)
        
        Args:
            msg (Python object): Message to send.
        
        Returns:
            reply (Python object): Reply from request.
        """

        REQUEST_TIMEOUT = 2500
        REQUEST_RETRIES = 3

        retries_left = REQUEST_RETRIES

        try:
            self.req_socket.connect(self.address)
            logger.info(f"Sending {msg} to controller.")
            self.req_socket.send_pyobj(msg)
            reply = None

            while True:
                ready = self.req_socket.poll(REQUEST_TIMEOUT)

                if ready:
                    reply = self.req_socket.recv_pyobj()
                    logger.info(f"Received {reply} from controller.")
                    break
                else:
                    retries_left -= 1
                    logger.info("No response from server.")

                # try to close and reconnect
                self.req_socket.setsockopt(LINGER, 0)
                self.req_socket.close()
                if retries_left == 0:
                    logger.info("Server seems to be offline. Giving up.")
                    break

                logger.info("Attempting to reconnect to server...")

                self.req_socket = self.context.socket(REQ)
                self.req_socket.connect(self.address)

                logger.info(f"Resending {msg} to controller.")
                self.req_socket.send_pyobj(msg)

        except asyncio.CancelledError:
            pass
        return reply

    def replyMsg(self, reply):
        """Safe version of receive/reply with controller.

        Args:
            reply (Python object): Message to send upon reply.

        Returns:
            msg (Python object): Message received.
        """

        msg = self.rep_socket.recv_pyobj()
        time.sleep(0.001)  
        self.rep_socket.send_pyobj(reply)
        return msg

    def put(self, msg=None):
        if self.pub_sub_flag and self.rep_req_flag:
            logger.error("EXACTLY ONE of pub_sub_flag and rep_req_flag may be set to true.")
            return
        elif (not self.pub_sub_flag) and (not self.rep_req_flag):
            logger.error("EXACTLY ONE of pub_sub_flag and rep_req_flag may be set to false.")
            return

        if (self.pub_sub_flag):
            logger.info(f"Putting message {msg} using PUB/SUB.")
            return self.sendMsg(msg)
        elif (self.rep_req_flag):
            logger.info(f"Putting message {msg} using REQ/REP.")
            return self.requestMsg(msg)
        
    def get(self, reply=None):
        if self.pub_sub_flag and self.rep_req_flag:
            logger.error("EXACTLY ONE of pub_sub_flag and rep_req_flag may be set to true.")
            return
        elif (not self.pub_sub) and (not self.rep_req_flag):
            logger.error("EXACTLY ONE of pub_sub_flag and rep_req_flag may be set to false.")
            return
    
        if (self.pub_sub_flag):
            logger.info(f"Getting message with PUB/SUB.")
            return self.recvMsg()
        elif (self.rep_req_flag):
            logger.info(f"Getting message using reply {reply}.")
            return self.replyMsg(reply)
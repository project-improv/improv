from improv.actor import Actor

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, EAGAIN, ETERM
from zmq.log.handlers import PUBHandler
import traceback

import zmq.asyncio
import asyncio
import time

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZmqPSActor(Actor):
    """
    Zmq actor with PUB/SUB pattern.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.context = zmq.Context()

    def setSendSocket(self, ip, port, timeout=0.001):
        """
        Sets up the send socket for the actor.
        """

        self.send_socket = self.context.socket(PUB)
        # bind to the socket according to the ip and port
        self.address = f"tcp://{ip}:{port}"
        self.send_socket.bind(self.address)
        time.sleep(timeout)

    def setRecvSocket(self, ip, port, timeout=0.001):
        """
        Sets up the receive socket for the actor.
        """

        self.recv_socket = self.context.socket(SUB)
        self.address = f"tcp://{ip}:{port}"
        self.recv_socket.connect(self.address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)

    def sendMsg(self, msg, msg_type="pyobj"):
        """
        Sends a message to the controller.
        """

        if msg_type == "multipart":
            self.send_socket.send_multipart(msg)
        if msg_type == "pyobj":
            self.send_socket.send_pyobj(msg)
        elif msg_type == None: 
            self.send_socket.send(msg)

    def recvMsg(self, msg_type="pyobj", flags=NOBLOCK):
        """
        Receives a message from the controller.
        """

        try:
            if msg_type == "multipart":
                recv_msg = self.recv_socket.recv_multipart(flags=flags)
            if msg_type == "pyobj":
                recv_msg = self.recv_socket.recv_pyobj(flags=flags)
            elif msg_type == None: 
                recv_msg = self.recv_socket.recv(flags=flags)
        except ZMQError as e:
            logger.info(f"ZMQ error: {e}")
            if e.errno == ETERM:
                pass  # interrupted  - pass or break if in try loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise  # raise real error
        return recv_msg
    

class ZmqRRActor(Actor):
    """
    Zmq actor with REQ/REP pattern.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.req_socket = None
        self.rep_socket = None
        self.address = None
        self.context = zmq.Context()

    def setReqSocket(self, ip, port, timeout=0.001):
        """
        Sets up the request socket for the actor.
        """

        self.req_socket = self.context.socket(REQ)
        # bind to the socket according to the ip and port
        self.address = f"tcp://{format}:{ip}"
        time.sleep(timeout)

    def setRepSocket(self, ip, port, timeout=0.001):
        """
        Sets up the reply socket for the actor.
        """

        self.rep_socket = self.context.socket(REP)
        self.address = f"tcp://{ip}:{port}"
        self.rep_socket.bind(self.address)
        time.sleep(timeout)

    def requestMsg(self, msg):
        """Safe version of send/receive with controller.
        Based on the Lazy Pirate pattern [here]
        (https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)
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

        self.req_socket.close()
        return reply
    

    def replyMsg(self, reply):
        """
        Safe version of receive/reply with controller.
        """
        msg = self.rep_socket.recv_pyobj()
        time.sleep(0.001)  
        self.rep_socket.send_pyobj(reply)
        self.rep_socket.close()
        return msg
import asyncio
import time

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK
from zmq.log.handlers import PUBHandler
import zmq.asyncio

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZmqActor(Actor):
    """
    Zmq actor with pub/sub or rep/req pattern.
    """
    def __init__(self, *args, type='PUB', ip='localhost', port=5555, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Constructed Zmq Actor")
        if str(type) in 'PUB' or str(type) in 'SUB':
            self.pub_sub_flag = True         #default
        else: self.pub_sub_flag = False
        self.rep_req_flag = not self.pub_sub_flag
        self.ip = ip
        self.port = port
        self.address = "tcp://{}:{}".format(self.ip, self.port)

        self.send_socket = None
        self.recv_socket = None
        self.req_socket = None
        self.rep_socket = None

        self.context = zmq.Context.instance()

        if str(type) in 'PUB':
            self.setSendSocket()
        elif str(type) in 'SUB':
            self.setRecvSocket()

    def setSendSocket(self, timeout=0.001):
        """
        Sets up the send socket for the actor.
        """
        self.send_socket = self.context.socket(PUB)
        self.send_socket.bind(self.address)
        time.sleep(timeout)

    def setRecvSocket(self, timeout=0.001):
        """
        Sets up the receive socket for the actor.
        """
        self.recv_socket = self.context.socket(SUB)
        self.recv_socket.connect(self.address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)

    def sendMsg(self, msg):
        """
        Sends a message to the controller.
        """
        self.send_socket.send_pyobj(msg)
        self.send_socket.close()

    def recvMsg(self):
        """
        Receives a message from the controller.
        """
        recv_msg = ""
        while True:
            try:
                recv_msg = self.recv_socket.recv_pyobj(flags=NOBLOCK)
                break
            except Again:
                pass
        self.recv_socket.close()
        return recv_msg
    
    def setReqSocket(self, timeout=0.0001):
        """
        Sets up the request socket for the actor.
        """
        self.req_socket = self.context.socket(REQ)
        time.sleep(timeout)

    def setRepSocket(self, timeout=0.0001):
        """
        Sets up the reply socket for the actor.
        """

        self.rep_socket = self.context.socket(REP)
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
            logger.debug(f"Sending {msg} to controller.")
            self.req_socket.send_pyobj(msg)
            reply = None

            while True:
                ready = self.req_socket.poll(REQUEST_TIMEOUT)

                if ready:
                    reply = self.req_socket.recv_pyobj()
                    logger.debug(f"Received {reply} from controller.")
                    break
                else:
                    retries_left -= 1
                    logger.debug("No response from server.")

                # try to close and reconnect
                self.req_socket.setsockopt(LINGER, 0)
                self.req_socket.close()
                if retries_left == 0:
                    logger.debug("Server seems to be offline. Giving up.")
                    break

                logger.debug("Attempting to reconnect to server...")

                self.req_socket = self.context.socket(REQ)
                self.req_socket.connect(self.address)

                logger.debug(f"Resending {msg} to controller.")
                self.req_socket.send_pyobj(msg)

        except asyncio.CancelledError:
            pass

        self.req_socket.close()
        return reply

    def replyMsg(self, reply, delay=0.0001):
        """
        Safe version of receive/reply with controller.
        """
        msg = self.rep_socket.recv_pyobj()
        time.sleep(delay)  
        self.rep_socket.send_pyobj(reply)
        self.rep_socket.close()
        return msg

    def put(self, msg=None):
        if self.pub_sub_flag:
            logger.debug(f"putting message {msg} using pub/sub")
            return self.sendMsg(msg)
        else:
            logger.debug(f"putting message {msg} using rep/req")
            return self.requestMsg(msg)
        
    def get(self, reply=None):
        if self.pub_sub_flag:
            logger.debug(f"getting message with pub/sub")
            return self.recvMsg()
        else:
            logger.debug(f"getting message using reply {reply} with pub/sub")
            return self.replyMsg(reply)
        

# class ZmqPSActor(Actor):
#     """
#     Zmq actor with PUB/SUB pattern.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.send_socket = None
#         self.recv_socket = None
#         self.address = None
#         self.context = zmq.Context()

#     def setSendSocket(self, ip, port, timeout=0.001):
#         """
#         Sets up the send socket for the actor.
#         """

#         self.send_socket = self.context.socket(PUB)
#         # bind to the socket according to the ip and port
#         self.address = "tcp://{}:{}".format(ip, port)
#         self.send_socket.bind(self.address)
#         time.sleep(timeout)

#     def setRecvSocket(self, ip, port, timeout=0.001):
#         """
#         Sets up the receive socket for the actor.
#         """

#         self.recv_socket = self.context.socket(SUB)
#         self.address = "tcp://{}:{}".format(ip, port)
#         self.recv_socket.connect(self.address)
#         self.recv_socket.setsockopt(SUBSCRIBE, b"")
#         time.sleep(timeout)

#     def sendMsg(self, msg):
#         """
#         Sends a message to the controller.
#         """

#         self.send_socket.send_pyobj(msg)
#         self.send_socket.close()

#     def recvMsg(self):
#         """
#         Receives a message from the controller.
#         """

#         recv_msg = ""
#         while True:
#             try:
#                 recv_msg = self.recv_socket.recv_pyobj(flags=NOBLOCK)
#                 break
#             except Again:
#                 pass
#         self.recv_socket.close()
#         return recv_msg


# class ZmqRRActor(Actor):
#     """
#     Zmq actor with REQ/REP pattern.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.req_socket = None
#         self.rep_socket = None
#         self.address = None
#         self.context = zmq.Context()

#     def setReqSocket(self, ip, port, timeout=0.001):
#         """
#         Sets up the request socket for the actor.
#         """

#         self.req_socket = self.context.socket(REQ)
#         # bind to the socket according to the ip and port
#         self.address = "tcp://{}:{}".format(ip, port)
#         time.sleep(timeout)

#     def setRepSocket(self, ip, port, timeout=0.001):
#         """
#         Sets up the reply socket for the actor.
#         """

#         self.rep_socket = self.context.socket(REP)
#         self.address = "tcp://{}:{}".format(ip, port)
#         self.rep_socket.bind(self.address)
#         time.sleep(timeout)

#     def requestMsg(self, msg):
#         """Safe version of send/receive with controller.
#         Based on the Lazy Pirate pattern [here]
#         (https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)
#         """

#         REQUEST_TIMEOUT = 2500
#         REQUEST_RETRIES = 3

#         retries_left = REQUEST_RETRIES

#         try:
#             self.req_socket.connect(self.address)
#             logger.info(f"Sending {msg} to controller.")
#             self.req_socket.send_pyobj(msg)
#             reply = None

#             while True:
#                 ready = self.req_socket.poll(REQUEST_TIMEOUT)

#                 if ready:
#                     reply = self.req_socket.recv_pyobj()
#                     logger.info(f"Received {reply} from controller.")
#                     break
#                 else:
#                     retries_left -= 1
#                     logger.info("No response from server.")

#                 # try to close and reconnect
#                 self.req_socket.setsockopt(LINGER, 0)
#                 self.req_socket.close()
#                 if retries_left == 0:
#                     logger.info("Server seems to be offline. Giving up.")
#                     break

#                 logger.info("Attempting to reconnect to server...")

#                 self.req_socket = self.context.socket(REQ)
#                 self.req_socket.connect(self.address)

#                 logger.info(f"Resending {msg} to controller.")
#                 self.req_socket.send_pyobj(msg)

#         except asyncio.CancelledError:
#             pass

#         self.req_socket.close()
#         return reply
    

#     def replyMsg(self, reply):
#         """
#         Safe version of receive/reply with controller.
#         """
#         msg = self.rep_socket.recv_pyobj()
#         time.sleep(0.001)  
#         self.rep_socket.send_pyobj(reply)
#         self.rep_socket.close()
#         return msg

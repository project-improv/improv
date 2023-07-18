from improv.actor import Actor

import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again
from zmq.log.handlers import PUBHandler
import traceback

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
        self.send_socket = None
        self.recv_socket = None
        self.address = None
        self.context = None

    def setSendSocket(self, ip, port, timeout=0.001):
        """
        Sets up the send socket for the actor.
        """

        self.context = zmq.Context()
        self.send_socket = self.context.socket(PUB)
        # bind to the socket according to the ip and port
        self.address = "tcp://{}:{}".format(ip, port)
        self.send_socket.bind(self.address)
        time.sleep(timeout)

    def setRecvSocket(self, ip, port, timeout=0.001):
        """
        Sets up the receive socket for the actor.
        """

        self.context = zmq.Context()
        self.recv_socket = self.context.socket(SUB)
        self.address = "tcp://{}:{}".format(ip, port)
        self.recv_socket.connect(self.address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)

    def sendMsg(self, msg):
        """
        Sends a message to the controller.
        """

        self.send_socket.send_pyobj(msg)
        self.send_socket.close()
        self.context.term()

    def recvMsg(self):
        """
        Receives a message from the controller.
        """

        recv_msg = ""
        while True:
            try:
                recv_msg = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
                break
            except Again:
                pass
        self.recv_socket.close()
        self.context.term()
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
        self.context = None

    def setReqSocket(self, ip, port, timeout=0.001):
        """
        Sets up the request socket for the actor.
        """

        self.context = zmq.Context()
        self.req_socket = self.context.socket(REQ)
        # bind to the socket according to the ip and port
        self.address = "tcp://{}:{}".format(ip, port)
        time.sleep(timeout)

    def setRepSocket(self, ip, port, timeout=0.001):
        """
        Sets up the reply socket for the actor.
        """

        self.context = zmq.Context()
        self.rep_socket = self.context.socket(REP)
        self.address = "tcp://{}:{}".format(ip, port)
        self.rep_socket.bind(self.address)
        time.sleep(timeout)

    async def requestMsg(self, msg):
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
            await self.req_socket.send_pyobj(msg)
            reply = None

            while True:
                ready = await self.req_socket.poll(REQUEST_TIMEOUT)

                if ready:
                    reply = await self.req_socket.recv_pyobj()
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
                await self.req_socket.send_pyobj(msg)

        except asyncio.CancelledError:
            pass

        self.req_socket.close()
        self.context.term()
        return reply

    async def replyMsg(self, reply):
        """
        Safe version of receive/reply with controller.
        """
        msg = await self.rep_socket.recv_pyobj()
        await self.rep_socket.send_pyobj(reply)
        self.rep_socket.close()
        self.context.term()
        return msg

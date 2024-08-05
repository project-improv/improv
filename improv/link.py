import asyncio
import json
import logging
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import zmq
from zmq import SocketOption

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ZmqLink:
    def __init__(self, socket: zmq.Socket, name, topic=None):
        self.name = name
        self.real_executor = None
        self.socket = socket  # this must already be set up
        self.socket_type = self.socket.getsockopt(SocketOption.TYPE)
        if self.socket_type == zmq.PUB and topic is None:
            raise Exception("Cannot open PUB link without topic")
        self.status = "pending"
        self.result = None
        self.topic = topic

    @property
    def _executor(self):
        if not self.real_executor:
            self.real_executor = ThreadPoolExecutor(max_workers=cpu_count())
        return self.real_executor

    def get(self, timeout=None):
        if timeout is None:
            if self.socket_type == zmq.SUB:
                res = self.socket.recv_multipart()
                return json.loads(res[1].decode("utf-8"))

            return self.socket.recv_pyobj()

        msg_ready = self.socket.poll(timeout=timeout)
        if msg_ready == 0:
            raise TimeoutError

        if self.socket_type == zmq.SUB:
            res = self.socket.recv_multipart()
            return json.loads(res[1])

        return self.socket.recv_pyobj()

    def get_nowait(self):
        return self.get(timeout=0)

    def put(
        self, item
    ):  # TODO: is it a problem we're implicitly handling inputs differently here?
        if self.socket_type == zmq.PUB:
            self.socket.send_multipart(
                [self.topic.encode("utf-8"), json.dumps(item).encode("utf-8")]
            )
        else:
            self.socket.send_pyobj(item)

    async def put_async(self, item):
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(self._executor, self.put, item)
        return res

    async def get_async(self):
        loop = asyncio.get_event_loop()
        self.status = "pending"
        # try:
        self.result = await loop.run_in_executor(self._executor, self.get)
        self.status = "done"
        return self.result
        # TODO: explicitly commenting these out because testing them is hard.
        #   It's better to let them bubble up so that we don't miss them
        #   due to being caught without a clear reproducible mechanism
        # except CancelledError:
        #     logger.info("Task {} Canceled".format(self.name))
        # except EOFError:
        #     logger.info("probably killed")
        # except FileNotFoundError:
        #     logger.info("probably killed")
        # except Exception as e:
        #     logger.exception("Error in get_async: {}".format(e))

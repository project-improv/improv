from __future__ import annotations

import logging
import signal
from logging import handlers
from logging.handlers import QueueHandler

import zmq
from zmq import SocketOption
from zmq.log.handlers import PUBHandler

from improv.messaging import LogInfoMsg

local_log = logging.getLogger(__name__)

DEBUG = True

# TODO: ideally there should be some kind of drain at shutdown time
#  so we don't miss any log messages, but that would make shutdown
#  also take longer. TBD?


def bootstrap_log_server(
    nexus_hostname, nexus_port, log_filename="global.log", logger_pull_port=None
):
    if DEBUG:
        local_log.addHandler(logging.FileHandler("log_server.log"))
    try:
        log_server = LogServer(
            nexus_hostname, nexus_port, log_filename, logger_pull_port
        )
        log_server.register_with_nexus()
        log_server.serve(log_server.read_and_log_message)
    except Exception as e:
        local_log.error(e)
        for handler in local_log.handlers:
            handler.close()


class ZmqPullListener(handlers.QueueListener):
    def __init__(self, ctx, /, *handlers, **kwargs):
        self.sentinel = False
        self.ctx = ctx
        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:0")
        pull_port_string = self.pull_socket.getsockopt_string(
            SocketOption.LAST_ENDPOINT
        )
        self.pull_port = int(pull_port_string.split(":")[-1])
        super().__init__(self.pull_socket, *handlers, **kwargs)

    def dequeue(self, block=True):
        msg = None
        while msg is None:
            if self.sentinel:
                return handlers.QueueListener._sentinel
            msg_ready = self.queue.poll(timeout=1000)
            if msg_ready != 0:
                msg = self.queue.recv_json()
        return logging.makeLogRecord(msg)

    def enqueue_sentinel(self):
        self.sentinel = True


class ZmqLogHandler(QueueHandler):
    def __init__(self, hostname, port, ctx=None):
        self.ctx = ctx if ctx else zmq.Context()
        self.ctx.setsockopt(SocketOption.LINGER, 0)
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{hostname}:{port}")
        super().__init__(self.socket)

    def enqueue(self, record):
        self.queue.send_json(record.__dict__)

    def close(self):
        self.queue.close(linger=0)


class LogServer:
    def __init__(self, nexus_hostname, nexus_comm_port, log_filename, pub_port):
        self.running = True
        self.pub_port: int | None = pub_port if pub_port else 0
        self.pub_socket: zmq.Socket | None = None
        self.log_filename = log_filename
        self.nexus_hostname: str = nexus_hostname
        self.nexus_comm_port: int = nexus_comm_port
        self.zmq_context: zmq.Context | None = None
        self.nexus_socket: zmq.Socket | None = None
        self.pull_socket: zmq.Socket | None = None
        self.listener: ZmqPullListener | None = None

        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            signal.signal(s, self.stop)

    def register_with_nexus(self):
        # connect to nexus
        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(SocketOption.LINGER, 0)
        self.nexus_socket = self.zmq_context.socket(zmq.REQ)
        self.nexus_socket.connect(f"tcp://{self.nexus_hostname}:{self.nexus_comm_port}")

        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")
        pub_port_string = self.pub_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
        self.pub_port = int(pub_port_string.split(":")[-1])

        self.listener = ZmqPullListener(
            self.zmq_context,
            # logging.StreamHandler(sys.stdout),
            logging.FileHandler(self.log_filename),
            PUBHandler(self.pub_socket, self.zmq_context, "nexus_logging"),
        )

        self.listener.start()

        local_log.info("logger started listening")

        port_info = LogInfoMsg(
            "broker",
            self.listener.pull_port,
            self.pub_port,
            "Port up and running, ready to log messages",
        )

        self.nexus_socket.send_pyobj(port_info)
        local_log.info("logger sent message to nexus")
        self.nexus_socket.recv_pyobj()

        local_log.info("logger got message from nexus")

        return

    def serve(self, log_func):
        local_log.info("logger serving")
        while self.running:
            log_func()  # this is more testable but may have a performance overhead
        self.shutdown()

    def read_and_log_message(self):  # receive and send back out
        pass

    def shutdown(self):
        if self.listener:
            self.listener.stop()

            for handler in self.listener.handlers:
                try:
                    handler.close()
                except Exception as e:
                    local_log.error(e)

        if self.pull_socket:
            self.pull_socket.close(linger=0)

        if self.nexus_socket:
            self.nexus_socket.close(linger=0)

        if self.pub_socket:
            self.pub_socket.close(linger=0)

        for handler in local_log.handlers:
            handler.close()

        if self.zmq_context:
            self.zmq_context.destroy(linger=0)

    def stop(self, signum, frame):
        local_log.info(f"Log server shutting down due to signal {signum}")

        self.running = False

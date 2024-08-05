import logging
import multiprocessing
import os
import signal
import time

import pytest
import subprocess

import zmq

from improv.actor import ZmqActor
from improv.harvester import bootstrap_harvester
from improv.nexus import Nexus

redis_port_num = 6379
WAIT_TIMEOUT = 10

SERVER_COUNTER = 0

signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)


@pytest.fixture
def ports():
    global SERVER_COUNTER
    CONTROL_PORT = 30000
    OUTPUT_PORT = 30001
    LOGGING_PORT = 30002
    ACTOR_IN_PORT = 30003
    yield (
        CONTROL_PORT + SERVER_COUNTER,
        OUTPUT_PORT + SERVER_COUNTER,
        LOGGING_PORT + SERVER_COUNTER,
        ACTOR_IN_PORT + SERVER_COUNTER,
    )
    SERVER_COUNTER += 4


@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__) + "/configs")
    yield None
    os.chdir(prev)


@pytest.fixture
def set_dir_config_parent():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    yield None
    os.chdir(prev)


@pytest.fixture
def sample_nex(setdir, ports):
    nex = Nexus("test")
    try:
        nex.create_nexus(
            file="good_config.yaml",
            store_size=40000000,
            control_port=ports[0],
            output_port=ports[1],
        )
    except Exception as e:
        print(f"error caught in test harness during create_nexus step: {e}")
        logging.error(f"error caught in test harness during create_nexus step: {e}")
        raise e
    yield nex
    try:
        nex.destroy_nexus()
    except Exception as e:
        print(f"error caught in test harness during destroy_nexus step: {e}")
        logging.error(f"error caught in test harness during destroy_nexus step: {e}")
        raise e


@pytest.fixture
def server_port_num():
    return redis_port_num


@pytest.fixture
# TODO: put in conftest.py
def setup_store(server_port_num):
    """Start the server"""
    p = subprocess.Popen(
        [
            "redis-server",
            "--save",
            '""',
            "--port",
            str(server_port_num),
            "--maxmemory",
            str(10000000),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)
    if p.poll() is not None:
        raise Exception("redis-server failed to start")

    yield p

    # kill the subprocess when the caller is done with it
    p.send_signal(signal.SIGINT)
    p.wait(timeout=WAIT_TIMEOUT)


def nex_startup(ports, filename):
    nex = Nexus("test")
    nex.create_nexus(
        file=filename,
        store_size=100000000,
        control_port=ports[0],
        output_port=ports[1],
        actor_in_port=ports[3],
    )
    nex.start_nexus()


@pytest.fixture
def start_nexus_minimal_zmq(ports):
    filename = "minimal.yaml"
    p = multiprocessing.Process(target=nex_startup, args=(ports, filename))
    p.start()
    time.sleep(1)

    yield p

    p.terminate()
    p.join(WAIT_TIMEOUT)
    if p.exitcode is None:
        logging.exception("Timed out waiting for nexus to stop")
        p.kill()


@pytest.fixture
def zmq_actor(ports):
    actor = ZmqActor(ports[3], None, None, None, None, None, name="test")

    p = multiprocessing.Process(target=actor_startup, args=(actor,))

    yield p

    p.terminate()
    p.join(WAIT_TIMEOUT)
    if p.exitcode is None:
        p.kill()


def actor_startup(actor):
    actor.register_with_nexus()


@pytest.fixture
def harvester(ports):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind("tcp://*:1234")
    p = multiprocessing.Process(
        target=bootstrap_harvester,
        args=(
            "localhost",
            ports[3],
            "localhost",
            6379,
            "localhost",
            1234,
            "localhost",
            12345,
        ),
    )
    p.start()
    time.sleep(1)
    yield ports, socket, p
    socket.close(linger=0)
    ctx.destroy(linger=0)


class SignalManager:
    def __init__(self):
        self.signal_handlers = dict()

    def __enter__(self):
        for sig in signals:
            self.signal_handlers[sig] = signal.getsignal(sig)

    def __exit__(self, type, value, traceback):
        for sig, handler in self.signal_handlers.items():
            signal.signal(sig, handler)

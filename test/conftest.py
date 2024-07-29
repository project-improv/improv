import os
import signal
import uuid
import pytest
import subprocess

store_loc = str(os.path.join("/tmp/", str(uuid.uuid4())))
redis_port_num = 6379
WAIT_TIMEOUT = 120


@pytest.fixture
def set_store_loc():
    return store_loc


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

    yield p

    # kill the subprocess when the caller is done with it
    p.send_signal(signal.SIGINT)
    p.wait(WAIT_TIMEOUT)


@pytest.fixture
def setup_plasma_store(set_store_loc, scope="module"):
    """Fixture to set up the store subprocess with 10 mb."""
    p = subprocess.Popen(
        ["plasma_store", "-s", set_store_loc, "-m", str(10000000)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield p
    p.send_signal(signal.SIGINT)
    p.wait(WAIT_TIMEOUT)

import os
import uuid
import pytest
import subprocess

store_loc = str(os.path.join("/tmp/", str(uuid.uuid4())))
redis_port_num = 6379
WAIT_TIMEOUT = 10


@pytest.fixture()
def set_store_loc():
    return store_loc


@pytest.fixture()
def server_port_num():
    return redis_port_num


@pytest.fixture()
# TODO: put in conftest.py
def setup_store(server_port_num):
    """Start the server"""
    print(
        f"Setting up Redis store."
        f"Store on port {server_port_num}"
        f"with save '"
        "' "
    )
    p = subprocess.Popen(
        [
            "redis-server",
            "--save",
            '""',
            "--port",
            str(server_port_num),
            "--maxmemory",
            str(10000000),
            "--dbfilename",
            "CI_test_nonexistent.rdb",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    yield p

    # kill the subprocess when the caller is done with it
    p.kill()
    p.wait(WAIT_TIMEOUT)

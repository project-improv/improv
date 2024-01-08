import os
import uuid
import pytest

store_loc = str(os.path.join("/tmp/", str(uuid.uuid4())))
redis_port_num = 6379


@pytest.fixture()
def set_store_loc():
    return store_loc


@pytest.fixture()
def server_port_num():
    return redis_port_num

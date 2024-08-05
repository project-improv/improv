import pytest

from improv.store import StoreInterface, RedisStoreInterface, ObjectNotFoundError

from scipy.sparse import csc_matrix
import numpy as np
import redis
import logging

from improv.store import CannotConnectToStoreInterfaceError

WAIT_TIMEOUT = 10

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_connect(setup_store, server_port_num):
    store = StoreInterface(server_port_num=server_port_num)
    assert isinstance(store.client, redis.Redis)


def test_redis_connect(setup_store, server_port_num):
    store = RedisStoreInterface(server_port_num=server_port_num)
    assert store.client.ping()


def test_redis_connect_wrong_port(server_port_num):
    bad_port_num = 1234
    with pytest.raises(CannotConnectToStoreInterfaceError) as e:
        RedisStoreInterface(server_port_num=bad_port_num)
    assert e.value.message == "Cannot connect to store at {}".format(str(bad_port_num))


# TODO: @pytest.parameterize...store.get and store.getID for diff datatypes,
def test_init_empty(setup_store, server_port_num):
    store = StoreInterface(server_port_num=server_port_num)
    # logger.info(store.client.config_get())
    assert store.get_all() == []


def test_is_csc_matrix_and_put(setup_store, server_port_num):
    mat = csc_matrix((3, 4), dtype=np.int8)
    store = StoreInterface(server_port_num=server_port_num)
    x = store.put(mat)
    assert isinstance(store.get(x), csc_matrix)


def test_get_list_and_all(setup_store, server_port_num):
    store = StoreInterface(server_port_num=server_port_num)
    id = store.put(1)
    id2 = store.put(2)
    store.put(3)
    assert [1, 2] == store.get_list([id, id2])
    assert [1, 2, 3] == sorted(store.get_all())


def test_reset(setup_store, server_port_num):
    store = StoreInterface(server_port_num=server_port_num)
    store.reset()
    id = store.put(1)
    assert store.get(id) == 1


def test_put_one(setup_store, server_port_num):
    store = StoreInterface(server_port_num=server_port_num)
    id = store.put(1)
    assert 1 == store.get(id)


def test_redis_put_one(setup_store, server_port_num):
    store = RedisStoreInterface(server_port_num=server_port_num)
    key = store.put(1)
    assert 1 == store.get(key)


def test_redis_get_unknown_object(setup_store, server_port_num):
    store = RedisStoreInterface(server_port_num=server_port_num)
    with pytest.raises(ObjectNotFoundError):
        store.get("unknown")


def test_redis_get_one(setup_store, server_port_num):
    store = RedisStoreInterface(server_port_num=server_port_num)
    key = store.put(3)
    assert 3 == store.get(key)

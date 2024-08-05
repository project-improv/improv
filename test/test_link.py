import json
import time

import pytest
import zmq
from improv.actor import Actor
from zmq import SocketOption

from improv.link import ZmqLink


@pytest.fixture
def test_sub_link():
    """Fixture to provide a commonly used Link object."""
    ctx = zmq.Context()
    link_socket = ctx.socket(zmq.SUB)
    link_socket.bind("tcp://*:0")
    link_socket_string = link_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
    link_socket_port = int(link_socket_string.split(":")[-1])

    link_socket.poll(timeout=0)

    link_pub_socket = ctx.socket(zmq.PUB)
    link_pub_socket.connect(f"tcp://localhost:{link_socket_port}")

    link_socket.poll(timeout=0)  # open libzmq bug (not pyzmq)
    link_socket.subscribe("test_topic")
    time.sleep(0.5)
    link_socket.poll(timeout=0)

    link = ZmqLink(link_socket, "test_link", "test_topic")

    yield link, link_pub_socket
    link.socket.close(linger=0)
    link_pub_socket.close(linger=0)
    ctx.destroy(linger=0)


@pytest.fixture
def test_pub_link():
    """Fixture to provide a commonly used Link object."""
    ctx = zmq.Context()
    link_socket = ctx.socket(zmq.PUB)
    link_socket.bind("tcp://*:0")
    link_socket_string = link_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
    link_socket_port = int(link_socket_string.split(":")[-1])

    link_sub_socket = ctx.socket(zmq.SUB)
    link_sub_socket.connect(f"tcp://localhost:{link_socket_port}")
    link_sub_socket.poll(timeout=0)
    link_sub_socket.subscribe("test_topic")
    time.sleep(0.5)
    link_sub_socket.poll(timeout=0)

    link = ZmqLink(link_socket, "test_link", "test_topic")

    yield link, link_sub_socket
    link.socket.close(linger=0)
    link_sub_socket.close(linger=0)
    ctx.destroy(linger=0)


@pytest.fixture
def test_req_link():
    """Fixture to provide a commonly used Link object."""
    ctx = zmq.Context()
    link_socket = ctx.socket(zmq.REQ)
    link_socket.bind("tcp://*:0")
    link_socket_string = link_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
    link_socket_port = int(link_socket_string.split(":")[-1])

    link_rep_socket = ctx.socket(zmq.REP)
    link_rep_socket.connect(f"tcp://localhost:{link_socket_port}")

    link = ZmqLink(link_socket, "test_link")

    yield link, link_rep_socket
    link.socket.close(linger=0)
    link_rep_socket.close(linger=0)
    ctx.destroy(linger=0)


@pytest.fixture
def test_rep_link():
    """Fixture to provide a commonly used Link object."""
    ctx = zmq.Context()
    link_socket = ctx.socket(zmq.REP)
    link_socket.bind("tcp://*:0")
    link_socket_string = link_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
    link_socket_port = int(link_socket_string.split(":")[-1])

    link_req_socket = ctx.socket(zmq.REQ)
    link_req_socket.connect(f"tcp://localhost:{link_socket_port}")

    link = ZmqLink(link_socket, "test_link")

    yield link, link_req_socket
    link.socket.close(linger=0)
    link_req_socket.close(linger=0)
    ctx.destroy(linger=0)


def test_pub_put(test_pub_link):
    """Tests if messages can be put into the link.

    TODO:
        Parametrize multiple test input types.
    """

    link, link_sub_socket = test_pub_link
    msg = "message"

    link.put(msg)
    res = link_sub_socket.recv_multipart()
    assert res[0].decode("utf-8") == "test_topic"
    assert json.loads(res[1].decode("utf-8")) == msg


def test_req_put(test_req_link):
    """Tests if messages can be put into the link.

    TODO:
        Parametrize multiple test input types.
    """

    link, link_rep_socket = test_req_link
    msg = "message"

    link.put(msg)
    res = link_rep_socket.recv_pyobj()
    assert res == msg


def test_put_unserializable(test_pub_link):
    """Tests if an unserializable object raises an error.

    Instantiates an actor, which is unserializable, and passes it into
    Link.put().

    Raises:
        SerializationCallbackError: Actor objects are unserializable.
    """
    act = Actor("test", "/tmp/store")
    link, link_sub_socket = test_pub_link
    sentinel = True
    try:
        link.put(act)
    except Exception:
        sentinel = False

    assert not sentinel


def test_put_irreducible(test_pub_link, setup_store):
    """Tests if an irreducible object raises an error."""

    link, link_sub_socket = test_pub_link
    store = setup_store
    with pytest.raises(TypeError):
        link.put(store)


def test_put_nowait(test_pub_link):
    """Tests if messages can be put into the link without blocking.

    TODO:
        Parametrize multiple test input types.
    """

    link, link_sub_socket = test_pub_link
    msg = "message"

    t_0 = time.perf_counter()

    link.put(msg)  # put is already async even in synchronous zmq

    t_1 = time.perf_counter()
    t_net = t_1 - t_0
    assert t_net < 0.005  # 5 ms


def test_put_multiple(test_pub_link):
    """Tests if async putting multiple objects preserves their order."""

    link, link_sub_socket = test_pub_link

    messages = [str(i) for i in range(10)]

    messages_out = []

    for msg in messages:
        link.put(msg)

    for msg in messages:
        messages_out.append(
            json.loads(link_sub_socket.recv_multipart()[1].decode("utf-8"))
        )

    assert messages_out == messages


@pytest.mark.asyncio
async def test_put_and_get_async(test_pub_link):
    """Tests if async get preserves order after async put."""

    messages = [str(i) for i in range(10)]

    messages_out = []

    link, link_sub_socket = test_pub_link

    for msg in messages:
        await link.put_async(msg)

    for msg in messages:
        messages_out.append(
            json.loads(link_sub_socket.recv_multipart()[1].decode("utf-8"))
        )

    assert messages_out == messages


@pytest.mark.parametrize(
    "message",
    [
        "message",
        "",
        None,
        [str(i) for i in range(5)],
    ],
)
@pytest.mark.parametrize(
    "timeout",
    [
        None,
        5,
    ],
)
def test_sub_get(test_sub_link, message, timeout):
    """Tests if get gets the correct element from the queue."""

    link, link_pub_socket = test_sub_link

    time.sleep(1)

    if type(message) is list:
        for i in message:
            link_pub_socket.send_multipart(
                [link.topic.encode("utf-8"), json.dumps(i).encode("utf-8")]
            )
        expected = message[0]
    else:
        link_pub_socket.send_multipart(
            [link.topic.encode("utf-8"), json.dumps(message).encode("utf-8")]
        )
        expected = message

    assert link.get(timeout=timeout) == expected


@pytest.mark.parametrize(
    "message",
    [
        "message",
        "",
        None,
        [str(i) for i in range(5)],
    ],
)
@pytest.mark.parametrize(
    "timeout",
    [
        None,
        5,
    ],
)
def test_rep_get(test_rep_link, message, timeout):
    """Tests if get gets the correct element from the queue."""

    link, link_req_socket = test_rep_link

    link_req_socket.send_pyobj(message)

    expected = message

    assert link.get(timeout=timeout) == expected


def test_get_empty(test_sub_link):
    """Tests if get blocks if the queue is empty."""

    link, unused = test_sub_link

    time.sleep(0.1)

    with pytest.raises(TimeoutError):
        link.get(timeout=0.5)


@pytest.mark.parametrize(
    "message",
    [
        "message",
        "",
        ([str(i) for i in range(5)]),
    ],
)
def test_get_nowait(test_sub_link, message):
    """Tests if get_nowait gets the correct element from the queue."""

    link, link_pub_socket = test_sub_link

    if type(message) is list:
        for i in message:
            link_pub_socket.send_multipart(
                [link.topic.encode("utf-8"), json.dumps(i).encode("utf-8")]
            )
        expected = message[0]
    else:
        link_pub_socket.send_multipart(
            [link.topic.encode("utf-8"), json.dumps(message).encode("utf-8")]
        )
        expected = message

    for i in range(20):
        time.sleep(0.1)
        avail = link.socket.poll(timeout=100)
        if avail:
            break
    else:
        pytest.fail("Message was not sent to link after 4s")

    t_0 = time.perf_counter()

    res = link.get_nowait()

    t_1 = time.perf_counter()

    assert res == expected
    assert t_1 - t_0 < 0.005  # 5 msg


def test_get_nowait_empty(test_sub_link):
    """Tests if get_nowait raises an error when the queue is empty."""

    link, unused = test_sub_link
    with pytest.raises(TimeoutError):
        link.get_nowait()


@pytest.mark.asyncio
async def test_get_async_success(test_sub_link):
    """Tests if async_get gets the correct element from the queue."""

    link, link_pub_socket = test_sub_link
    msg = "message"
    link_pub_socket.send_multipart(
        [link.topic.encode("utf-8"), json.dumps(msg).encode("utf-8")]
    )
    res = await link.get_async()
    assert res == "message"


def test_pub_put_no_topic():
    ctx = zmq.Context()
    s = ctx.socket(zmq.PUB)
    with pytest.raises(Exception, match="Cannot open PUB link without topic"):
        ZmqLink(s, "test")
    s.close(linger=0)
    ctx.destroy(linger=0)

import asyncio
import queue
import subprocess
import time

import pytest

from improv.actor import Actor

from improv.store import StoreInterface
from improv.link import Link


@pytest.fixture
def setup_store():
    """Fixture to set up the store subprocess with 10 mb.

    This fixture runs a subprocess that instantiates the store with a
    memory of 10 megabytes. It specifies that "/tmp/store/" is the
    location of the store socket.

    Yields:
        store: An instance of the store.

    TODO:
        Figure out the scope.
    """

    p = subprocess.Popen(
        ["plasma_store", "-s", "/tmp/store", "-m", str(10000000)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    store = StoreInterface(store_loc="/tmp/store")
    yield store
    p.kill()
    p.wait()


def init_actors(n=1):
    """Function to return n unique actors.

    Returns:
        list: A list of n actors, each being named after its index.
    """

    # the links must be specified as an empty dictionary to avoid
    # actors sharing a dictionary of links

    return [Actor("test " + str(i), "/tmp/store", links={}) for i in range(n)]


@pytest.fixture
def example_link(setup_store):
    """Fixture to provide a commonly used Link object."""
    setup_store
    act = init_actors(2)
    lnk = Link("Example", act[0].name, act[1].name)
    yield lnk
    lnk = None


@pytest.fixture
def example_actor_system(setup_store):
    """Fixture to provide a list of 4 connected actors."""

    # store = setup_store
    acts = init_actors(4)

    L01 = Link("L01", acts[0].name, acts[1].name)
    L13 = Link("L13", acts[1].name, acts[3].name)
    L12 = Link("L12", acts[1].name, acts[2].name)
    L23 = Link("L23", acts[2].name, acts[3].name)

    links = [L01, L13, L12, L23]

    acts[0].addLink("q_out_1", L01)
    acts[1].addLink("q_out_1", L13)
    acts[1].addLink("q_out_2", L12)
    acts[2].addLink("q_out_1", L23)

    acts[1].addLink("q_in_1", L01)
    acts[2].addLink("q_in_1", L12)
    acts[3].addLink("q_in_1", L13)
    acts[3].addLink("q_in_2", L23)

    yield [acts, links]  # also yield Links
    acts = None


@pytest.fixture
def _kill_pytest_processes():
    """Kills all processes with "pytest" in their name.

    NOTE:
        This fixture should only be used at the end of testing.
    """

    subprocess.Popen(
        ["kill", "`pgrep pytest`"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("name", "Example"),
        ("real_executor", None),
        ("cancelled_join", False),
        ("status", "pending"),
        ("result", None),
    ],
)
def test_Link_init(setup_store, example_link, attribute, expected):
    """Tests if the default initialization attributes are set."""

    lnk = example_link
    atr = getattr(lnk, attribute)
    assert atr == expected


def test_Link_init_start_end(setup_store):
    """Tests if the initialization has the right actors."""

    act = init_actors(2)
    lnk = Link("example_link", act[0].name, act[1].name)

    assert lnk.start == act[0].name
    assert lnk.end == act[1].name


def test_getstate(example_link):
    """Tests if __getstate__ has the right values on initialization.

    Gets the dictionary of the link, then compares them against known
    default values. Does not compare store and actors.

    TODO:
        Compare store and actors.
    """

    res = example_link.__getstate__()
    errors = []
    errors.append(res["_real_executor"] is None)
    errors.append(res["cancelled_join"] is False)

    assert all(errors)


@pytest.mark.parametrize(
    "input",
    [([None]), ([1]), ([i for i in range(5)]), ([str(i**i) for i in range(10)])],
)
def test_qsize_empty(example_link, input):
    """Tests that the queue has the number of elements in "input"."""

    lnk = example_link
    for i in input:
        lnk.put(i)

    qsize = lnk.queue.qsize()
    assert qsize == len(input)


def test_getStart(example_link):
    """Tests if getStart returns the starting actor."""

    lnk = example_link

    assert lnk.getStart() == Actor("test 0", "/tmp/store").name


def test_getEnd(example_link):
    """Tests if getEnd returns the ending actor."""

    lnk = example_link

    assert lnk.getEnd() == Actor("test 1", "/tmp/store").name


def test_put(example_link):
    """Tests if messages can be put into the link.

    TODO:
        Parametrize multiple test input types.
    """

    lnk = example_link
    msg = "message"

    lnk.put(msg)
    assert lnk.get() == "message"


def test_put_unserializable(example_link, caplog, setup_store):
    """Tests if an unserializable object raises an error.

    Instantiates an actor, which is unserializable, and passes it into
    Link.put().

    Raises:
        SerializationCallbackError: Actor objects are unserializable.
    """
    # store = setup_store
    act = Actor("test", "/tmp/store")
    lnk = example_link
    sentinel = True
    try:
        lnk.put(act)
    except Exception:
        sentinel = False

    assert sentinel, "Unable to put"
    assert str(lnk.get()) == str(act)


def test_put_irreducible(example_link, setup_store):
    """Tests if an irreducible object raises an error."""

    lnk = example_link
    store = setup_store
    with pytest.raises(TypeError):
        lnk.put(store)


def test_put_nowait(example_link):
    """Tests if messages can be put into the link without blocking.

    TODO:
        Parametrize multiple test input types.
    """

    lnk = example_link
    msg = "message"

    t_0 = time.perf_counter()

    lnk.put_nowait(msg)

    t_1 = time.perf_counter()
    t_net = t_1 - t_0
    assert t_net < 0.005  # 5 ms


@pytest.mark.asyncio
async def test_put_async_success(example_link):
    """Tests if put_async returns None.

    TODO:
        Parametrize test input.
    """

    lnk = example_link
    msg = "message"
    res = await lnk.put_async(msg)
    assert res is None


@pytest.mark.asyncio
async def test_put_async_multiple(example_link):
    """Tests if async putting multiple objects preserves their order."""

    messages = [str(i) for i in range(10)]

    messages_out = []

    for msg in messages:
        await example_link.put_async(msg)

    for msg in messages:
        messages_out.append(example_link.get())

    assert messages_out == messages


@pytest.mark.asyncio
async def test_put_and_get_async(example_link):
    """Tests if async get preserves order after async put."""

    messages = [str(i) for i in range(10)]

    messages_out = []

    for msg in messages:
        await example_link.put_async(msg)

    for msg in messages:
        messages_out.append(await example_link.get_async())

    assert messages_out == messages


def test_put_overflow(setup_store, caplog):
    """Tests if putting too large of an object raises an error."""

    p = subprocess.Popen(
        ["plasma_store", "-s", "/tmp/store", "-m", str(1000)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    StoreInterface(store_loc="/tmp/store")

    acts = init_actors(2)
    lnk = Link("L1", acts[0], acts[1])

    message = [i for i in range(10**6)]  # 24000 bytes

    lnk.put(message)

    p.kill()
    p.wait()
    setup_store  # restore the 10 mb store

    if caplog.records:
        for record in caplog.records:
            if "PlasmaStoreInterfaceFull" in record.msg:
                assert True
    else:
        pytest.fail("expected an error!")


@pytest.mark.parametrize(
    "message",
    [
        ("message"),
        (""),
        (None),
        ([str(i) for i in range(5)]),
    ],
)
def test_get(example_link, message):
    """Tests if get gets the correct element from the queue."""

    lnk = example_link

    if type(message) is list:
        for i in message:
            lnk.put(i)
        expected = message[0]
    else:
        lnk.put(message)
        expected = message

    assert lnk.get() == expected


def test_get_empty(example_link):
    """Tests if get blocks if the queue is empty."""

    lnk = example_link
    if lnk.queue.empty:
        with pytest.raises(queue.Empty):
            lnk.get(timeout=5.0)
    else:
        pytest.fail("expected a timeout!")


@pytest.mark.parametrize(
    "message",
    [
        ("message"),
        (""),
        ([str(i) for i in range(5)]),
    ],
)
def test_get_nowait(example_link, message):
    """Tests if get_nowait gets the correct element from the queue."""

    lnk = example_link

    if type(message) is list:
        for i in message:
            lnk.put(i)
        expected = message[0]
    else:
        lnk.put(message)
        expected = message

    t_0 = time.perf_counter()

    res = lnk.get_nowait()

    t_1 = time.perf_counter()

    assert res == expected
    assert t_1 - t_0 < 0.005  # 5 msg


def test_get_nowait_empty(example_link):
    """Tests if get_nowait raises an error when the queue is empty."""

    lnk = example_link
    if lnk.queue.empty():
        with pytest.raises(queue.Empty):
            lnk.get_nowait()
    else:
        pytest.fail("the queue is not empty")


@pytest.mark.asyncio
async def test_get_async_success(example_link):
    """Tests if async_get gets the correct element from the queue."""

    lnk = example_link
    msg = "message"
    await lnk.put_async(msg)
    res = await lnk.get_async()
    assert res == "message"


@pytest.mark.asyncio
async def test_get_async_empty(example_link):
    """Tests if get_async times out given an empty queue.

    TODO:
        Implement a way to kill the task after execution (subprocess)?
    """

    lnk = example_link
    timeout = 5.0

    with pytest.raises(asyncio.TimeoutError):
        task = asyncio.create_task(lnk.get_async())
        await asyncio.wait_for(task, timeout)
        task.cancel()

    lnk.put("exit")  # this is here to break out of get_async()


@pytest.mark.skip(reason="unfinished")
def test_cancel_join_thread(example_link):
    """Tests cancel_join_thread. This test is unfinished

    TODO:
        Identify where and when cancel_join_thread is being called.
    """

    lnk = example_link
    lnk.cancel_join_thread()

    assert lnk._cancelled_join is True


@pytest.mark.skip(reason="unfinished")
@pytest.mark.asyncio
async def test_join_thread(example_link):
    """Tests join_thread. This test is unfinished

    TODO:
        Identify where and when join_thread is being called.
    """
    lnk = example_link
    await lnk.put_async("message")
    # msg = await lnk.get_async()
    lnk.join_thread()
    assert True


@pytest.mark.asyncio
async def test_multi_actor_system(example_actor_system, setup_store):
    """Tests if async puts/gets with many actors have good messages."""

    setup_store

    graph = example_actor_system

    acts = graph[0]

    heavy_msg = [str(i) for i in range(10**6)]
    light_msgs = ["message" + str(i) for i in range(3)]

    await acts[0].links["q_out_1"].put_async(heavy_msg)
    await acts[1].links["q_out_1"].put_async(light_msgs[0])
    await acts[1].links["q_out_2"].put_async(light_msgs[1])
    await acts[2].links["q_out_1"].put_async(light_msgs[2])

    assert await acts[1].links["q_in_1"].get_async() == heavy_msg
    assert await acts[2].links["q_in_1"].get_async() == light_msgs[1]
    assert await acts[3].links["q_in_1"].get_async() == light_msgs[0]
    assert await acts[3].links["q_in_2"].get_async() == light_msgs[2]

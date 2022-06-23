import pytest
import subprocess
import queue
import asyncio
import signal
import logging
import concurrent.futures

from async_timeout import timeout

from improv.link import Link
from improv.link import AsyncQueue
from improv.store import Limbo
from improv.actor import Actor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s %(message)s',
                    handlers=[logging.FileHandler("global.log"),
                              logging.StreamHandler()])

def with_timeout(t):
    def wrapper(corofunc):
        async def run(*args, **kwargs):
            with timeout(t):
                return await corofunc(*args, **kwargs)
        return run       
    return wrapper


class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

@pytest.fixture
def setup_store():
    """ Fixture to set up the store subprocess with 10 mb.

    TODO:
        Figure out the scope.
    """

    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lmb = Limbo(store_loc = "/tmp/store")
    yield lmb
    p.kill()


def init_actors(n: "int"=1) -> "list":
    """ Function to return n unique actors.
    """

    actors_out = []
    actor_num = 1
    for i in range(n):
        act = Actor("test " + str(actor_num))
        actors_out.append(act)
        actor_num += 1

    return actors_out


@pytest.fixture
def example_link(setup_store):
    act = init_actors(2)
    lnk = Link("Example", act[0], act[1], setup_store)
    yield lnk
    lnk = None


@pytest.mark.parametrize("attribute, expected",[
    ("name", "Example"),
    ("real_executor", None),
    ("cancelled_join", False),
    ("status", "pending"),
    ("result", None),
    ("num", 0)
])
def test_Link_init(setup_store, example_link, attribute, expected):
    """ Tests if the default initialization attributes are set.
    """

    lnk = example_link
    atr = getattr(lnk, attribute) 
    assert atr == expected


def test_Link_init_start_end(setup_store):
    """ Tests if the initialization has the right actors.
    """

    act = init_actors(2)
    lnk = Link("example_link", act[0], act[1], setup_store)

    assert lnk.start == act[0] and lnk.end == act[1]


def test_Link_init_limbo(setup_store, example_link):
    """ Tests if the initialization has the right store.
    """

    assert example_link.limbo == setup_store


@pytest.mark.parametrize("input",[
    ([None]),
    ([1]),
    ([i for i in range(5)]),
    ([str(i ** i) for i in range(10)]),
    ([Actor("test " + str(i)) for i in range(3)])
])
def test_qsize_empty(example_link, input):
    """ Checks that the queue has the number of elements in "input".
    """

    lnk = example_link
    [lnk.put(i) for i in input]
    qsize = lnk.queue.qsize()
    assert qsize == len(input) 


def test_getStart(example_link):
    lnk = example_link

    assert str(lnk.getStart()) == str(Actor("test 1"))


def test_getEnd(example_link):
    lnk = example_link

    assert str(lnk.getEnd()) == str(Actor("test 2"))


def test_put(example_link):
    """ Tests if messages can be put into the link.

    TODO:
        Parametrize multiple test inputs.
    """

    lnk = example_link
    msg = "message"

    lnk.put(msg)
    assert lnk.get() == "message"


def test_put_nowait(example_link):
    """ Tests if messages can be put into the link without blocking.

    TODO:
        Parametrize multiple test inputs.
    """

    lnk = example_link
    msg = "message"

    lnk.put_nowait(msg)
    assert lnk.get() == "message"


@pytest.mark.asyncio
async def test_put_async_success(example_link):
    """ Tests if put_async returns None.

    TODO:
        Parametrize test input.
    """

    lnk = example_link
    msg = "message"
    res = await lnk.put_async(msg)
    assert res == None


@pytest.mark.parametrize("message", [
    ("message"),
    (""),
    (None),
    ([str(i) for i in range(5)]),
])
def test_get(example_link, message):
    """ Tests if get gets the correct element from the queue.
    """

    lnk = example_link

    if type(message) == list:
        for i in message:
            lnk.put(i)
        expected = message[0]
    else:
        lnk.put(message)
        expected = message

    assert lnk.get() == expected


@pytest.mark.parametrize("message", [
    ("message"),
    (""),
    ([str(i) for i in range(5)]),
])
def test_get_nowait(example_link, message):
    """ Tests if get_nowait gets the correct element from the queue.
    """

    lnk = example_link

    if type(message) == list:
        for i in message:
            lnk.put(i)
        expected = message[0]
    else:
        lnk.put(message)
        expected = message

    assert lnk.get_nowait() == expected

def test_get_nowait_empty(example_link):
    """ Tests if get_nowait raises an error when the queue is empty.
    """
    
    lnk = example_link
    if lnk.queue.empty():
        with pytest.raises(queue.Empty):
            lnk.get_nowait()
    else:
        assert False, "the queue is not empty"


@pytest.mark.asyncio
async def test_get_async_success(example_link):
    """ Tests if async_get gets the correct element from the queue.
    """

    lnk = example_link
    msg = "message"
    await lnk.put_async(msg)    
    res = await lnk.get_async()
    assert res == "message"


@pytest.mark.asyncio
async def test_get_async_empty(example_link):
    """ Tests if get_async times out given an empty queue.

    TODO:
        Implement a way to kill the task after execution (subprocess)?
    """

    lnk = example_link
    timeout = 5.0

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lnk.get_async(), timeout)
    

@pytest.mark.skip(reason="unfinished")
def test_cancel_join_thread(example_link):
    lnk = example_link
    lnk.cancel_join_thread()

    assert lnk._cancelled_join == True


@pytest.mark.skip(reason="unfinished")
@pytest.mark.asyncio
async def test_join_thread(example_link):
    lnk = example_link
    await lnk.put_async("message")
    msg = await lnk.get_async()
    lnk.join_thread()
    assert True


def test_log_to_limbo_success(example_link):
    lnk = example_link
    msg = "message"
    lnk.log_to_limbo(msg)
    res = lnk.limbo.get(f"q__{lnk.start}__{lnk.num - 1}")
    assert res == "message"
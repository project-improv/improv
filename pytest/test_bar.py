import asyncio
import queue
import subprocess
import time

import pyarrow
import pytest
import concurrent.futures
from async_timeout import timeout
import logging; logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from improv.actor import Actor
from improv.link import AsyncQueue
from improv.store import Limbo
from improv.link import Link

@pytest.fixture
def setup_store():
    """ Fixture to set up the store subprocess with 10 mb.

    This fixture runs a subprocess that instantiates the store with a 
    memory of 10 megabytes. It specifies that "/tmp/store/" is the 
    location of the store socket.

    Yields:
        Limbo: An instance of the store.

    TODO:
        Figure out the scope.
    """

    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lmb = Limbo(store_loc = "/tmp/store")
    yield lmb
    p.kill()

def init_actors(n = 1):
    """ Function to return n unique actors.

    Returns:
        list: A list of n actors, each being named after its index.
    """

    #the links must be specified as an empty dictionary to avoid
    #actors sharing a dictionary of links

    return [Actor("test " + str(i), links={}) for i in range(n)]


@pytest.fixture
def example_link(setup_store):
    """ Fixture to provide a commonly used Link object.
    """

    act = init_actors(2)
    lnk = Link("Example", act[0].name, act[1].name, setup_store)
    yield lnk
    lnk = None












@pytest.mark.asyncio
async def test_get_async_empty(example_link):
    """ Tests if get_async times out given an empty queue.

    TODO:
        Implement a way to kill the task after execution (subprocess)?
    """

    lnk = example_link
    timeout = 1.0


    


    loop = asyncio.get_event_loop()
    with pytest.raises(asyncio.TimeoutError):
        task = asyncio.create_task(lnk.get_async())
        res = await asyncio.wait_for(task, timeout)
        loop.stop()
        loop.close()


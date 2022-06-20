import pytest
import subprocess

from improv.link import Link
from improv.link import AsyncQueue
from improv.store import Limbo
from improv.actor import Actor

@pytest.fixture
def setup_store():
    """ Fixture to set up the store subprocess.

    TODO:
        Figure out the scope.
    """

    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lmb = Limbo(store_loc = "/tmp/store")
    yield lmb
    p.kill()

def init_actors(n=1):
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

def test_Link_init(setup_store, example_link):
    """ Tests if all the initialization attributes are set.

    TODO:
        Check for the attributes in AsyncQueue.
    """

    lnk = example_link

    name_check = (lnk.name == "Example")

    assert name_check

def test_getStart(example_link):
    lnk = example_link

    assert str(lnk.getStart()) == str(Actor("test 1"))

def test_getEnd(example_link):
    lnk = example_link

    assert str(lnk.getEnd()) == str(Actor("test 2"))

@pytest.mark.skip(reason = "This test is unfinished.")
def test_put():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_put_nowait():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_put_async():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_get_async():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_cancel_join_thread():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_join_thread():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_log_to_limbo():
    assert True

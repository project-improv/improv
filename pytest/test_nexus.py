import os
import pytest
import subprocess

from improv.nexus import Nexus
from improv.link import Link
from improv.actor import Actor
from improv.store import Limbo


os.chdir(os.getcwd() + '/./configs')

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

def test_init():
    nex = Nexus("test")
    assert nex.name == "test"

@pytest.mark.skip
def test_createNexus():
    nex = Nexus("test")
    nex.createNexus()
    assert nex.comm_queues == {} 
    assert nex.sig_queues == {}
    assert nex.data_queues == {}
    assert nex.actors == {}
    assert nex.flags == {'quit': False, 'run': False, 'load': False}
    assert nex.processes == []

@pytest.mark.skip
def test_loadTweak():
    nex = Nexus("test")
    nex.createNexus()
    nex.loadTweak()
    assert True
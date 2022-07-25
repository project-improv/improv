import os
import pytest
import subprocess

from improv.nexus import Nexus
from improv.link import Link
from improv.actor import Actor
from improv.store import Limbo

@pytest.fixture
def setdir():
    os.chdir(os.getcwd() + '/./configs')
    yield None
    os.chdir(os.getcwd() + "/../")

@pytest.fixture
def sample_nex(setdir):
    setdir
    nex = Nexus("test")
    nex.createNexus()
    return nex

# @pytest.fixture
# def setup_store(setdir):
#     """ Fixture to set up the store subprocess with 10 mb.

#     This fixture runs a subprocess that instantiates the store with a 
#     memory of 10 megabytes. It specifies that "/tmp/store/" is the 
#     location of the store socket.

#     Yields:
#         Limbo: An instance of the store.

#     TODO:
#         Figure out the scope.
#     """
#     setdir
#     p = subprocess.Popen(
#         ['plasma_store', '-s', '/tmp/store/', '-m', str(10000000)],\
#         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     lmb = Limbo(store_loc = "/tmp/store/")
#     yield lmb
#     p.kill()

def test_init(setdir):
    setdir
    # lmb = setup_store
    nex = Nexus("test")
    assert str(nex) == "test"

def test_createNexus(setdir):
    setdir
    nex = Nexus("test")
    nex.createNexus(file = "basic_demo.yaml")
    assert list(nex.comm_queues.keys()) == ["GUI_comm", "Acquirer_comm", "Processor_comm", "Analysis_comm", "InputStim_comm"]
    assert list(nex.sig_queues.keys()) == ["Acquirer_sig", "Processor_sig", "Analysis_sig", "InputStim_sig"]
    assert list(nex.data_queues.keys()) == ["Acquirer.q_out", "Processor.q_in", "Processor.q_out", "Analysis.q_in", "InputStim.q_out", "Analysis.input_stim_queue"]
    assert list(nex.actors.keys()) == ["Acquirer", "Processor", "Analysis", "InputStim"]
    assert list(nex.flags.keys()) == ["quit", "run", "load"] 
    assert nex.processes == []

def test_loadTweak(sample_nex):
    nex = sample_nex 
    nex.loadTweak()
    assert set(nex.comm_queues.keys()) == set(["Acquirer_comm", "Analysis_comm", "GUI_comm", "InputStim_comm", "Processor_comm"])

# @pytest.mark.skip("this is unfinished")
def test_startNexus(sample_nex):
    nex = sample_nex
    nex.startNexus()
    assert [p.name for p in nex.processes] == ["Acquirer", "Processor", "Analysis", "InputStim"]

@pytest.mark.skip(reason = "This test is unfinished.")
def test_basic_demo_config():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_good_config_config():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_simple_graph():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_complex_graph():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_single_actor_graph():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_hasGUI_True():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_hasGUI_False():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_queue_message():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_queue_readin():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_queue_sendout():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_run_sig():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_setup_sig():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_quit_sig():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_usehdd_True():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_usehdd_False():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_startstore():
    assert True

@pytest.mark.skip(reason = "This test is unfinished.")
def test_closestore():
    assert True

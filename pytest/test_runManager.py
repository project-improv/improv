import os
import psutil
import pytest
import subprocess
from improv.link import Link, AsyncQueue
from improv.actor import AbstractActor
from improv.actor import RunManager
from improv.store import Store 
from actors import sample_generator, sample_processor




@pytest.fixture
def init_RM():
    RM = RunManager("Test", sample_generator.runStep, sample_generator.links)

@pytest.mark.skip(reason="unfinished")
def test_RM_init(init_rm):
    RM = init_rm
    assert self.run == False
    assert self.run == False
    assert selfconfig == False

    assert self.actorName == name

    assert self.actions is sample_generator.runStep
    assert self.links is sample_generator.links

    assert self.timeout == 1e-6


@pytest.mark.skip(reason="unfinished")
def test_RM_run(init_rm):
    act = Actor("test")
    with init_rm as RM:
        logger.info(rm)

    #what to assert here?
    #assert                     

    pass 

@pytest.mark.skip(reason="unfinished")
def test_RM_stop_after_run(init_rm):
    with init_rm as RM:
        logger.info(rm)

    rm.q_sig_in.append("stop")
    assert True #how to assert that this RM has actuallly stopped?

@pytest.mark.skip(reason="unfinished")
def test_RM_config(init_rm):
    pass

@pytest.mark.skip(reason="unfinished")
def test_RM_stop_after_config(init_rm):
    pass

@pytest.mark.skip(reason="unfinished")
def test_RM_signals(init_rm):

    with init_rm as RM:
        logger.info(RM)

    RM.q_sig_in.append("setup")
    assert True
    RM.q_sig_in.append("run")
    assert True
    RM.q_sig_in.append("stop")
    assert True
    RM.q_sig_in.append("quit")
    assert True

@pytest.mark.skip(reason="unfinished")
def test_RM_exit(init_rm):
    pass



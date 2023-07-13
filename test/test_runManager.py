import time
import os
import psutil
import pytest
import subprocess
from improv.link import Link, AsyncQueue
from improv.actor import AbstractActor
from improv.actor import RunManager
from improv.actor import Signal
from improv.store import StoreInterface
from actors.sample_generator import Generator
from actors.sample_processor import Processor




@pytest.fixture
def init_rm():
    samp_gen = Generator("Test")
    samp_proc = Processor("Test")
    link_dict = {
        'q_sig': Link("q_sig", samp_gen, samp_proc),
        'q_comm': Link("q_comm", samp_gen, samp_proc),
        'q_in': Link("q_in", samp_gen, samp_proc),
        'q_out': Link("q_out", samp_gen, samp_proc)
            }
    samp_gen.setLinks(link_dict)
    RM = RunManager("Test", samp_gen.runStep, samp_gen.links)
    yield [RM, samp_gen, samp_proc]

@pytest.mark.skip(reason="unfinished")
def test_RM_init(init_rm):
    [RM, gen, proc] = init_rm
    
    assert RM.run == False
    assert RM.config == False
    assert RM.actorName == "Test"
    assert RM.links is gen.links
    assert RM.timeout == 1e-6



@pytest.mark.skip(reason="unfinished")
def test_RM_run(init_rm):
    [RM, gen, proc] = init_rm

    #what to assert here?
    #assert                     
    

    RM.q_sig.put(Signal.setup())
    for i in range(100):
        RM.q_sig.put(Signal.run())
    RM.q_sig.put(Signal.stop())
    RM.q_sig.put(Signal.quit())
    with RM:
        print(RM)
    assert RM.run == False
    assert RM.stop == False

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



import pytest
import os
import yaml
from inspect import signature
from importlib import import_module
from improv.tweak import Tweak as tweak
import logging; logger = logging.getLogger(__name__)

#set global variables
pytest.config_dir = os.getcwd() + "/./configs"
pytest.demo_dir = os.getcwd() + "/./../demos/basic"

#set current working directory to directory with the demo files
@pytest.fixture
def set_cwd():
    #os.chdir(pytest.config_dir)
    os.chdir(pytest.demo_dir)
    return None

@pytest.mark.parametrize("test_input, expected", [(None, os.getcwd() + "/basic_demo.yaml"), ("complex_demo.yaml", os.getcwd() + "/complex_demo.yaml")])
def test_init(test_input, expected):
    """
    Tests the initialization by checking if the config files match what is passed in to the constructor
    """
    #TODO figure out how init is supposed to be tested
    twk = tweak(test_input)
    assert twk.configFile == expected

def test_createConfig(set_cwd):
    """
    Check if the way tweak create settings is correct
    """
    twk = tweak()
    #twk.createConfig()
    #assert twk.settings == {}
    assert twk.configFile  == "/home/chrisunix/improv/demos/basic/basic_demo.yaml"
    #TODO figure out why even though we have the fixture, the working directory is still improv/pytest

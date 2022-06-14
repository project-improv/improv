import pytest
import os
import yaml
from inspect import signature
from importlib import import_module

from improv.tweak import RepeatedActorError
from improv.tweak import Tweak as tweak

import logging; logger = logging.getLogger(__name__)

#set global variables

pytest.config_dir = os.getcwd() + "/./configs"

@pytest.fixture

def set_cwd():
    """ Sets the current working directory to the configs file.
    """

    os.chdir(pytest.config_dir)
    return None

@pytest.mark.parametrize(
        "test_input, expected",
        [(None, os.getcwd() + "/basic_demo.yaml"),
        ("good_config.yaml", os.getcwd() + "/good_config.yaml")])

def test_init(test_input, expected):
    """ Tests the initialization by checking if the config files match what is
    passed in to the constructor.

        Asserts:
            Whether tweak has the correct config file.

        TODO:
            Figure out what parts of __init__ should be tested.
    """

    twk = tweak(test_input)
    assert twk.configFile == expected

def test_init_attributes():
    """ Checks if tweak has the correct default attributes on initialization.

    Checks if actors, connection, and hasGUI are all empty or nonexistent.
    Detects errors by maintaining a list of errors, and then adding to it
    every time an unexpected behavior is encountered.

    Asserts:
        If the default attributes are empty or nonexistent.

    """

    twk = tweak()
    errors = []
    sentinel = False

    if(twk.actors != {}):
        errors.append("tweak.actors is not empty! ")
        sentinel = True
    if(twk.connections != {}):
        errors.append("tweak.connections is not empty! ")
        sentinel = True
    if(twk.hasGUI):
        errors.append("tweak.hasGUI already exists! ")
        sentinel = True

    assert not errors, "The following errors occurred:\n{}".format(
                                                            "\n".join(errors))

def test_createConfig_settings(set_cwd):
    """ Check if the default way tweak creates settings is correct.
    """

    twk = tweak("good_config.yaml")
    twk.createConfig()
    assert twk.settings == {'use_watcher' : None}

@pytest.mark.skip(reason = "this case is automatically handled by\
                            yaml.safe_load")

def test_createConfig_RepeatedActorError():
    """ Checks if there is an error with a duplicate actor in the config.
    """

    twk = tweak("repeated_actors.yaml")
    with pytest.raises(RepeatedActorError):
        twk.createConfig()

@pytest.mark.skip(reason = "this test is unfinished")
def test_createConfig_repeatedConnectionsError():
    assert True

@pytest.mark.skip(reason = "this test is unfinished")
def test_createConfig_clean():
    """ Given a good config file, tests if createConfig runs without error.
    """

@pytest.mark.skip(reason = "this test is unfinished")
def test_createConfig_actorsDict():
    """ Checks if tweak has read in the right actors.

        TODO:
            Check if cfg['actors'] has the right key, val pairs.
    """

@pytest.mark.skip(reason = "this test is unfinished")
def test_saveConfig_clean():
    """ Given a good config file, tests if saveConfig runs without error.
    """
    
    twk = tweak("good_config.yaml")
    twk.createConfig()
    twk.saveConfig()
    

@pytest.mark.skip(reason = "this test is unfinished")
def test_saveConfig_noActor():
    """ Checks if there is an error while saving.
    """
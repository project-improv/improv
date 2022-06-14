import pytest
import os
import yaml
from inspect import signature
from importlib import import_module

from improv.tweak import RepeatedActorError, RepeatedConnectionsError
from improv.tweak import Tweak as tweak

import logging; logger = logging.getLogger(__name__)

#set global variables

pytest.config_dir = os.getcwd() + "/./configs"

@pytest.fixture
def set_configdir():
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

def test_createConfig_settings(set_configdir):
    """ Check if the default way tweak creates settings is correct.
    """

    twk = tweak("good_config.yaml")
    twk.createConfig()
    assert twk.settings == {'use_watcher' : None}

def test_createConfig_clean(set_configdir):
    """ Given a good config file, tests if createConfig runs without error.
    """
    twk = tweak("good_config.yaml")
    try:
        twk.createConfig()
    except Exception as exc:
        assert False, f"'createConfig() raised an exception {exc}'"

def test_createConfig_noActor(set_configdir):
    """ Tests if AttributeError is raised when there are no actors.
    """

    twk = tweak("no_actor.yaml")
    with pytest.raises(AttributeError):
        twk.createConfig()

def test_createConfig_ModuleNotFound(set_configdir):
    """ Tests if ModuleNotFoundError is raised when the package can't be found.
    """
    twk = tweak("bad_package.yaml")
    with pytest.raises(ModuleNotFoundError):
        twk.createConfig()

def test_createConfig_ImportError(set_configdir):
    """ Tests if AttributeError is raised when the class name is invalid.
    """

    twk = tweak("bad_class.yaml")
    with pytest.raises(AttributeError):
        twk.createConfig()

def test_createConfig_AttributeError(set_configdir):
    """ Tests if AttributeError is raised.
    """

    twk = tweak("bad_class.yaml")
    with pytest.raises(AttributeError):
        twk.createConfig()

def test_createConfig_blank_file(set_configdir):
    twk = tweak("blank_file.yaml")
    with pytest.raises(TypeError):
        twk.createConfig()

def test_createConfig_nonsense_file(set_configdir):
    twk = tweak("nonsense.yaml")
    with pytest.raises(TypeError):
        twk.createConfig()

#@pytest.mark.skip(reason = "this test is unfinished")
def test_saveConfig_clean():
    """ Given a good config file, tests if saveConfig runs without error.
    """

    twk = tweak("good_config.yaml")
    x = twk.saveConfig()
    y = ['Acquirer', 'Processor', 'Analysis']
    assert x.keys() == y

@pytest.mark.skip(reason = "this test is unfinished")
def test_saveConfig_noActor():
    """ Checks if there is an error while saving.
    """

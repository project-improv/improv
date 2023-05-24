import pytest
import os
import yaml

# from inspect import signature
# from importlib import import_module

# from improv.config import RepeatedActorError
from improv.config import Config as config
from improv.utils import checks

import logging

logger = logging.getLogger(__name__)

# set global variables


@pytest.fixture()
def set_configdir():
    """Sets the current working directory to the configs file."""
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__) + "/configs")
    yield None
    os.chdir(prev)


@pytest.mark.parametrize("test_input", [("good_config.yaml")])
def test_init(test_input, set_configdir):
    """Checks if cfg.configFile matches the provided configFile.

    Asserts:
        Whether config has the correct config file.
    """

    cfg = config(test_input)
    assert cfg.configFile == test_input


# def test_init_attributes():
#     """ Tests if config has correct default attributes on initialization.

#     Checks if actors, connection, and hasGUI are all empty or
#     nonexistent. Detects errors by maintaining a list of errors, and
#     then adding to it every time an unexpected behavior is encountered.

#     Asserts:
#         If the default attributes are empty or nonexistent.

#     """

#     cfg = config()
#     errors = []

#     if(cfg.actors != {}):
#         errors.append("config.actors is not empty! ")
#     if(cfg.connections != {}):
#         errors.append("config.connections is not empty! ")
#     if(cfg.hasGUI):
#         errors.append("config.hasGUI already exists! ")

#     assert not errors, "The following errors occurred:\n{}".format(
#                                                             "\n".join(errors))


def test_createConfig_settings(set_configdir):
    """Check if the default way config creates config.settings is correct.

    Asserts:
        If the default setting is the dictionary {"use_watcher": "None"}
    """

    cfg = config("good_config.yaml")
    cfg.createConfig()
    assert cfg.settings == {"use_watcher": None}


def test_createConfig_clean(set_configdir):
    """Tests if createConfig runs without error given a good config.

    Asserts:
        If createConfig does not raise any errors.
    """

    cfg = config("good_config.yaml")
    try:
        cfg.createConfig()
    except Exception as exc:
        pytest.fail(f"createConfig() raised an exception {exc}")


def test_createConfig_noActor(set_configdir):
    """Tests if AttributeError is raised when there are no actors."""

    cfg = config("no_actor.yaml")
    with pytest.raises(AttributeError):
        cfg.createConfig()


def test_createConfig_ModuleNotFound(set_configdir):
    """Tests if an error is raised when the package can"t be found."""

    cfg = config("bad_package.yaml")
    with pytest.raises(ModuleNotFoundError):
        cfg.createConfig()


def test_createConfig_class_ImportError(set_configdir):
    """Tests if an error is raised when the class name is invalid."""

    cfg = config("bad_class.yaml")
    with pytest.raises(AttributeError):
        cfg.createConfig()


def test_createConfig_AttributeError(set_configdir):
    """Tests if AttributeError is raised."""

    cfg = config("bad_class.yaml")
    with pytest.raises(AttributeError):
        cfg.createConfig()


def test_createConfig_blank_file(set_configdir):
    """Tests if a blank config file raises an error."""

    cfg = config("blank_file.yaml")
    with pytest.raises(TypeError):
        cfg.createConfig()


def test_createConfig_nonsense_file(set_configdir, caplog):
    """Tests if an improperly formatted config raises an error."""

    cfg = config("nonsense.yaml")
    with pytest.raises(TypeError):
        cfg.createConfig()


def test_acyclic_graph(set_configdir):
    path = os.getcwd() + "/good_config.yaml"
    assert checks.check_if_connections_acyclic(path)


def test_cyclic_graph(set_configdir):
    path = os.getcwd() + "/cyclic_config.yaml"
    assert not checks.check_if_connections_acyclic(path)


def test_saveActors_clean(set_configdir):
    """Compares internal actor representation to what was saved in the file."""

    cfg = config("good_config.yaml")
    cfg.createConfig()
    cfg.saveActors()

    with open("good_config_actors.yaml") as savedConfig:
        data = yaml.safe_load(savedConfig)
    savedKeys = len(data.keys())

    originalKeys = len(cfg.actors.keys())

    assert savedKeys == originalKeys

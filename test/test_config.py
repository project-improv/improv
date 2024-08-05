import pytest
import os
import yaml

# from inspect import signature
# from importlib import import_module

# from improv.config import RepeatedActorError
from improv.config import Config, CannotCreateConfigException
from improv.utils import checks

import logging

logger = logging.getLogger(__name__)


# set global variables


@pytest.fixture
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

    cfg = Config(test_input)
    assert cfg.config_file == test_input


def test_create_config_settings(set_configdir):
    """Check if the default way config creates config.settings is correct.

    Asserts:
        If the default setting is the config as a dict.
    """

    cfg = Config("good_config.yaml")
    cfg.parse_config()
    cfg.create_config()
    assert cfg.settings == {
        "control_port": 5555,
        "output_port": 5556,
        "store_size": 250_000_000,
        "actor_in_port": 0,
        "harvest_data_from_memory": None,
    }


def test_create_config_init_typo(set_configdir):
    """Tests if createConfig can catch actors with errors in init function.

    Asserts:
        If createConfig raise any errors.
    """

    cfg = Config("minimal_wrong_init.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_create_config_wrong_import(set_configdir):
    """Tests if createConfig can catch actors with errors during import.

    Asserts:
        If createConfig raise any errors.
    """

    cfg = Config("minimal_wrong_import.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_create_config_clean(set_configdir):
    """Tests if createConfig runs without error given a good config.

    Asserts:
        If createConfig does not raise any errors.
    """

    cfg = Config("good_config.yaml")
    cfg.parse_config()
    try:
        cfg.create_config()
    except Exception as exc:
        pytest.fail(f"createConfig() raised an exception {exc}")


def test_create_config_no_actor(set_configdir):
    """Tests if AttributeError is raised when there are no actors."""

    cfg = Config("no_actor.yaml")
    cfg.parse_config()
    with pytest.raises(AttributeError):
        cfg.create_config()


def test_create_config_module_not_found(set_configdir):
    """Tests if an error is raised when the package can"t be found."""

    cfg = Config("bad_package.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_create_config_class_import_error(set_configdir):
    """Tests if an error is raised when the class name is invalid."""

    cfg = Config("bad_class.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_create_config_attribute_error(set_configdir):
    """Tests if AttributeError is raised."""

    cfg = Config("bad_class.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_create_config_blank_file(set_configdir):
    """Tests if a blank config file raises an error."""

    with pytest.raises(CannotCreateConfigException):
        Config("blank_file.yaml")


def test_create_config_nonsense_file(set_configdir, caplog):
    """Tests if an improperly formatted config raises an error."""

    with pytest.raises(TypeError):
        Config("nonsense.yaml")


def test_acyclic_graph(set_configdir):
    path = os.getcwd() + "/good_config.yaml"
    assert checks.check_if_connections_acyclic(path)


def test_cyclic_graph(set_configdir):
    path = os.getcwd() + "/cyclic_config.yaml"
    assert not checks.check_if_connections_acyclic(path)


def test_save_actors_clean(set_configdir):
    """Compares internal actor representation to what was saved in the file."""

    cfg = Config("good_config.yaml")
    cfg.parse_config()
    cfg.create_config()
    cfg.save_actors()

    with open("good_config_actors.yaml") as savedConfig:
        data = yaml.safe_load(savedConfig)
    savedKeys = len(data.keys())

    originalKeys = len(cfg.actors.keys())

    assert savedKeys == originalKeys

    os.remove("good_config_actors.yaml")


def test_config_settings_read(set_configdir):
    cfg = Config("minimal_with_settings.yaml")
    cfg.parse_config()
    cfg.create_config()

    assert "store_size" in cfg.settings


def test_config_bad_actor_args(set_configdir):
    cfg = Config("bad_args.yaml")
    cfg.parse_config()
    res = cfg.create_config()
    assert res == -1


def test_config_harvester_disabled(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["settings"] = dict()
    cfg.parse_config()
    assert cfg.settings["harvest_data_from_memory"] is None


def test_config_harvester_enabled(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["settings"] = dict()
    cfg.config["settings"]["harvest_data_from_memory"] = True
    cfg.parse_config()
    assert cfg.settings["harvest_data_from_memory"]


def test_config_redis_aof_enabled_saving_not_specified(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["aof_dirname"] = "test"
    cfg.parse_config()
    assert cfg.redis_config["aof_dirname"] == "test"
    assert cfg.redis_config["enable_saving"] is True


def test_config_redis_ephemeral_dirname_enabled_saving_not_specified(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["generate_ephemeral_aof_dirname"] = True
    cfg.parse_config()
    assert cfg.redis_config["generate_ephemeral_aof_dirname"] is True
    assert cfg.redis_config["enable_saving"] is True


def test_config_redis_ephemeral_dirname_and_aof_dirname_specified(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["generate_ephemeral_aof_dirname"] = True
    cfg.config["redis_config"]["aof_dirname"] = "test"
    with pytest.raises(
        Exception, match="Cannot use unique dirname and use the one provided."
    ):
        cfg.parse_config()


def test_config_redis_ephemeral_dirname_enabled_saving_disabled(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["generate_ephemeral_aof_dirname"] = True
    cfg.config["redis_config"]["enable_saving"] = False
    with pytest.raises(Exception, match="Cannot persist to disk with saving disabled."):
        cfg.parse_config()


def test_config_redis_aof_dirname_enabled_saving_disabled(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["aof_dirname"] = "test"
    cfg.config["redis_config"]["enable_saving"] = False
    with pytest.raises(Exception, match="Cannot persist to disk with saving disabled."):
        cfg.parse_config()


def test_config_redis_fsync_enabled_saving_disabled(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["fsync_frequency"] = "always"
    cfg.config["redis_config"]["enable_saving"] = False
    with pytest.raises(Exception, match="Cannot persist to disk with saving disabled."):
        cfg.parse_config()


def test_config_redis_unknown_fsync_freq(set_configdir):
    cfg = Config("minimal.yaml")
    cfg.config = dict()
    cfg.config["redis_config"] = dict()
    cfg.config["redis_config"]["fsync_frequency"] = "unknown"
    with pytest.raises(Exception, match="Cannot use unknown fsync frequency unknown"):
        cfg.parse_config()


def test_config_gui(set_configdir):
    cfg = Config("minimal_gui.yaml")
    cfg.parse_config()
    cfg.create_config()

    assert cfg.hasGUI is True
    assert cfg.gui.classname == "Generator"

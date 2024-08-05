import yaml
import logging
from inspect import signature
from importlib import import_module

logger = logging.getLogger(__name__)


class CannotCreateConfigException(Exception):
    def __init__(self, msg):
        super().__init__("Cannot create config: {}".format(msg))


class Config:
    """Handles configuration and logs of configs for
    the entire server/processing pipeline.
    """

    def __init__(self, config_file):
        self.actors = {}
        self.connections = {}
        self.hasGUI = False
        self.config_file = config_file

        with open(self.config_file, "r") as ymlfile:
            self.config = yaml.safe_load(ymlfile)

        if self.config is None:
            logger.error("The config file is empty")
            raise CannotCreateConfigException("The config file is empty")

        if type(self.config) is not dict:
            logger.error("Error: The config file is not in dictionary format")
            raise TypeError

    def parse_config(self):
        self.populate_defaults()
        self.validate_config()

        self.settings = self.config["settings"]
        self.redis_config = self.config["redis_config"]

    def populate_defaults(self):
        self.populate_settings_defaults()
        self.popoulate_redis_defaults()

    def validate_config(self):
        self.validate_redis_config()

    def create_config(self):
        """Read yaml config file and create config for Nexus
        TODO: check for config file compliance, error handle it
        beyond what we have below.
        """
        cfg = self.config

        for name, actor in cfg["actors"].items():
            packagename = actor.pop("package")
            classname = actor.pop("class")

            try:
                __import__(packagename, fromlist=[classname])
                mod = import_module(packagename)

                actor_class = getattr(mod, classname)
                sig = signature(actor_class)
                config_module = ConfigModule(
                    name, packagename, classname, options=actor
                )
                sig.bind(config_module.options)

            # TODO: this is not trivial to test, since our code formatting
            #   tools won't allow a file with a syntax error to exist
            except SyntaxError as e:
                logger.error(f"Error: syntax error when initializing actor {name}: {e}")
                return -1

            except ModuleNotFoundError as e:
                logger.error(
                    f"Error: failed to import packages, {e}. Please check both each "
                    f"actor's imports and the package name in the yaml file."
                )

                return -1

            except AttributeError:
                logger.error("Error: Classname not valid within package")
                return -1

            except TypeError:
                logger.error("Error: Invalid arguments passed")
                params = ""
                for param_name, param in sig.parameters.items():
                    params = params + ", " + param.name
                logger.warning("Expected Parameters:" + params)
                return -1

            except Exception as e:  # TODO: figure out how to test this
                logger.error(f"Error: {e}")
                return -1

            if "GUI" in name:
                logger.info(f"Config detected a GUI actor: {name}")
                self.hasGUI = True
                self.gui = config_module
            else:
                self.actors.update({name: config_module})

        for name, conn in cfg["connections"].items():
            self.connections.update({name: conn})

        return 0

    def save_actors(self):
        """Saves the actors config to a specific file."""
        wflag = True
        saveFile = self.config_file.split(".")[0]
        pathName = saveFile + "_actors.yaml"

        for a in self.actors.values():
            wflag = a.save_config_modules(pathName, wflag)

    def populate_settings_defaults(self):
        if "settings" not in self.config:
            self.config["settings"] = {}

        if "store_size" not in self.config["settings"]:
            self.config["settings"]["store_size"] = 250_000_000
        if "control_port" not in self.config["settings"]:
            self.config["settings"]["control_port"] = 5555
        if "output_port" not in self.config["settings"]:
            self.config["settings"]["output_port"] = 5556
        if "actor_in_port" not in self.config["settings"]:
            self.config["settings"]["actor_in_port"] = 0
        if "harvest_data_from_memory" not in self.config["settings"]:
            self.config["settings"]["harvest_data_from_memory"] = None

    def popoulate_redis_defaults(self):
        if "redis_config" not in self.config:
            self.config["redis_config"] = {}

        if "enable_saving" not in self.config["redis_config"]:
            self.config["redis_config"]["enable_saving"] = None
        if "aof_dirname" not in self.config["redis_config"]:
            self.config["redis_config"]["aof_dirname"] = None
        if "generate_ephemeral_aof_dirname" not in self.config["redis_config"]:
            self.config["redis_config"]["generate_ephemeral_aof_dirname"] = False
        if "fsync_frequency" not in self.config["redis_config"]:
            self.config["redis_config"]["fsync_frequency"] = None

        # enable saving automatically if the user configured a saving option
        if (
            self.config["redis_config"]["aof_dirname"]
            or self.config["redis_config"]["generate_ephemeral_aof_dirname"]
            or self.config["redis_config"]["fsync_frequency"]
        ) and self.config["redis_config"]["enable_saving"] is None:
            self.config["redis_config"]["enable_saving"] = True

        if "port" not in self.config["redis_config"]:
            self.config["redis_config"]["port"] = 6379

    def validate_redis_config(self):
        fsync_name_dict = {
            "every_write": "always",
            "every_second": "everysec",
            "no_schedule": "no",
        }
        if (
            self.config["redis_config"]["aof_dirname"]
            and self.config["redis_config"]["generate_ephemeral_aof_dirname"]
        ):
            logger.error(
                "Cannot both generate a unique dirname and use the one provided."
            )
            raise Exception("Cannot use unique dirname and use the one provided.")

        if (
            self.config["redis_config"]["aof_dirname"]
            or self.config["redis_config"]["generate_ephemeral_aof_dirname"]
            or self.config["redis_config"]["fsync_frequency"]
        ):
            if not self.config["redis_config"]["enable_saving"]:
                logger.error(
                    "Invalid configuration. Cannot save to disk with saving disabled."
                )
                raise Exception("Cannot persist to disk with saving disabled.")

        if self.config["redis_config"]["fsync_frequency"] and self.config[
            "redis_config"
        ]["fsync_frequency"] not in [
            "every_write",
            "every_second",
            "no_schedule",
        ]:
            logger.error(
                f"Cannot use unknown fsync frequency "
                f'{self.config["redis_config"]["fsync_frequency"]}'
            )
            raise Exception(
                f"Cannot use unknown fsync frequency "
                f'{self.config["redis_config"]["fsync_frequency"]}'
            )

        if self.config["redis_config"]["fsync_frequency"] is None:
            self.config["redis_config"]["fsync_frequency"] = "no_schedule"

        self.config["redis_config"]["fsync_frequency"] = (fsync_name_dict)[
            self.config["redis_config"]["fsync_frequency"]
        ]


class ConfigModule:
    def __init__(self, name, packagename, classname, options=None):
        self.name = name
        self.packagename = packagename
        self.classname = classname
        self.options = options

    def save_config_modules(self, path_name, wflag):
        """Loops through each actor to save the modules to the config file.

        Args:
            path_name:
            wflag (bool):

        Returns:
            bool: wflag
        """

        if wflag:
            writeOption = "w"
            wflag = False
        else:
            writeOption = "a"

        cfg = {self.name: {"package": self.packagename, "class": self.classname}}

        for key, value in self.options.items():
            cfg[self.name].update({key: value})

        with open(path_name, writeOption) as file:
            yaml.dump(cfg, file)

        return wflag

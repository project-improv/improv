import yaml
import logging
from inspect import signature
from importlib import import_module

logger = logging.getLogger(__name__)


class Config:
    """Handles configuration and logs of configs for
    the entire server/processing pipeline.
    """

    def __init__(self, configFile):
        if configFile is None:
            logger.error("Need to specify a config file")
            raise Exception
        else:
            # Reading config from other yaml file
            self.configFile = configFile

        with open(self.configFile, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        try:
            if "settings" in cfg:
                self.settings = cfg["settings"]
            else:
                self.settings = {}
            
            if not "use_watcher" in self.settings:
                self.settings["use_watcher"] = None

        except TypeError:
            if cfg is None:
                logger.error("Error: The config file is empty")

        if type(cfg) is not dict:
            logger.error("Error: The config file is not in dictionary format")
            raise TypeError
        
        self.config = cfg

        self.actors = {}
        self.connections = {}
        self.hasGUI = False

    def createConfig(self):
        """Read yaml config file and create config for Nexus
        TODO: check for config file compliance, error handle it
        beyond what we have below.
        """
        cfg = self.config

        for name, actor in cfg["actors"].items():
            if name in self.actors.keys():
                raise RepeatedActorError(name)

            packagename = actor.pop("package")
            classname = actor.pop("class")

            try:
                __import__(packagename, fromlist=[classname])
                mod = import_module(packagename)

                clss = getattr(mod, classname)
                sig = signature(clss)
                configModule = ConfigModule(name, packagename, classname, options=actor)
                sig.bind(configModule.options)

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
                for parameter in sig.parameters:
                    params = params + " " + parameter.name
                logger.warning("Expected Parameters:" + params)
                return -1

            except Exception as e:
                logger.error(f"Error: {e}")
                return -1

            if "GUI" in name:
                logger.info(f"Config detected a GUI actor: {name}")
                self.hasGUI = True
                self.gui = configModule
            else:
                self.actors.update({name: configModule})

        for name, conn in cfg["connections"].items():
            if name in self.connections.keys():
                raise RepeatedConnectionsError(name)

            self.connections.update({name: conn})
        return 0

    def addParams(self, type, param):
        """Function to add paramter param of type type
        TODO: Future work
        """
        pass

    def saveActors(self):
        """Saves the actors config to a specific file."""
        wflag = True
        saveFile = self.configFile.split(".")[0]
        pathName = saveFile + "_actors.yaml"

        for a in self.actors.values():
            wflag = a.saveConfigModules(pathName, wflag)


class ConfigModule:
    def __init__(self, name, packagename, classname, options=None):
        self.name = name
        self.packagename = packagename
        self.classname = classname
        self.options = options

    def saveConfigModules(self, pathName, wflag):
        """Loops through each actor to save the modules to the config file."""

        if wflag:
            writeOption = "w"
            wflag = False
        else:
            writeOption = "a"

        cfg = {self.name: {"package": self.packagename, "class": self.classname}}

        for key, value in self.options.items():
            cfg[self.name].update({key: value})

        with open(pathName, writeOption) as file:
            yaml.dump(cfg, file)

        return wflag


class RepeatedActorError(Exception):
    def __init__(self, repeat):
        super().__init__()

        self.name = "RepeatedActorError"
        self.repeat = repeat

        self.message = 'Actor name has already been used: "{}"'.format(repeat)

    def __str__(self):
        return self.message


class RepeatedConnectionsError(Exception):
    def __init__(self, repeat):
        super().__init__()
        self.name = "RepeatedConnectionsError"
        self.repeat = repeat

        self.message = 'Connection name has already been used: "{}"'.format(repeat)

    def __str__(self):
        return self.message

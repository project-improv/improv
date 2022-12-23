import os
import yaml
import io
from inspect import signature
from importlib import import_module
import logging; logger = logging.getLogger(__name__)

#TODO: Write a save function for Config objects output as YAML configFile but using ConfigModule objects

class Config():
    ''' Handles configuration and logs of configs for
        the entire server/processing pipeline.
    '''

    def __init__(self, configFile):
        cwd = os.getcwd()
        if configFile is None:
            logger.error('Need to specify a config file')
            raise Exception
        else:
            # Reading config from other yaml file
            self.configFile = configFile

        self.actors = {}
        self.connections = {}
        self.hasGUI = False

    def createConfig(self):
        ''' Read yaml config file and create config for Nexus
            TODO: check for config file compliance, error handle it
        '''
        with open(self.configFile, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        try:
            if 'settings' in cfg:
                self.settings = cfg['settings']
            else:
                self.settings = {}
                self.settings['use_watcher'] = None
        except TypeError:
            if cfg is None:
                logger.error('Error: The config file is empty')

        if type(cfg) != dict:
            logger.error("Error: The config file is not in dictionary format")
            raise TypeError

        for name,actor in cfg['actors'].items():
            # put import/name info in ConfigModule object TODO: make ordered?

            if name in self.actors.keys():
                # Should be actor.keys() - self.actors.keys() is empty until update?
                raise RepeatedActorError(name)

            packagename = actor.pop('package')
            classname = actor.pop('class')

            try:
                __import__(packagename, fromlist=[classname])

            except ModuleNotFoundError:
                logger.error('Error: Packagename not valid')

            except AttributeError:
                logger.error('Error: Classname not valid within package')

            mod = import_module(packagename)

            clss = getattr(mod, classname)
            sig= signature(clss)
            configModule = ConfigModule(name, packagename, classname, options=actor)
            try:
                sig.bind(configModule.options)
            except TypeError as e:
                logger.error('Error: Invalid arguments passed')
                params= ''
                for parameter in sig.parameters:
                    params = params + ' ' + parameter.name
                logger.warning('Expected Parameters:' + params)
            if "GUI" in name:
                logger.info('Config detected a GUI actor: {}'.format(name))
                self.hasGUI = True
                self.gui = configModule

            else:
                self.actors.update({name:configModule})

        for name,conn in cfg['connections'].items():
            #TODO check for correctness  TODO: make more generic (not just q_out)
            if name in self.connections.keys():
                raise RepeatedConnectionsError(name)

            self.connections.update({name:conn}) #conn should be a list


    def addParams(self, type, param):
        ''' Function to add paramter param of type type
        '''

    def saveActors(self):
        ''' Saves the config to a specific file.
        '''
        
        wflag = True
        
        cfg = self.actors

        saveFile = self.configFile.split('.')[0]
        pathName = saveFile + '_actors.yaml'

        for a in self.actors.values():
            wflag = a.saveConfigModules(pathName, wflag)


class ConfigModule():
    def __init__(self, name, packagename, classname, options=None):
        self.name = name
        self.packagename = packagename
        self.classname = classname
        self.options = options

    def saveConfigModules(self, pathName, wflag):
        ''' Loops through each actor to save the modules to the config file.
        '''

        if wflag: 
            writeOption = 'w'
            wflag = False
            #cfg = {"actors": []}
        else:
            writeOption = 'a'

        cfg = {self.name:
                    {'package':self.packagename,
                    'class':self.classname}} # fix indentation

        #for name in self.name:
        #for b in self.name:
        for key, value in self.options.items():
            cfg[self.name].update({key:value})         
        
        with open(pathName, writeOption) as file:
            documents = yaml.dump(cfg, file)
        
        return wflag

class RepeatedActorError(Exception):
    def __init__(self, repeat):

        super().__init__()

        self.name = 'RepeatedActorError'
        self.repeat = repeat

        self.message = 'Actor name has already been used: "{}"'.format(repeat)

    def __str__(self):
        return self.message


class RepeatedConnectionsError(Exception):
    def __init__(self, repeat):

        super().__init__()
        self.name= 'RepeatedConnectionsError'
        self.repeat=repeat

        self.message= 'Connection name has already been used: "{}"'.format(repeat)

    def __str__(self):
        return self.message


if __name__ == '__main__':
    config = Config(configFile='pytest/configs/good_config.yaml')
    config.createConfig()
    config.saveActors()
    # for actor in config.actors:
    #     print(actor)

    # for connection in config.connections:
    #     print(connection)

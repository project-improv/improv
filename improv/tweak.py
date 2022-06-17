import os
import yaml
import io
from inspect import signature
from importlib import import_module
import logging; logger = logging.getLogger(__name__)

#TODO: Write a save function for Tweak objects output as YAML configFile but using TweakModule objects

class Tweak():
    ''' Handles configuration and logs of configs for
        the entire server/processing pipeline.
    '''

    def __init__(self, configFile=None):
        cwd = os.getcwd()
        if configFile is None:
            # Going with default config
            self.configFile = cwd+'/basic_demo.yaml'
        else:
            # Reading config from other yaml file
            self.configFile = cwd+'/'+configFile

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
            logger.error('Error: The config file is empty')

        try:
            actor_items = cfg['actors'].items()[0]
        except TypeError:
            logger.error('Error: The config file is formatted incorrectly')

        for name,actor in cfg['actors'].items():
            # put import/name info in TweakModule object TODO: make ordered?

            if name in self.actors.keys():
<<<<<<< HEAD
                # Should be actor.keys() - self.actors.keys() is empty until update?
=======
                logger.error('Duplicated actor names detected')
>>>>>>> docs
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
            tweakModule = TweakModule(name, packagename, classname, options=actor)
            try:
                sig.bind(tweakModule.options)
            except TypeError as e:
                logger.error('Error: Invalid arguments passed')
                params= ''
                for parameter in sig.parameters:
                    params = params + ' ' + parameter.name
                logger.warning('Expected Parameters:' + params)
            if "GUI" in name:
                self.hasGUI = True
                self.gui = tweakModule

            else:
                self.actors.update({name:tweakModule})

        for name,conn in cfg['connections'].items():
            #TODO check for correctness  TODO: make more generic (not just q_out)
            if name in self.connections.keys():
                raise RepeatedConnectionsError(name)

            self.connections.update({name:conn}) #conn should be a list


    def addParams(self, type, param):
        ''' Function to add paramter param of type type
        '''

    def saveConfig(self):
        #remake cfg TODO
        cfg = self.actors
        
        saveFile = self.configFile.split('.')[0]
        pathName = saveFile + '_save.yaml'

        for a in self.actors.values():
            a.saveConfig(pathName) 
        # TODO iterate through strings in twkmodule
       

class TweakModule():
    def __init__(self, name, packagename, classname, options=None):
        self.name = name
        self.packagename = packagename
        self.classname = classname
        self.options = options

    def saveConfigModules(self, pathName):

        cfg = {'package':self.packagename,
                'class':self.classname,
                'name':self.name,
                'options':self.options} 
        # TODO finish building dictionary of tweakModule strings
        # TODO run through options in for loop

        #for key, value in self.options:
        #    [].extend([
        #        {key:value}
        #    ])

        cfg.update(self.options) # TODO append to combine the 2 dictionaries

        with open(pathName, 'w') as file: # TODO append to file instead of write
            documents = yaml.dump(cfg, file)

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
    tweak = Tweak(configFile='test/basic_demo')
    tweak.createConfig()
    for actor in tweak.actors:
        print(actor)

    for connection in tweak.connections:
        print(connection)

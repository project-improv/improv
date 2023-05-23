import os
import yaml
import io
import filecmp

import logging

logger = logging.getLogger(__name__)
from improv.config import Config, ConfigModule
from unittest import TestCase
from test.test_utils import StoreDependentTestCase

# from test.visual import visual
import improv.actors.acquire
import improv.actors.process
import improv.actors.analysis
from inspect import signature


class createConfBasic(StoreDependentTestCase):
    def setUp(self):
        super(createConfBasic, self).setUp()
        self.config = Config(configFile="test/configs/good_config.yaml")

    def test_actor(self):
        self.config.createConfig()
        self.assertEqual(
            self.config.actors["Processor"].packagename, "improv.actors.process"
        )

    """
    Commented out since we our moving visual testing to its own test directory as /test/visual
    def test_GUI(self):
        self.config.createConfig()
        self.assertEqual(self.config.gui.packagename, 'visual.visual')
        self.assertEqual(self.config.gui.classname, 'DisplayVisual')
    """

    def test_connections(self):
        self.config.createConfig()
        self.assertEqual(self.config.connections["Acquirer.q_out"], ["Processor.q_in"])
        self.assertEqual(self.config.connections["Processor.q_out"], ["Analysis.q_in"])

    def tearDown(self):
        super(createConfBasic, self).tearDown()


class FailCreateConf(StoreDependentTestCase):
    def setUp(self):
        super(FailCreateConf, self).setUp()
        self.config1 = Config(configFile="test/configs/repeated_actors.yaml")
        self.config2 = Config(configFile="test/configs/no_actor.yaml")
        self.config3 = Config(configFile="test/configs/repeated_actors")

    def MissingPackageorClass(self):
        cwd = os.getcwd()
        self.config1.createConfig()
        self.assertEqual(
            self.config.configFile, cwd + "test/configs/MissingPackage.yaml"
        )
        self.assertRaises(KeyError)
        # TODO: create repeated actor error

    def noactors(self):
        cwd = os.getcwd()
        self.config2.createConfig()
        self.assertRaises(AttributeError)

    def repeatedActor(self):
        cwd = os.getcwd()
        self.config3.createConfig()
        self.assertRaises(RepeatedActorError)

    def tearDown(self):
        super(FailCreateConf, self).tearDown()


class testPackageClass(StoreDependentTestCase):
    def setUp(self):
        super(testPackageClass, self).setUp()
        self.config1 = Config(configFile="test/configs/bad_package.yaml")
        self.config2 = Config(configFile="test/configs/bad_class.yaml")

    def badpackage(self):
        cwd = os.getcwd()
        self.config1.createConfig()
        self.assertRaises(ModuleNotFoundError)

    def badclass(self):
        cwd = os.getcwd()
        self.config2.createConfig()
        self.assertRaises(ImportError)

    def tearDown(self):
        super(testPackageClass, self).tearDown()


class testArgs(StoreDependentTestCase):
    def setUp(self):
        super(testArgs, self).setUp()
        self.config = Config(configFile="test/configs/bad_args.yaml")

    def testArgs(self):
        cwd = os.getcwd()
        self.config.createConfig()
        self.assertRaises(TypeError)

    def tearDown(self):
        super(testArgs, self).tearDown()


class testconnections(StoreDependentTestCase):
    def setUp(self):
        super(testconnections, self).setUp()
        self.config = Config(configFile="test/configs/repeat_connection")

    def repeatConnection(self):
        cwd = os.getcwd()
        self.config.createConfig()
        self.assertRaises(RepeatedConnectionsError)

    def tearDown(self):
        super(testconnections, self).tearDown()


class FailCreateConf(StoreDependentTestCase):
    def setUp(self):
        super(FailCreateConf, self).setUp()
        self.config = Config()

    def MissingPackageorClass(self):
        cwd = os.getcwd()
        self.config.createConfig(configFile="test/repeated_actors.yaml")
        self.assertEqual(self.config.configFile, cwd + "test/MissingPackage.yaml")
        self.assertRaises(KeyError)
        # TODO: create repeated actor error

    def noactors(self):
        cwd = os.getcwd()
        self.config.createConfig(configFile="test/no_actor.yaml")
        self.assertRaises(AttributeError)

    def tearDown(self):
        super(FailCreateConf)


# TODO: create config but with different config files

:py:mod:`improv.config`
=======================

.. py:module:: improv.config


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.config.Config
   improv.config.ConfigModule




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.config.logger


.. py:data:: logger

   

.. py:class:: Config(configFile)

   Handles configuration and logs of configs for
   the entire server/processing pipeline.

   .. py:method:: createConfig()

      Read yaml config file and create config for Nexus
      TODO: check for config file compliance, error handle it
      beyond what we have below.


   .. py:method:: addParams(type, param)

      Function to add paramter param of type type
      TODO: Future work


   .. py:method:: saveActors()

      Saves the actors config to a specific file.



.. py:class:: ConfigModule(name, packagename, classname, options=None)

   .. py:method:: saveConfigModules(pathName, wflag)

      Loops through each actor to save the modules to the config file.

      :param pathName:
      :param wflag:
      :type wflag: bool

      :returns: wflag
      :rtype: bool



.. py:exception:: RepeatedActorError(repeat)

   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.


.. py:exception:: RepeatedConnectionsError(repeat)

   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.



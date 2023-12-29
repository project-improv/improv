:py:mod:`improv.watcher`
========================

.. py:module:: improv.watcher


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.watcher.BasicWatcher
   improv.watcher.Watcher




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.watcher.logger


.. py:data:: logger

   

.. py:class:: BasicWatcher(*args, inputs=None)

   Bases: :py:obj:`improv.actor.Actor`

   Actor that monitors stored objects from the other actors
   and saves objects that have been flagged by those actors

   .. py:method:: setup()

      set up tasks and polling based on inputs which will
      be used for asynchronous polling of input queues


   .. py:method:: run()

      continually run the watcher to check all of the
      input queues for objects to save


   .. py:method:: watchrun()

      set up async loop for polling


   .. py:method:: watch()
      :async:

      function for asynchronous polling of input queues
      loops through each of the queues in watchin and checks
      if an object is present and then saves the object if found



.. py:class:: Watcher(name, client)

   Monitors the store as separate process
   TODO: Facilitate Watcher being used in multiple processes (shared list)

   .. py:method:: setLinks(links)


   .. py:method:: run()


   .. py:method:: saveObj(obj, name)


   .. py:method:: checkStoreInterface2()




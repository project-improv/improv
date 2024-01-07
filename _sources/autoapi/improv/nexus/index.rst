:py:mod:`improv.nexus`
======================

.. py:module:: improv.nexus


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.nexus.Nexus




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.nexus.logger


.. py:data:: logger

   

.. py:class:: Nexus(name='Server')

   Main server class for handling objects in improv

   .. py:method:: createNexus(file=None, use_hdd=False, use_watcher=None, store_size=10000000, control_port=0, output_port=0)

      Function to initialize class variables based on config file.

      Starts a store of class Limbo, and then loads the config file.
      The config file specifies the specific actors that nexus will
      be connected to, as well as their links.

      :param file: Name of the config file.
      :type file: string
      :param use_hdd: Whether to use hdd for the store.
      :type use_hdd: bool
      :param use_watcher: Whether to use watcher for the store.
      :type use_watcher: bool
      :param store_size: initial store size
      :type store_size: int
      :param control_port: port number for input socket
      :type control_port: int
      :param output_port: port number for output socket
      :type output_port: int

      :returns: "Shutting down", to notify start() that pollQueues has completed.
      :rtype: string


   .. py:method:: loadConfig(file)

      Load configuration file.
      file: a YAML configuration file name


   .. py:method:: initConfig()

      For each connection:
      create a Link with a name (purpose), start, and end
      Start links to one actor's name, end to the other.
      Nexus gives start_actor the Link as a q_in,
      and end_actor the Link as a q_out.
      Nexus maintains dict of name and associated Link.
      Nexus also has list of Links that it is itself connected to
      for communication purposes.

      OR
      For each connection, create 2 Links. Nexus acts as intermediary.

      :param file: input config filepath
      :type file: string


   .. py:method:: startNexus()

      Puts all actors in separate processes and begins polling
      to listen to comm queues


   .. py:method:: start()

      Start all the processes in Nexus


   .. py:method:: destroyNexus()

      Method that calls the internal method
      to kill the process running the store (plasma server)


   .. py:method:: pollQueues()
      :async:

      Listens to links and processes their signals.

      For every communications queue connected to Nexus, a task is
      created that gets from the queue. Throughout runtime, when these
      queues output a signal, they are processed by other functions.
      At the end of runtime (when the gui has been closed), polling is
      stopped.

      :returns: "Shutting down", Notifies start() that pollQueues has completed.
      :rtype: string


   .. py:method:: stop_polling_and_quit(signal, queues)

      quit the process and stop polling signals from queues

      :param signal (): Signal for signal handler.
      :param queues: Comm queues for links.
      :type queues: improv.link.AsyncQueue


   .. py:method:: remote_input()
      :async:


   .. py:method:: processGuiSignal(flag, name)

      Receive flags from the Front End as user input


   .. py:method:: processActorSignal(sig, name)


   .. py:method:: setup()


   .. py:method:: run()


   .. py:method:: quit()


   .. py:method:: stop()


   .. py:method:: revive()


   .. py:method:: stop_polling(stop_signal, queues)

      Cancels outstanding tasks and fills their last request.

      Puts a string into all active queues, then cancels their
      corresponding tasks. These tasks are not fully cancelled until
      the next run of the event loop.

      :param stop_signal (): Signal for signal handler.
      :param queues: Comm queues for links.
      :type queues: improv.link.AsyncQueue


   .. py:method:: createStoreInterface(name)

      Creates StoreInterface w/ or w/out LMDB
      functionality based on {self.use_hdd}.


   .. py:method:: createActor(name, actor)

      Function to instantiate actor, add signal and comm Links,
      and update self.actors dictionary

      :param name: name of the actor
      :param actor: improv.actor.Actor


   .. py:method:: runActor(actor)

      Run the actor continually; used for separate processes
      #TODO: hook into monitoring here?

      :param actor:


   .. py:method:: createConnections()

      Assemble links (multi or other)
      for later assignment


   .. py:method:: assignLink(name, link)

      Function to set up Links between actors
      for data location passing
      Actor must already be instantiated

      #NOTE: Could use this for reassigning links if actors crash?

      #TODO: Adjust to use default q_out and q_in vs being specified


   .. py:method:: startWatcher()




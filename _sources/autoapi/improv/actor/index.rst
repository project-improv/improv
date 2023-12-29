:py:mod:`improv.actor`
======================

.. py:module:: improv.actor


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.actor.AbstractActor
   improv.actor.ManagedActor
   improv.actor.AsyncActor
   improv.actor.RunManager
   improv.actor.AsyncRunManager
   improv.actor.Signal




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.actor.logger
   improv.actor.Actor


.. py:data:: logger

   

.. py:class:: AbstractActor(name, store_loc, method='fork')

   Base class for an actor that Nexus
   controls and interacts with.
   Needs to have a store and links for communication
   Also needs to be responsive to sent Signals (e.g. run, setup, etc)

   .. py:method:: setStoreInterface(client)

      Sets the client interface to the store

      :param client: Set client interface to the store
      :type client: improv.store.StoreInterface


   .. py:method:: setLinks(links)

      General full dict set for links

      :param links: The dict to store all the links
      :type links: dict


   .. py:method:: setCommLinks(q_comm, q_sig)

      Set explicit communication links to/from Nexus (q_comm, q_sig)

      :param q_comm: for messages from this actor to Nexus
      :type q_comm: improv.nexus.Link
      :param q_sig: signals from Nexus and must be checked first
      :type q_sig: improv.nexus.Link


   .. py:method:: setLinkIn(q_in)

      Set the dedicated input queue

      :param q_in: for input signals to this actor
      :type q_in: improv.nexus.Link


   .. py:method:: setLinkOut(q_out)

      Set the dedicated output queue

      :param q_out: for output signals from this actor
      :type q_out: improv.nexus.Link


   .. py:method:: setLinkWatch(q_watch)

      Set the dedicated watchout queue

      :param q_watch: watchout queue
      :type q_watch: improv.nexus.Link


   .. py:method:: addLink(name, link)

      Function provided to add additional data links by name
      using same form as q_in or q_out
      Must be done during registration and not during run

      :param name: customized link name
      :type name: string
      :param link: customized data link
      :type link: improv.nexus.Link


   .. py:method:: getLinks()

      Returns dictionary of links for the current actor

      :returns: dictionary of links
      :rtype: dict


   .. py:method:: put(idnames, q_out=None, save=None)

      TODO: This is deprecated? Prefer using Links explicitly


   .. py:method:: setup()

      Essenitally the registration process
      Can also be an initialization for the actor
      options is a list of options, can be empty


   .. py:method:: run()
      :abstractmethod:

      Must run in continuous mode
      Also must check q_sig either at top of a run-loop
      or as async with the primary function

      Suggested implementation for synchronous running: see RunManager class below


   .. py:method:: stop()

      Specify method for momentarily stopping the run and saving data.
      Not used by default


   .. py:method:: changePriority()

      Try to lower this process' priority
      Only changes priority if lower_priority is set
      TODO: Only works on unix machines. Add Windows functionality



.. py:class:: ManagedActor(*args, **kwargs)

   Bases: :py:obj:`AbstractActor`

   Base class for an actor that Nexus
   controls and interacts with.
   Needs to have a store and links for communication
   Also needs to be responsive to sent Signals (e.g. run, setup, etc)

   .. py:method:: run()

      Must run in continuous mode
      Also must check q_sig either at top of a run-loop
      or as async with the primary function

      Suggested implementation for synchronous running: see RunManager class below


   .. py:method:: runStep()
      :abstractmethod:



.. py:class:: AsyncActor(*args, **kwargs)

   Bases: :py:obj:`AbstractActor`

   Base class for an actor that Nexus
   controls and interacts with.
   Needs to have a store and links for communication
   Also needs to be responsive to sent Signals (e.g. run, setup, etc)

   .. py:method:: run()

      Run the actor in an async loop


   .. py:method:: setup()
      :async:

      Essenitally the registration process
      Can also be an initialization for the actor
      options is a list of options, can be empty


   .. py:method:: runStep()
      :abstractmethod:
      :async:


   .. py:method:: stop()
      :async:



.. py:data:: Actor

   

.. py:class:: RunManager(name, actions, links, runStoreInterface=None, timeout=1e-06)


.. py:class:: AsyncRunManager(name, actions, links, runStore=None, timeout=1e-06)

   Asynchronous run manager. Communicates with nexus core using q_sig and q_comm.
   To be used with [async with]
   Afterwards, the run manager listens for signals without blocking.

   .. py:method:: run_actor()
      :async:



.. py:class:: Signal

   Class containing definition of signals Nexus uses
   to communicate with its actors
   TODO: doc each of these with expected handling behavior

   .. py:method:: run()
      :staticmethod:


   .. py:method:: quit()
      :staticmethod:


   .. py:method:: pause()
      :staticmethod:


   .. py:method:: resume()
      :staticmethod:


   .. py:method:: reset()
      :staticmethod:


   .. py:method:: load()
      :staticmethod:


   .. py:method:: setup()
      :staticmethod:


   .. py:method:: ready()
      :staticmethod:


   .. py:method:: kill()
      :staticmethod:


   .. py:method:: revive()
      :staticmethod:


   .. py:method:: stop()
      :staticmethod:


   .. py:method:: stop_success()
      :staticmethod:




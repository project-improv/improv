import time
import signal
import asyncio
import traceback
from queue import Empty
from improv.store import StoreInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbstractActor:
    """Base class for an actor that Nexus
    controls and interacts with.
    Needs to have a store and links for communication
    Also needs to be responsive to sent Signals (e.g. run, setup, etc)
    """

    def __init__(self, name, store_loc, method="fork"):
        """Require a name for multiple instances of the same actor/class
        Create initial empty dict of Links for easier referencing
        """
        self.q_watchout = None
        self.name = name
        self.links = {}
        self.method = method
        self.client = None
        self.store_loc = store_loc
        self.lower_priority = False

        # Start with no explicit data queues.
        # q_in and q_out are reserved for passing ID information
        # to access data in the store
        self.q_in = None
        self.q_out = None

    def __repr__(self):
        """Internal representation of the Actor mostly for printing purposes.

        Returns:
            [str]: instance name and links dict
        """
        return self.name + ": " + str(self.links.keys())

    def setStoreInterface(self, client):
        """Sets the client interface to the store

        Args:
            client (improv.store.StoreInterface): Set client interface to the store
        """
        self.client = client

    def _getStoreInterface(self):
        # TODO: Where do we require this be run? Add a Signal and include in RM?
        if not self.client:
            store = StoreInterface(self.name, self.store_loc)
            self.setStoreInterface(store)

    def setLinks(self, links):
        """General full dict set for links

        Args:
            links (dict): The dict to store all the links
        """
        self.links = links

    def setCommLinks(self, q_comm, q_sig):
        """Set explicit communication links to/from Nexus (q_comm, q_sig)

        Args:
            q_comm (improv.nexus.Link): for messages from this actor to Nexus
            q_sig (improv.nexus.Link): signals from Nexus and must be checked first
        """
        self.q_comm = q_comm
        self.q_sig = q_sig
        self.links.update({"q_comm": self.q_comm, "q_sig": self.q_sig})

    def setLinkIn(self, q_in):
        """Set the dedicated input queue

        Args:
            q_in (improv.nexus.Link): for input signals to this actor
        """
        self.q_in = q_in
        self.links.update({"q_in": self.q_in})

    def setLinkOut(self, q_out):
        """Set the dedicated output queue

        Args:
            q_out (improv.nexus.Link): for output signals from this actor
        """
        self.q_out = q_out
        self.links.update({"q_out": self.q_out})

    def setLinkWatch(self, q_watch):
        """Set the dedicated watchout queue

        Args:
            q_watch (improv.nexus.Link): watchout queue
        """
        self.q_watchout = q_watch
        self.links.update({"q_watchout": self.q_watchout})

    def addLink(self, name, link):
        """Function provided to add additional data links by name
        using same form as q_in or q_out
        Must be done during registration and not during run

        Args:
            name (string): customized link name
            link (improv.nexus.Link): customized data link
        """
        self.links.update({name: link})
        # User can then use: self.my_queue = self.links['my_queue'] in a setup fcn,
        # or continue to reference it using self.links['my_queue']

    def getLinks(self):
        """Returns dictionary of links for the current actor

        Returns:
            dict: dictionary of links
        """
        return self.links

    def put(self, idnames, q_out=None, save=None):
        """TODO: This is deprecated? Prefer using Links explicitly"""
        if save is None:
            save = [False] * len(idnames)

        if len(save) < len(idnames):
            save = save + [False] * (len(idnames) - len(save))

        if q_out is None:
            q_out = self.q_out

        q_out.put(idnames)

        for i in range(len(idnames)):
            if save[i]:
                if self.q_watchout:
                    self.q_watchout.put(idnames[i])

    def setup(self):
        """Essenitally the registration process
        Can also be an initialization for the actor
        options is a list of options, can be empty
        """
        pass

    def run(self):
        """Must run in continuous mode
        Also must check q_sig either at top of a run-loop
        or as async with the primary function

        Suggested implementation for synchronous running: see RunManager class below
        """
        raise NotImplementedError

    def stop(self):
        """Specify method for momentarily stopping the run and saving data.
        Not used by default
        """
        pass

    def reset(self):
        """Specify method for resetting the current state of the Actor.
        Not used by default
        """

        pass

    def changePriority(self):
        """Try to lower this process' priority
        Only changes priority if lower_priority is set
        TODO: Only works on unix machines. Add Windows functionality
        """
        if self.lower_priority is True:
            import os
            import psutil

            p = psutil.Process(os.getpid())
            p.nice(19)  # lowest as default
            logger.info("Lowered priority of this process: {}".format(self.name))
            print("Lowered ", os.getpid(), " for ", self.name)


class ManagedActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions["setup"] = self.setup
        self.actions["run"] = self.runStep
        self.actions["stop"] = self.stop
        self.actions["reset"] = self.reset

    def run(self):
        with RunManager(self.name, self.actions, self.links):
            pass

    def runStep(self):
        raise NotImplementedError


class AsyncActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions["setup"] = self.setup
        self.actions["run"] = self.runStep
        self.actions["stop"] = self.stop

    def run(self):
        """Run the actor in an async loop"""
        result = asyncio.run(
            AsyncRunManager(self.name, self.actions, self.links).run_actor()
        )
        return result

    async def setup(self):
        """Essenitally the registration process
        Can also be an initialization for the actor
        options is a list of options, can be empty
        """
        pass

    async def runStep(self):
        raise NotImplementedError

    async def stop(self):
        pass


# Aliasing
Actor = ManagedActor


class RunManager:
    def __init__(self, name, actions, links, runStoreInterface=None, timeout=1e-6):
        self.run = False
        self.stop = False
        self.reset = False
        self.config = False

        self.actorName = name
        logger.debug("RunManager for {} created".format(self.actorName))

        self.actions = actions
        self.links = links
        self.q_sig = self.links["q_sig"]
        self.q_comm = self.links["q_comm"]

        self.runStoreInterface = runStoreInterface
        self.timeout = timeout

    def __enter__(self):
        self.start = time.time()
        an = self.actorName

        while True:
            # Run any actions given a received Signal
            if self.run:
                try:
                    self.actions["run"]()
                except Exception as e:
                    logger.error("Actor {} error in run: {}".format(an, e))
                    logger.error(traceback.format_exc())
            elif self.stop:
                try:
                    self.actions["stop"]()
                except Exception as e:
                    logger.error("Actor {} error in stop: {}".format(an, e))
                    logger.error(traceback.format_exc())
                self.stop = False  # Run once
            elif self.reset:
                try:
                    self.actions["reset"]()
                except Exception as e:
                    logger.error("Actor {} error in reset: {}".format(an, e))
                    logger.error(traceback.format_exc())
                self.reset = False  # Run once
            elif self.config:
                try:
                    if self.runStoreInterface:
                        self.runStoreInterface()
                    self.actions["setup"]()
                    self.q_comm.put([Signal.ready()])
                except Exception as e:
                    logger.error("Actor {} error in setup: {}".format(an, e))
                    logger.error(traceback.format_exc())
                self.config = False

            # Check for new Signals received from Nexus
            try:
                signal = self.q_sig.get(timeout=self.timeout)
                logger.debug("{} received Signal {}".format(self.actorName, signal))
                if signal == Signal.run():
                    self.run = True
                    logger.warning("Received run signal, begin running")
                elif signal == Signal.setup():
                    self.config = True
                elif signal == Signal.stop():
                    self.run = False
                    self.stop = True
                    logger.warning(f"actor {self.actorName} received stop signal")
                elif signal == Signal.reset():
                    self.run = False
                    self.reset = True
                    logger.warning(f"actor {self.actorName} received reset signal")
                elif signal == Signal.quit():
                    logger.warning("Received quit signal, aborting")
                    break
                elif signal == Signal.pause():
                    logger.warning("Received pause signal, pending...")
                    self.run = False
                elif signal == Signal.resume():  # currently treat as same as run
                    logger.warning("Received resume signal, resuming")
                    self.run = True
            except KeyboardInterrupt:
                break
            except Empty:
                pass  # No signal from Nexus

        return None

    def __exit__(self, type, value, traceback):
        logger.info("Ran for " + str(time.time() - self.start) + " seconds")
        logger.warning("Exiting RunManager")
        return None


class AsyncRunManager:
    """
    Asynchronous run manager. Communicates with nexus core using q_sig and q_comm.
    To be used with [async with]
    Afterwards, the run manager listens for signals without blocking.
    """

    def __init__(self, name, actions, links, runStore=None, timeout=1e-6):
        self.run = False
        self.config = False
        self.stop = False
        self.actorName = name
        logger.debug("AsyncRunManager for {} created".format(self.actorName))
        self.actions = actions
        self.links = links
        # q_sig, q_comm are AsyncQueue
        self.q_sig = self.links["q_sig"]
        self.q_comm = self.links["q_comm"]

        self.runStore = runStore
        self.timeout = timeout

        self.loop = asyncio.get_event_loop()
        self.start = time.time()

        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            self.loop.add_signal_handler(s, lambda s=s: self.loop.stop())

    async def run_actor(self):
        an = self.actorName
        while True:
            # Run any actions given a received Signal
            if self.run:
                try:
                    await self.actions["run"]()
                except Exception as e:
                    logger.error("Actor {} error in run: {}".format(an, e))
                    logger.error(traceback.format_exc())
            elif self.stop:
                try:
                    await self.actions["stop"]()
                except Exception as e:
                    logger.error("Actor {} error in stop: {}".format(an, e))
                    logger.error(traceback.format_exc())
                self.stop = False  # Run once
            elif self.config:
                try:
                    if self.runStore:
                        self.runStore()
                    await self.actions["setup"]()
                    self.q_comm.put([Signal.ready()])
                except Exception as e:
                    logger.error("Actor {} error in setup: {}".format(an, e))
                    logger.error(traceback.format_exc())
                self.config = False

            # Check for new Signals received from Nexus
            try:
                signal = self.q_sig.get(timeout=self.timeout)
                logger.debug("{} received Signal {}".format(self.actorName, signal))
                if signal == Signal.run():
                    self.run = True
                    logger.warning("Received run signal, begin running")
                elif signal == Signal.setup():
                    self.config = True
                elif signal == Signal.stop():
                    self.run = False
                    self.stop = True
                    logger.warning(f"actor {self.actorName} received stop signal")
                elif signal == Signal.quit():
                    logger.warning("Received quit signal, aborting")
                    break
                elif signal == Signal.pause():
                    logger.warning("Received pause signal, pending...")
                    self.run = False
                elif signal == Signal.resume():  # currently treat as same as run
                    logger.warning("Received resume signal, resuming")
                    self.run = True
            except KeyboardInterrupt:
                break
            except Empty:
                pass  # No signal from Nexus

        return None

    async def __aenter__(self):
        self.start = time.time()
        return self

    async def __aexit__(self, type, value, traceback):
        logger.info("Ran for {} seconds".format(time.time() - self.start))
        logger.warning("Exiting AsyncRunManager")
        return None



class Signal:
    """Class containing definition of signals Nexus uses
    to communicate with its actors
    TODO: doc each of these with expected handling behavior
    """

    @staticmethod
    def run():
        return "run"

    @staticmethod
    def quit():
        return "quit"

    @staticmethod
    def pause():
        return "pause"

    @staticmethod
    def resume():
        return "resume"

    @staticmethod
    def reset():  # TODO: implement in Nexus
        return "reset"

    @staticmethod
    def load():
        return "load"

    @staticmethod
    def setup():
        return "setup"

    @staticmethod
    def ready():
        return "ready"

    @staticmethod
    def kill():
        return "kill"

    @staticmethod
    def revive():
        return "revive"

    @staticmethod
    def stop():
        return "stop"

    @staticmethod
    def stop_success():
        return "stop success"

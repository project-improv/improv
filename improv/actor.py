import time
import asyncio
import traceback
from queue import Empty
from typing import Awaitable, Callable
from improv.store import Store

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbstractActor:
    """Base class for an actor that Nexus
    controls and interacts with.
    Needs to have a store and links for communication
    Also needs to be responsive to sent Signals (e.g. run, setup, etc)
    """

    def __init__(self, name, method='fork'):
        """Require a name for multiple instances of the same actor/class
        Create initial empty dict of Links for easier referencing
        """
        self.q_watchout = None
        self.name = name
        self.links = {}
        self.method = method
        self.client = None

        self.lower_priority = False

        # start with no explicit data queues.
        # q_in and q_out are for passing ID information to access data in the store
        self.q_in = None
        self.q_out = None

    def __repr__(self):
        """Internal representation of the Actor mostly for printing purposes.

        Returns:
            [str]: _description_
        """
        """ Return this instance name and links dict
        """
        return self.name + ': ' + str(self.links.keys())

    def setStore(self, client):
        """Sets the client interface to the store

        Args:
            client (improv.nexus.Link): _description_
        """
        """ Set client interface to the store
        """
        self.client = client

    def _getStoreInterface(self):
        ## TODO: Where do we require this be run? Add a Signal and include in RM?
        if not self.client:
            store = Store(self.name)
            self.setStore(store)

    def setLinks(self, links):
        """General full dict set for links"""
        self.links = links

    def setCommLinks(self, q_comm, q_sig):
        """Set explicit communication links to/from Nexus (q_comm, q_sig)
        q_comm is for messages from this actor to Nexus
        q_sig is signals from Nexus and must be checked first
        """
        self.q_comm = q_comm
        self.q_sig = q_sig
        self.links.update({'q_comm': self.q_comm, 'q_sig': self.q_sig})

    def setLinkIn(self, q_in):
        """Set the dedicated input queue"""
        self.q_in = q_in
        self.links.update({'q_in': self.q_in})

    def setLinkOut(self, q_out):
        """Set the dedicated output queue"""
        self.q_out = q_out
        self.links.update({'q_out': self.q_out})

    def setLinkWatch(self, q_watch):
        """ """
        self.q_watchout = q_watch
        self.links.update({'q_watchout': self.q_watchout})

    def addLink(self, name, link):
        """Function provided to add additional data links by name
        using same form as q_in or q_out
        Must be done during registration and not during run
        """
        self.links.update({name: link})
        # User can then use: self.my_queue = self.links['my_queue'] in a setup fcn,
        # or continue to reference it using self.links['my_queue']

    def getLinks(self):
        """Returns dictionary of links"""
        return self.links

    def put(self, idnames, q_out=None, save=None):
        if save == None:
            save = [False] * len(idnames)

        if len(save) < len(idnames):
            save = save + [False] * (len(idnames) - len(save))

        if q_out == None:
            q_out = self.q_out

        q_out.put(idnames)

        for i in range(len(idnames)):
            if save[i]:
                if self.q_watchout:
                    self.q_watchout.put(idnames[i])

    def run(self):
        """Must run in continuous mode
        Also must check q_sig either at top of a run-loop
        or as async with the primary function
        """
        raise NotImplementedError

        """ Suggested implementation for synchronous running: see RunManager class below
        """

    def stop(self):
        """Specify method for momentarily stopping the run and saving data."""
        pass

    def changePriority(self):
        """Try to lower this process' priority
        Only changes priority if lower_priority is set
        TODO: Only works on unix machines. Add Windows functionality
        """
        if self.lower_priority is True:
            import os, psutil

            p = psutil.Process(os.getpid())
            p.nice(19)  # lowest as default
            logger.info('Lowered priority of this process: {}'.format(self.name))
            print('Lowered ', os.getpid(), ' for ', self.name)


class ManagedActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions['setup'] = self.setup
        self.actions['run'] = self.runStep
        self.actions['stop'] = self.stop

    def run(self):
        with RunManager(self.name, self.actions, self.links) as rm:
            pass

    def setup(self):
        """Essenitally the registration process
        Can also be an initialization for the actor
        options is a list of options, can be empty
        """
        pass

    def runStep(self):
        raise NotImplementedError

    def stop(self):
        pass


class AsyncActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions['setup'] = self.setup
        self.actions['run'] = self.runStep
        self.actions['stop'] = self.stop

    def run(self):
        with AsyncRunManager(self.name, self.actions, self.links) as rm:
            pass

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


## Aliasing
Actor = ManagedActor


class RunManager:
    """ """

    def __init__(self, name, actions, links, runStore=None, timeout=1e-6):
        self.run = False
        self.stop = False
        self.config = False

        self.actorName = name
        logger.debug('RunManager for {} created'.format(self.actorName))

        self.actions = actions
        self.links = links
        self.q_sig = self.links['q_sig']
        self.q_comm = self.links['q_comm']

        self.runStore = runStore
        self.timeout = timeout

    def __enter__(self):
        self.start = time.time()

        while True:
            # Run any actions given a received Signal
            if self.run:
                try:
                    self.actions['run']()
                except Exception as e:
                    logger.error(
                        'Actor '
                        + self.actorName
                        + ' exception during run: {}'.format(e)
                    )
                    logger.error(traceback.format_exc())
            elif self.stop:
                try:
                    self.actions['stop']()
                except Exception as e:
                    logger.error(
                        'Actor '
                        + self.actorName
                        + ' exception during stop: {}'.format(e)
                    )
                    logger.error(traceback.format_exc())
                self.stop = False  # Run once
            elif self.config:
                try:
                    if self.runStore:
                        self.runStore()
                    self.actions['setup']()
                    self.q_comm.put([Signal.ready()])
                except Exception as e:
                    logger.error(
                        'Actor '
                        + self.actorName
                        + ' exception during setup: {}'.format(e)
                    )
                    logger.error(traceback.format_exc())
                self.config = False

            # Check for new Signals received from Nexus
            try:
                signal = self.q_sig.get(timeout=self.timeout)
                logger.debug('{} received Signal {}'.format(self.actorName, signal))
                if signal == Signal.run():
                    self.run = True
                    logger.warning('Received run signal, begin running')
                elif signal == Signal.setup():
                    self.config = True
                elif signal == Signal.stop():
                    self.run = False
                    self.stop = True
                    logger.warning(f"actor {self.actorName} received stop signal")
                elif signal == Signal.quit():
                    logger.warning('Received quit signal, aborting')
                    break
                elif signal == Signal.pause():
                    logger.warning('Received pause signal, pending...')
                    self.run = False
                elif signal == Signal.resume():  # currently treat as same as run
                    logger.warning('Received resume signal, resuming')
                    self.run = True
            except KeyboardInterrupt:
                break
            except Empty as e:
                pass  # No signal from Nexus

        return None

    def __exit__(self, type, value, traceback):
        logger.info('Ran for ' + str(time.time() - self.start) + ' seconds')
        logger.warning('Exiting RunManager')
        return None


class AsyncRunManager:
    """
    Asynchronous run manager. Communicates with nexus core using q_sig and q_comm.

    To be used with [async with]

    Afterwards, the run manager listens for signals without blocking.

    """

    def __init__(
        self, name, run_method: Callable[[], Awaitable[None]], setup, q_sig, q_comm
    ):  # q_sig, q_comm are AsyncQueue.
        self.run = False
        self.config = False
        self.run_method = run_method
        self.setup = setup
        self.q_sig = q_sig
        self.q_comm = q_comm
        self.module_name = name
        self.loop = asyncio.get_event_loop()
        self.start = time.time()

        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            self.loop.add_signal_handler(s, lambda s=s: self.loop.stop())

    async def __aenter__(self):
        while True:
            signal = await self.q_sig.get_async()
            if signal == Signal.run() or signal == Signal.resume():
                if not self.run:
                    self.run = True
                    asyncio.create_task(self.run_method(), loop=self.loop)
                    print('Received run signal, begin running')
            elif signal == Signal.setup():
                self.setup()
                await self.q_comm.put_async([Signal.ready()])
            elif signal == Signal.quit():
                logger.warning('Received quit signal, aborting')
                self.loop.stop()
                break
            elif signal == Signal.pause():
                logger.warning('Received pause signal, pending...')
                while self.q_sig.get() != Signal.resume():  # Intentionally blocking
                    time.sleep(1e-3)

    async def __aexit__(self, type, value, traceback):
        logger.info('Ran for {} seconds'.format(time.time() - self.start))
        logger.warning('Exiting AsyncRunManager')


class Signal:
    """Class containing definition of signals Nexus uses
    to communicate with its actors
    TODO: doc each of these with expected handling behavior
    """

    @staticmethod
    def run():
        return 'run'

    @staticmethod
    def quit():
        return 'quit'

    @staticmethod
    def pause():
        return 'pause'

    @staticmethod
    def resume():
        return 'resume'

    @staticmethod
    def reset():  # TODO: implement in Nexus
        return 'reset'

    @staticmethod
    def load():
        return 'load'

    @staticmethod
    def setup():
        return 'setup'

    @staticmethod
    def ready():
        return 'ready'

    @staticmethod
    def kill():
        return 'kill'

    @staticmethod
    def revive():
        return 'revive'

    @staticmethod
    def stop():
        return 'stop'

    @staticmethod
    def stop_success():
        return 'stop success'

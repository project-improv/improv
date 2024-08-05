from __future__ import annotations

import time
import signal
import asyncio
import traceback
from queue import Empty

import zmq
from zmq import SocketOption

from improv import log
from improv.link import ZmqLink
from improv.messaging import ActorStateMsg, ActorStateReplyMsg, ActorSignalReplyMsg
from improv.store import StoreInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO construct log handler within post-fork setup based on
# TODO arguments sent to actor by nexus
# TODO and add it below; fine to leave logger here as-is


class LinkInfo:
    def __init__(self, link_name, link_topic):
        self.link_name = link_name
        self.link_topic = link_topic


class AbstractActor:
    """Base class for an actor that Nexus
    controls and interacts with.
    Needs to have a store and links for communication
    Also needs to be responsive to sent Signals (e.g. run, setup, etc)
    """

    def __init__(self, name, method="fork", store_port_num=None, *args, **kwargs):
        """Require a name for multiple instances of the same actor/class
        Create initial empty dict of Links for easier referencing
        """
        self.q_watchout = None
        self.name = name
        self.links = {}
        self.method = method
        self.client = None
        self.lower_priority = False
        self.improv_logger = None
        self.store_port_num = store_port_num

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

    def set_store_interface(self, client):
        """Sets the client interface to the store

        Args:
            client (improv.store.StoreInterface): Set client interface to the store
        """
        self.client = client

    def _get_store_interface(self):
        # TODO: Where do we require this be run? Add a Signal and include in RM?
        if not self.client:
            store = None
            store = StoreInterface(self.name, self.store_port_num)
            self.set_store_interface(store)

    def set_links(self, links):
        """General full dict set for links

        Args:
            links (dict): The dict to store all the links
        """
        self.links = links

    def set_comm_links(self, q_comm, q_sig):
        """Set explicit communication links to/from Nexus (q_comm, q_sig)

        Args:
            q_comm (improv.nexus.Link): for messages from this actor to Nexus
            q_sig (improv.nexus.Link): signals from Nexus and must be checked first
        """
        self.q_comm = q_comm
        self.q_sig = q_sig
        self.links.update({"q_comm": self.q_comm, "q_sig": self.q_sig})

    def set_link_in(self, q_in):
        """Set the dedicated input queue

        Args:
            q_in (improv.nexus.Link): for input signals to this actor
        """
        self.q_in = q_in
        self.links.update({"q_in": self.q_in})

    def set_link_out(self, q_out):
        """Set the dedicated output queue

        Args:
            q_out (improv.nexus.Link): for output signals from this actor
        """
        self.q_out = q_out
        self.links.update({"q_out": self.q_out})

    def set_link_watch(self, q_watch):
        """Set the dedicated watchout queue

        Args:
            q_watch (improv.nexus.Link): watchout queue
        """
        self.q_watchout = q_watch
        self.links.update({"q_watchout": self.q_watchout})

    def add_link(self, name, link):
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

    def get_links(self):
        """Returns dictionary of links for the current actor

        Returns:
            dict: dictionary of links
        """
        return self.links

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

    def change_priority(self):
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

    def register_with_nexus(self):
        pass

    def register_with_broker(self):
        pass

    def setup_links(self):
        pass

    def setup_logging(self):
        raise NotImplementedError


class ManagedActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions["setup"] = self.setup
        self.actions["run"] = self.run_step
        self.actions["stop"] = self.stop
        self.nexus_sig_port: int = None

    def run(self):
        self.register_with_nexus()
        self.setup_logging()
        self.register_with_broker()
        self.setup_links()
        with RunManager(
            self.name, self.actions, self.links, self.nexus_sig_port, self.improv_logger
        ):
            pass

    def run_step(self):
        raise NotImplementedError


class ZmqActor(ManagedActor):
    def __init__(
        self,
        nexus_comm_port,
        broker_sub_port,
        broker_pub_port,
        log_pull_port,
        outgoing_links,
        incoming_links,
        broker_host="localhost",
        log_host="localhost",
        nexus_host="localhost",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.broker_pub_socket: zmq.Socket | None = None
        self.zmq_context: zmq.Context | None = None
        self.nexus_host: str = nexus_host
        self.nexus_comm_port: int = nexus_comm_port
        self.nexus_sig_port: int | None = None
        self.nexus_comm_socket: zmq.Socket | None = None
        self.nexus_sig_socket: zmq.Socket | None = None
        self.broker_sub_port: int = broker_sub_port
        self.broker_pub_port: int = broker_pub_port
        self.broker_host: str = broker_host
        self.log_pull_port: int = log_pull_port
        self.log_host: str = log_host
        self.outgoing_links: list[LinkInfo] = outgoing_links
        self.incoming_links: list[LinkInfo] = incoming_links
        self.incoming_sockets: dict[str, zmq.Socket] = dict()

        # Redefine dictionary of actions for the RunManager
        self.actions = {"setup": self.setup, "run": self.run_step, "stop": self.stop}

    def register_with_nexus(self):
        logger.info(f"Actor {self.name} registering with nexus")
        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(SocketOption.LINGER, 0)
        # create a REQ socket pointed at nexus' global actor in port
        self.nexus_comm_socket = self.zmq_context.socket(zmq.REQ)
        self.nexus_comm_socket.connect(
            f"tcp://{self.nexus_host}:{self.nexus_comm_port}"
        )

        # create a REP socket for comms from Nexus and save its state
        self.nexus_sig_socket = self.zmq_context.socket(zmq.REP)
        self.nexus_sig_socket.bind("tcp://*:0")  # find any available port
        sig_socket_addr = self.nexus_sig_socket.getsockopt_string(
            SocketOption.LAST_ENDPOINT
        )
        self.nexus_sig_port = int(sig_socket_addr.split(":")[-1])

        # build and send a message to nexus
        actor_state = ActorStateMsg(
            self.name,
            Signal.waiting(),
            self.nexus_sig_port,
            "comms opened and actor ready to initialize",
        )

        self.nexus_comm_socket.send_pyobj(actor_state)

        rep: ActorStateReplyMsg = self.nexus_comm_socket.recv_pyobj()
        logger.info(
            f"Got response from nexus:\n"
            f"Status: {rep.status}\n"
            f"Info: {rep.info}\n"
        )

        self.links["q_comm"] = ZmqLink(self.nexus_comm_socket, f"{self.name}.q_comm")
        self.links["q_sig"] = ZmqLink(self.nexus_sig_socket, f"{self.name}.q_sig")

        logger.info(f"Actor {self.name} registered with Nexus")

    def register_with_broker(self):  # really opening sockets here
        self.improv_logger.info(f"Actor {self.name} registering with broker")
        if len(self.outgoing_links) > 0:
            self.broker_pub_socket = self.zmq_context.socket(zmq.PUB)
            self.broker_pub_socket.connect(
                f"tcp://{self.broker_host}:{self.broker_sub_port}"
            )

        for incoming_link in self.incoming_links:
            new_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            new_socket.connect(f"tcp://{self.broker_host}:{self.broker_pub_port}")
            new_socket.subscribe(incoming_link.link_topic)
            self.incoming_sockets[incoming_link.link_name] = new_socket
        self.improv_logger.info(f"Actor {self.name} registered with broker")

    def setup_links(self):
        self.improv_logger.info(f"Actor {self.name} setting up links")
        for outgoing_link in self.outgoing_links:
            self.links[outgoing_link.link_name] = ZmqLink(
                self.broker_pub_socket,
                outgoing_link.link_name,
                outgoing_link.link_topic,
            )

        for incoming_link in self.incoming_links:
            self.links[incoming_link.link_name] = ZmqLink(
                self.incoming_sockets[incoming_link.link_name],
                incoming_link.link_name,
                incoming_link.link_topic,
            )

        if "q_out" in self.links.keys():
            self.q_out = self.links["q_out"]
        if "q_in" in self.links.keys():
            self.q_in = self.links["q_in"]
        self.improv_logger.info(f"Actor {self.name} finished setting up links")

    def setup_logging(self):
        self.improv_logger = logging.getLogger(self.name)
        self.improv_logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            self.improv_logger.addHandler(handler)
        self.improv_logger.addHandler(
            log.ZmqLogHandler(self.log_host, self.log_pull_port, self.zmq_context)
        )


class AsyncActor(AbstractActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # Define dictionary of actions for the RunManager
        self.actions = {}
        self.actions["setup"] = self.setup
        self.actions["run"] = self.run_step
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

    async def run_step(self):
        raise NotImplementedError

    async def stop(self):
        pass


# Aliasing
Actor = ManagedActor


class RunManager:
    def __init__(
        self,
        name,
        actions,
        links,
        nexus_sig_port,
        improv_logger,
        runStoreInterface=None,
        timeout=1e-6,
    ):
        self.run = False
        self.stop = False
        self.config = False
        self.nexus_sig_port = nexus_sig_port
        self.improv_logger = improv_logger

        self.actorName = name
        self.improv_logger.debug("RunManager for {} created".format(self.actorName))

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
                    self.improv_logger.error("Actor {} error in run: {}".format(an, e))
                    self.improv_logger.error(traceback.format_exc())
            elif self.stop:
                try:
                    self.actions["stop"]()
                except Exception as e:
                    self.improv_logger.error("Actor {} error in stop: {}".format(an, e))
                    self.improv_logger.error(traceback.format_exc())
                self.stop = False  # Run once
            elif self.config:
                try:
                    if self.runStoreInterface:
                        self.runStoreInterface()
                    self.actions["setup"]()
                    self.q_comm.put(
                        ActorStateMsg(
                            self.actorName, Signal.ready(), self.nexus_sig_port, ""
                        )
                    )
                    res = self.q_comm.get()
                    self.improv_logger.info(
                        f"Actor {res.actor_name} got state update reply:\n"
                        f"Status: {res.status}\n"
                        f"Info: {res.info}\n"
                    )
                except Exception as e:
                    self.improv_logger.error(
                        "Actor {} error in setup: {}".format(an, e)
                    )
                    self.improv_logger.error(traceback.format_exc())
                self.config = False

            # Check for new Signals received from Nexus
            try:
                signal_msg = self.q_sig.get(timeout=self.timeout)
                signal = signal_msg.signal
                self.q_sig.put(ActorSignalReplyMsg(an, signal, "OK", ""))
                self.improv_logger.warning(
                    "{} received Signal {}".format(self.actorName, signal)
                )
                if signal == Signal.run():
                    self.run = True
                    self.improv_logger.warning("Received run signal, begin running")
                elif signal == Signal.setup():
                    self.config = True
                elif signal == Signal.stop():
                    self.run = False
                    self.stop = True
                    self.improv_logger.warning(
                        f"actor {self.actorName} received stop signal"
                    )
                elif signal == Signal.quit():
                    self.improv_logger.warning("Received quit signal, aborting")
                    break
                elif signal == Signal.pause():
                    self.improv_logger.warning("Received pause signal, pending...")
                    self.run = False
                elif signal == Signal.resume():  # currently treat as same as run
                    self.improv_logger.warning("Received resume signal, resuming")
                    self.run = True
                elif signal == Signal.status():
                    self.improv_logger.info(
                        f"Actor {self.actorName} received status request"
                    )
            except KeyboardInterrupt:
                break
            except Empty:
                pass  # No signal from Nexus
            except TimeoutError:
                pass  # No signal from Nexus over zmq

        return None

    def __exit__(self, type, value, traceback):
        self.improv_logger.info("Ran for " + str(time.time() - self.start) + " seconds")
        self.improv_logger.warning("Exiting RunManager")
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

    @staticmethod
    def status():
        return "status"

    @staticmethod
    def waiting():
        return "waiting"

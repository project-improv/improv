from __future__ import annotations

import multiprocessing
import time
import uuid
import signal
import logging
import asyncio
import concurrent
import subprocess

from datetime import datetime
from multiprocessing import get_context
from importlib import import_module

import zmq as zmq_sync
import zmq.asyncio as zmq
from zmq import PUB, REP, REQ, SocketOption

from improv import log
from improv.broker import bootstrap_broker
from improv.harvester import bootstrap_harvester
from improv.log import bootstrap_log_server
from improv.messaging import (
    ActorStateMsg,
    ActorStateReplyMsg,
    ActorSignalMsg,
    BrokerInfoReplyMsg,
    BrokerInfoMsg,
    LogInfoMsg,
    LogInfoReplyMsg,
    HarvesterInfoMsg,
    HarvesterInfoReplyMsg,
)
from improv.store import StoreInterface, RedisStoreInterface
from improv.actor import Signal, Actor, LinkInfo
from improv.config import Config

ASYNC_DEBUG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.level == logging.DEBUG:
    logger.addHandler(logging.StreamHandler())

# TODO: redo docsctrings since things are pretty different now

# TODO: socket setup can fail - need to check it

# TODO: redo how actors register with nexus so that we get actor states
#   earlier


class ConfigFileNotProvidedException(Exception):
    def __init__(self):
        super().__init__("Config file not provided")


class ConfigFileNotValidException(Exception):
    def __init__(self):
        super().__init__("Config file not valid")


class ActorState:
    def __init__(
        self, actor_name, status, nexus_in_port, hostname="localhost", sig_socket=None
    ):
        self.actor_name = actor_name
        self.status = status
        self.nexus_in_port = nexus_in_port
        self.hostname = hostname
        self.sig_socket = None


class Nexus:
    """Main server class for handling objects in improv"""

    def __init__(self, name="Server"):
        self.logger_in_port: int | None = None
        self.zmq_sync_context: zmq_sync.Context | None = None
        self.logfile: str | None = None
        self.p_harvester: multiprocessing.Process | None = None
        self.p_broker: multiprocessing.Process | None = None
        self.actor_in_socket_port: int | None = None
        self.actor_in_socket: zmq.Socket | None = None
        self.in_socket: zmq.Socket | None = None
        self.out_socket: zmq.Socket | None = None
        self.zmq_context: zmq.Context | None = None
        self.logger_pub_port: int | None = None
        self.logger_pull_port: int | None = None
        self.logger_in_socket: zmq.Socket | None = None
        self.p_logger: multiprocessing.Process | None = None
        self.broker_pub_port = None
        self.broker_sub_port = None
        self.broker_in_port: int | None = None
        self.broker_in_socket: zmq_sync.Socket | None = None
        self.actor_states: dict[str, ActorState | None] = dict()
        self.redis_fsync_frequency = None
        self.store = None
        self.config = None
        self.name = name
        self.aof_dir = None
        self.redis_saving_enabled = False
        self.allow_setup = False
        self.outgoing_topics = dict()
        self.incoming_topics = dict()
        self.data_queues = {}
        self.actors = {}
        self.flags = {}
        self.processes: list[multiprocessing.Process] = []

    def __str__(self):
        return self.name

    def create_nexus(
        self,
        file=None,
        store_size=None,
        control_port=None,
        output_port=None,
        log_server_pub_port=None,
        actor_in_port=None,
        logfile="global.log",
    ):
        """Function to initialize class variables based on config file.

        Starts a store of class Limbo, and then loads the config file.
        The config file specifies the specific actors that nexus will
        be connected to, as well as their links.

        Args:
            file (string): Name of the config file.
            store_size (int): initial store size
            control_port (int): port number for input socket
            output_port (int): port number for output socket
            actor_in_port (int): port number for the socket which receives
                                 actor communications

        Returns:
            string: "Shutting down", to notify start() that pollQueues has completed.
        """

        self.logfile = logfile

        curr_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"************ new improv server session {curr_dt} ************")

        if file is None:
            logger.exception("Need a config file!")
            raise ConfigFileNotProvidedException

        logger.info(f"Loading configuration file {file}:")
        self.config = Config(config_file=file)
        self.config.parse_config()

        with open(file, "r") as f:  # write config file to log
            logger.info(f.read())

        logger.debug("Applying CLI parameter configuration overrides")
        self.apply_cli_config_overrides(
            store_size=store_size,
            control_port=control_port,
            output_port=output_port,
            actor_in_port=actor_in_port,
        )

        logger.debug("Setting up sockets")
        self.set_up_sockets(actor_in_port=self.config.settings["actor_in_port"])

        logger.debug("Setting up services")
        self.start_improv_services(
            log_server_pub_port=log_server_pub_port,
            store_size=self.config.settings["store_size"],
        )

        logger.debug("initializing config")
        self.init_config()

        self.flags.update({"quit": False, "run": False, "load": False})
        self.allowStart = False
        self.stopped = False

        logger.info(
            f"control: {self.config.settings['control_port']},"
            f" output: {self.config.settings['output_port']},"
            f" logging: {self.logger_pub_port}"
        )
        return (
            self.config.settings["control_port"],
            self.config.settings["output_port"],
            self.logger_pub_port,
        )

    def init_config(self):
        """For each connection:
        create a Link with a name (purpose), start, and end
        Start links to one actor's name, end to the other.
        Nexus gives start_actor the Link as a q_in,
        and end_actor the Link as a q_out.
        Nexus maintains dict of name and associated Link.
        Nexus also has list of Links that it is itself connected to
        for communication purposes.

        OR
        For each connection, create 2 Links. Nexus acts as intermediary.

        Args:
            file (string): input config filepath
        """
        # TODO load from file or user input, as in dialogue through FrontEnd?

        logger.info("initializing config")
        flag = self.config.create_config()
        if flag == -1:
            logger.error(
                "An error occurred when loading the configuration file. "
                "Please see the log file for more details."
            )
            self.destroy_nexus()
            raise ConfigFileNotValidException

        # create all data links requested from Config config
        self.create_connections()

        # if self.config.hasGUI:
        #     # Have to load GUI first (at least with Caiman)
        #     name = self.config.gui.name
        #     m = self.config.gui  # m is ConfigModule
        #     # treat GUI uniquely since user communication comes from here
        #     try:
        #         visualClass = m.options["visual"]
        #         # need to instantiate this actor
        #         visualActor = self.config.actors[visualClass]
        #         self.create_actor(visualClass, visualActor)
        #         # then add links for visual
        #         for k, l in {
        #             key: self.data_queues[key]
        #             for key in self.data_queues.keys()
        #             if visualClass in key
        #         }.items():
        #             self.assign_link(k, l)
        #
        #         # then give it to our GUI
        #         self.create_actor(name, m)
        #         self.actors[name].setup(visual=self.actors[visualClass])
        #
        #         self.p_GUI = Process(target=self.actors[name].run, name=name)
        #         self.p_GUI.daemon = True
        #         self.p_GUI.start()
        #
        #     except Exception as e:
        #         logger.error(f"Exception in setting up GUI {name}: {e}")

        # First set up each class/actor
        for name, actor in self.config.actors.items():
            if name not in self.actors.keys():
                # Check for actors being instantiated twice
                try:
                    self.create_actor(name, actor)
                    logger.info(f"Setting up actor {name}")
                except Exception as e:
                    logger.error(f"Exception in setting up actor {name}: {e}.")
                    self.quit()
                    raise e

        # Second set up each connection b/t actors
        # TODO: error handling for if a user tries to use q_in without defining it
        # for name, link in self.data_queues.items():
        #     self.assign_link(name, link)

    def configure_redis_persistence(self):
        # invalid configs: specifying filename and using an ephemeral filename,
        # specifying that saving is off but providing either filename option
        aof_dirname = self.config.redis_config["aof_dirname"]
        generate_unique_dirname = self.config.redis_config[
            "generate_ephemeral_aof_dirname"
        ]
        self.redis_saving_enabled = self.config.redis_config["enable_saving"]
        self.redis_fsync_frequency = self.config.redis_config["fsync_frequency"]

        if aof_dirname:
            self.aof_dir = aof_dirname
        elif generate_unique_dirname:
            self.aof_dir = "improv_persistence_" + str(uuid.uuid1())

        if self.redis_saving_enabled and self.aof_dir is not None:
            logger.info(
                "Redis saving enabled. Saving to directory "
                + self.aof_dir
                + " on schedule "
                + "'{}'".format(self.redis_fsync_frequency)
            )
        elif self.redis_saving_enabled:
            logger.info(
                "Redis saving enabled with default directory "
                + "on schedule "
                + "'{}'".format(self.redis_fsync_frequency)
            )
        else:
            logger.info("Redis saving disabled.")

        return

    def start_nexus(self, serve_function, *args, **kwargs):
        """
        Puts all actors in separate processes and begins polling
        to listen to comm queues
        """
        for name, m in self.actors.items():
            if "GUI" not in name:  # GUI already started
                if "method" in self.config.actors[name].options:
                    meth = self.config.actors[name].options["method"]
                    logger.info("This actor wants: {}".format(meth))
                    ctx = get_context(meth)
                    p = ctx.Process(target=m.run, name=name)
                else:
                    ctx = get_context("fork")
                    p = ctx.Process(target=self.run_actor, name=name, args=(m,))
                    if "daemon" in self.config.actors[name].options:
                        p.daemon = self.config.actors[name].options["daemon"]
                        logger.info("Setting daemon for {}".format(name))
                    else:
                        p.daemon = True  # default behavior
                self.processes.append(p)

        self.start()

        loop = asyncio.get_event_loop()
        res = ""
        try:
            self.out_socket.send_string("Awaiting input:")
            res = loop.run_until_complete(self.serve(serve_function, *args, **kwargs))
        except asyncio.CancelledError:
            logger.info("Loop is cancelled")

        try:
            logger.info(f"Result of run_until_complete: {res}")
        except Exception as e:
            logger.info(f"Res failed to await: {e}")

        logger.info(f"Current loop: {asyncio.get_event_loop()}")

        # loop.stop()
        # loop.close()
        logger.info("Shutdown loop")

    def start(self):
        """
        Start all the processes in Nexus
        """
        logger.info("Starting processes")

        for p in self.processes:
            logger.info(str(p))
            p.start()

        logger.info("All processes started")

    def destroy_nexus(self):
        """Method that calls the internal method
        to kill the processes running the store
        and the message broker
        """
        logger.warning("Destroying Nexus")
        self._shutdown_harvester()
        self._close_store_interface()

        if self.out_socket:
            self.out_socket.close(linger=0)
        if self.in_socket:
            self.in_socket.close(linger=0)
        if self.actor_in_socket:
            self.actor_in_socket.close(linger=0)
        if self.broker_in_socket:
            self.broker_in_socket.close(linger=0)
        if self.logger_in_socket:
            self.logger_in_socket.close(linger=0)

        self._shutdown_broker()
        self._shutdown_logger()

        for handler in logger.handlers:
            # need to close this one since we're about to destroy the zmq context
            if isinstance(handler, log.ZmqLogHandler):
                handler.close()
                logger.removeHandler(handler)

        if self.zmq_context:
            self.zmq_context.destroy(linger=0)
        if self.zmq_sync_context:
            self.zmq_sync_context.destroy(linger=0)

    async def poll_queues(self, poll_function, *args, **kwargs):
        """
        Listens to links and processes their signals.

        For every communications queue connected to Nexus, a task is
        created that gets from the queue. Throughout runtime, when these
        queues output a signal, they are processed by other functions.
        At the end of runtime (when the gui has been closed), polling is
        stopped.

        Returns:
            string: "Shutting down", Notifies start() that pollQueues has completed.
        """
        self.actorStates = dict.fromkeys(self.actors.keys())
        self.actor_states = dict.fromkeys(self.actors.keys(), None)
        if not self.config.hasGUI:
            # Since Visual is not started, it cannot send a ready signal.
            try:
                del self.actorStates["Visual"]
            except Exception as e:
                logger.info("Visual is not started: {0}".format(e))
                pass

        self.tasks = []
        self.tasks.append(asyncio.create_task(self.process_actor_message()))
        self.tasks.append(asyncio.create_task(self.remote_input()))

        self.early_exit = False

        # add signal handlers
        loop = asyncio.get_event_loop()
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(self.stop_polling_and_quit(s))
            )

        logger.info("Nexus signal handler added")

        while not self.flags["quit"]:
            await poll_function(*args, **kwargs)

        return "Shutting Down"

    async def stop_polling_and_quit(self, signal):
        """
        quit the process and stop polling signals from queues

        Args:
            signal (signal): Signal for handling async polling.
                             One of: signal.SIGHUP, signal.SIGTERM, signal.SIGINT
            queues (improv.link.AsyncQueue): Comm queues for links.
        """
        logger.warning(
            "Shutting down via signal handler due to {}. \
                Steps may be out of order or dirty.".format(
                signal
            )
        )
        await self.stop_polling(signal)
        logger.info("Nexus waiting for async tasks to have a chance to send")
        await asyncio.sleep(0)
        self.flags["quit"] = True
        self.early_exit = True
        self.quit()

    def process_actor_state_update(self, msg: ActorStateMsg):
        actor_state = None
        if msg.actor_name in self.actor_states.keys():
            actor_state = self.actor_states[msg.actor_name]
        if not actor_state:
            logger.info(
                f"Received state message from new actor {msg.actor_name}"
                f" with info: {msg.info}\n"
            )
            self.actor_states[msg.actor_name] = ActorState(
                msg.actor_name, msg.status, msg.nexus_in_port
            )
            actor_state = self.actor_states[msg.actor_name]
        else:
            logger.info(
                f"Received state message from actor {msg.actor_name}"
                f" with info: {msg.info}\n"
                f"Current state:\n"
                f"Name: {actor_state.actor_name}\n"
                f"Status: {actor_state.status}\n"
                f"Nexus control port: {actor_state.nexus_in_port}\n"
            )
            if msg.nexus_in_port != actor_state.nexus_in_port:
                pass
                # TODO: this actor's signal socket changed

            actor_state.actor_name = msg.actor_name
            actor_state.status = msg.status
            actor_state.nexus_in_port = msg.nexus_in_port

        logger.info(
            "Updated actor state:\n"
            f"Name: {actor_state.actor_name}\n"
            f"Status: {actor_state.status}\n"
            f"Nexus control port: {actor_state.nexus_in_port}\n"
        )

        if all(
            [
                actor_state is not None and actor_state.status == Signal.ready()
                for actor_state in self.actor_states.values()
            ]
        ):
            self.allowStart = True

        return True

    async def process_actor_message(self):
        msg = await self.actor_in_socket.recv_pyobj()
        if isinstance(msg, ActorStateMsg):
            if self.process_actor_state_update(msg):
                await self.actor_in_socket.send_pyobj(
                    ActorStateReplyMsg(
                        msg.actor_name, "OK", "actor state updated successfully"
                    )
                )
            else:
                await self.actor_in_socket.send_pyobj(
                    ActorStateReplyMsg(
                        msg.actor_name, "ERROR", "actor state update failed"
                    )
                )
            if (not self.allow_setup) and all(
                [
                    actor_state is not None and actor_state.status == Signal.waiting()
                    for actor_state in self.actor_states.values()
                ]
            ):
                logger.info("All actors connected to Nexus. Allowing setup.")
                self.allow_setup = True

            if (not self.allowStart) and all(
                [
                    actor_state is not None and actor_state.status == Signal.ready()
                    for actor_state in self.actor_states.values()
                ]
            ):
                logger.info("All actors ready. Allowing run.")
                self.allowStart = True

    async def remote_input(self):
        msg = await self.in_socket.recv_multipart()
        command = msg[0].decode("utf-8")
        await self.in_socket.send_string("Awaiting input:")
        if command == Signal.quit():
            await self.out_socket.send_string("QUIT")
        await self.process_gui_signal([command], "TUI_Nexus")

    async def process_gui_signal(self, flag, name):
        """Receive flags from the Front End as user input"""
        name = name.split("_")[0]
        if flag:
            logger.info("Received signal from user: " + flag[0])
            if flag[0] == Signal.run():
                logger.info("Begin run!")
                # self.flags['run'] = True
                await self.run()
            elif flag[0] == Signal.setup():
                logger.info("Running setup")
                await self.setup()
            elif flag[0] == Signal.ready():
                logger.info("GUI ready")
                self.actorStates[name] = flag[0]
            elif flag[0] == Signal.quit():
                logger.warning("Quitting the program!")
                task = asyncio.create_task(self.stop_polling_and_quit(Signal.quit()))
                done, pending = await asyncio.wait(task)
                while len(done) == 0:
                    done, pending = await asyncio.wait(task)
                self.flags["quit"] = True
            elif flag[0] == Signal.load():
                logger.info("Loading Config config from file " + flag[1])
                self.config = Config(flag[1])
                self.config.parse_config()
            elif flag[0] == Signal.pause():
                logger.info("Pausing processes")
                # TODO. Also resume, reset

            # temporary WiP
            elif flag[0] == Signal.kill():
                # TODO: specify actor to kill
                list(self.processes)[0].kill()
            elif flag[0] == Signal.revive():
                dead = [p for p in list(self.processes) if p.exitcode is not None]
                for pro in dead:
                    name = pro.name
                    m = self.actors[pro.name]
                    actor = self.config.actors[name]
                    if "GUI" not in name:  # GUI hard to revive independently
                        if "method" in actor.options:
                            meth = actor.options["method"]
                            logger.info("This actor wants: {}".format(meth))
                            ctx = get_context(meth)
                            p = ctx.Process(target=m.run, name=name)
                        else:
                            ctx = get_context("fork")
                            p = ctx.Process(target=self.run_actor, name=name, args=(m,))
                            if "daemon" in actor.options:
                                p.daemon = actor.options["daemon"]
                                logger.info("Setting daemon for {}".format(name))
                            else:
                                p.daemon = True

                    # Setting the stores for each actor to be the same
                    # TODO: test if this works for fork -- don't think it does?
                    al = [act for act in self.actors.values() if act.name != pro.name]
                    m.set_store_interface(al[0].client)
                    m.client = None
                    m._get_store_interface()

                    self.processes.append(p)
                    p.start()
                    m.q_sig.put_nowait(Signal.setup())
                    # TODO: ensure waiting for ready before run?
                    m.q_sig.put_nowait(Signal.run())

                self.processes = [p for p in list(self.processes) if p.exitcode is None]
            elif flag[0] == Signal.stop():
                logger.info("Nexus received stop signal")
                await self.stop()
        elif flag:
            logger.error("Unknown signal received from Nexus: {}".format(flag))

    async def setup(self):
        if not self.allow_setup:
            logger.error(
                "Not all actors connected to Nexus. Please wait, then try again."
            )
            return

        for actor in self.actor_states.values():
            logger.info("Starting setup: " + str(actor.actor_name))
            actor.sig_socket = self.zmq_context.socket(REQ)
            actor.sig_socket.connect(f"tcp://{actor.hostname}:{actor.nexus_in_port}")
            await actor.sig_socket.send_pyobj(
                ActorSignalMsg(actor.actor_name, Signal.setup(), "")
            )
            await actor.sig_socket.recv_pyobj()

    async def run(self):
        if self.allowStart:
            for actor in self.actor_states.values():
                await actor.sig_socket.send_pyobj(
                    ActorSignalMsg(actor.actor_name, Signal.run(), "")
                )
                await actor.sig_socket.recv_pyobj()
        else:
            logger.error("Not all actors ready yet, please wait and then try again.")

    def quit(self):
        logger.warning("Killing child processes")
        self.out_socket.send_string("QUIT")

        if self.config.hasGUI:
            self.processes.append(self.p_GUI)

        for p in self.processes:
            p.terminate()
            p.join(timeout=5)
            if p.exitcode is None:
                p.kill()
                logger.error("Process did not exit in time. Kill signal sent.")

        logger.warning("Actors terminated")

        self.destroy_nexus()

    async def stop(self):
        logger.warning("Starting stop procedure")
        self.allowStart = False

        for actor in self.actor_states.values():
            try:
                await actor.sig_socket.send_pyobj(
                    ActorSignalMsg(
                        actor.actor_name, Signal.stop(), "Nexus sending stop signal"
                    )
                )
                msg_ready = await actor.sig_socket.poll(timeout=1000)
                if msg_ready == 0:
                    raise TimeoutError
                await actor.sig_socket.recv_pyobj()
            except TimeoutError:
                logger.info(
                    f"Timed out waiting for reply to stop message "
                    f"from actor {actor.actor_name}. "
                    f"Closing connection."
                )
                actor.sig_socket.close(linger=0)
            except Exception as e:
                logger.info(
                    f"Unable to send stop message "
                    f"to actor {actor.actor_name}: "
                    f"{e}"
                )
        self.allowStart = True

    def revive(self):
        logger.warning("Starting revive")

    async def stop_polling(self, stop_signal):
        """Cancels outstanding tasks and fills their last request.

        Puts a string into all active queues, then cancels their
        corresponding tasks. These tasks are not fully cancelled until
        the next run of the event loop.

        Args:
            stop_signal (improv.actor.Signal): Signal for signal handler.
            queues (improv.link.AsyncQueue): Comm queues for links.
        """
        logger.info("Received shutdown order")

        logger.info(f"Stop signal: {stop_signal}")
        shutdown_message = Signal.quit()
        for actor in self.actor_states.values():
            try:
                await actor.sig_socket.send_pyobj(
                    ActorSignalMsg(
                        actor.actor_name, shutdown_message, "Nexus sending quit signal"
                    )
                )
                msg_ready = await actor.sig_socket.poll(timeout=1000)
                if msg_ready == 0:
                    raise TimeoutError
                await actor.sig_socket.recv_pyobj()
            except TimeoutError:
                logger.info(
                    f"Timed out waiting for reply to quit message "
                    f"from actor {actor.actor_name}. "
                    f"Closing connection."
                )
                actor.sig_socket.close(linger=0)
            except Exception as e:
                logger.info(
                    f"Unable to send shutdown message "
                    f"to actor {actor.actor_name}: "
                    f"{e}"
                )

        logger.info("Canceling outstanding tasks")

        [task.cancel() for task in self.tasks]

        logger.info("Polling has stopped.")

    def create_store_interface(self):
        """Creates StoreInterface"""
        return RedisStoreInterface(server_port_num=self.store_port)

    def _start_store_interface(self, size, attempts=20):
        """Start a subprocess that runs the redis store
        Raises a RuntimeError exception if size is undefined
        Raises an Exception if the redis store doesn't start

        Args:
            size: in bytes

        Raises:
            RuntimeError: if the size is undefined
            Exception: if the store doesn't start

        """
        if size is None:
            raise RuntimeError("Server size needs to be specified")

        logger.info("Setting up Redis store.")
        self.store_port = self.config.redis_config["port"] if self.config else 6379
        logger.info("Searching for open port starting at specified port.")
        for attempt in range(attempts):
            logger.info(
                "Attempting to connect to Redis on port {}".format(self.store_port)
            )
            # try with failure, incrementing port number
            self.p_StoreInterface = self.start_redis(size)
            time.sleep(1)
            if self.p_StoreInterface.poll():  # Redis could not start
                logger.info("Could not connect to port {}".format(self.store_port))
                self.store_port = str(int(self.store_port) + 1)
            else:
                break
        else:
            logger.error("Could not start Redis on any tried port.")
            raise Exception("Could not start Redis on any tried ports.")

        logger.info(f"StoreInterface start successful on port {self.store_port}")

    def start_redis(self, size):
        subprocess_command = [
            "redis-server",
            "--port",
            str(self.store_port),
            "--maxmemory",
            str(size),
            "--save",  # this only turns off RDB, which we want permanently off
            '""',
        ]

        if self.aof_dir is not None and len(self.aof_dir) == 0:
            raise Exception("Persistence directory specified but no filename given.")

        if self.aof_dir is not None:  # use specified (possibly pre-existing) file
            # subprocess_command += ["--save", "1 1"]
            subprocess_command += [
                "--appendonly",
                "yes",
                "--appendfsync",
                self.redis_fsync_frequency,
                "--appenddirname",
                self.aof_dir,
            ]
            logger.info("Redis persistence directory set to {}".format(self.aof_dir))
        elif (
            self.redis_saving_enabled
        ):  # just use the (possibly preexisting) default aof dir
            subprocess_command += [
                "--appendonly",
                "yes",
                "--appendfsync",
                self.redis_fsync_frequency,
            ]
            logger.info("Proceeding with using default Redis dump file.")

        logger.info(
            "Starting Redis server with command: \n {}".format(subprocess_command)
        )

        return subprocess.Popen(
            subprocess_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _close_store_interface(self):
        """Internal method to kill the subprocess
        running the store
        """
        if hasattr(self, "p_StoreInterface"):
            try:
                self.p_StoreInterface.send_signal(signal.SIGINT)
                try:
                    self.p_StoreInterface.wait(timeout=30)
                    logger.info(
                        "StoreInterface close successful: {}".format(self.store_port)
                    )
                except subprocess.TimeoutExpired as e:
                    logger.error(e)
                    self.p_StoreInterface.send_signal(signal.SIGKILL)
                    logger.info("Killed datastore process")

            except Exception as e:
                logger.exception("Cannot close store {}".format(e))

    def create_actor(self, name, actor):
        """Function to instantiate actor, add signal and comm Links,
        and update self.actors dictionary

        Args:
            name: name of the actor
            actor: improv.actor.Actor
        """
        # Instantiate selected class
        mod = import_module(actor.packagename)
        clss = getattr(mod, actor.classname)
        outgoing_links = (
            self.outgoing_topics[actor.name]
            if actor.name in self.outgoing_topics
            else []
        )
        incoming_links = (
            self.incoming_topics[actor.name]
            if actor.name in self.incoming_topics
            else []
        )
        instance = clss(
            name=actor.name,
            nexus_comm_port=self.actor_in_socket_port,
            broker_sub_port=self.broker_sub_port,
            broker_pub_port=self.broker_pub_port,
            log_pull_port=self.logger_pull_port,
            outgoing_links=outgoing_links,
            incoming_links=incoming_links,
            store_port_num=self.store_port,
            **actor.options,
        )

        if "method" in actor.options.keys():
            # check for spawn
            if "fork" == actor.options["method"]:
                # Add link to StoreInterface store
                store = self.create_store_interface()
                instance.set_store_interface(store)
            else:
                # spawn or forkserver; can't pickle plasma store
                logger.info("No store for this actor yet {}".format(name))
        else:
            # Add link to StoreInterface store
            store = self.create_store_interface()
            instance.set_store_interface(store)

        # Update information
        self.actors.update({name: instance})

    def run_actor(self, actor: Actor):
        """Run the actor continually; used for separate processes
        #TODO: hook into monitoring here?

        Args:
            actor:
        """
        actor.run()

    def create_connections(self):
        for name, connection in self.config.connections.items():
            sources = connection["sources"]
            if not isinstance(sources, list):
                sources = [sources]
            sinks = connection["sinks"]
            if not isinstance(sinks, list):
                sinks = [sinks]

            name_input = tuple(
                ["inputs:"]
                + [name for name in sources]
                + ["outputs:"]
                + [name for name in sinks]
            )
            name = str(
                hash(name_input)
            )  # space-efficient key for uniquely referring to this connection
            logger.info(f"Created link name {name} for link {name_input}")

            for source in sources:
                source_actor = source.split(".")[0]
                source_link = source.split(".")[1]
                if source_actor not in self.outgoing_topics.keys():
                    self.outgoing_topics[source_actor] = [LinkInfo(source_link, name)]
                else:
                    self.outgoing_topics[source_actor].append(
                        LinkInfo(source_link, name)
                    )

            for sink in sinks:
                sink_actor = sink.split(".")[0]
                sink_link = sink.split(".")[1]
                if sink_actor not in self.incoming_topics.keys():
                    self.incoming_topics[sink_actor] = [LinkInfo(sink_link, name)]
                else:
                    self.incoming_topics[sink_actor].append(LinkInfo(sink_link, name))

    def assign_link(self, name, link):
        """Function to set up Links between actors
        for data location passing
        Actor must already be instantiated

        #NOTE: Could use this for reassigning links if actors crash?

        #TODO: Adjust to use default q_out and q_in vs being specified
        """
        classname = name.split(".")[0]
        linktype = name.split(".")[1]
        if linktype == "q_out":
            self.actors[classname].set_link_out(link)
        elif linktype == "q_in":
            self.actors[classname].set_link_in(link)
        elif linktype == "watchout":
            self.actors[classname].set_link_watch(link)
        else:
            self.actors[classname].add_link(linktype, link)

    def start_logger(self, log_server_pub_port):
        spawn_context = get_context("spawn")
        self.p_logger = spawn_context.Process(
            target=bootstrap_log_server,
            args=(
                "localhost",
                self.logger_in_port,
                self.logfile,
                log_server_pub_port,
            ),
        )
        logger.debug("logger created")
        self.p_logger.start()
        time.sleep(1)
        logger.debug("logger started")
        if not self.p_logger.is_alive():
            logger.error(
                "Logger process failed to start. "
                "Please see the log server log file for more information. "
                "The improv server will now exit."
            )
            self.quit()
            raise Exception("Could not start log server.")
        logger.debug("logger is alive")
        poll_res = self.logger_in_socket.poll(timeout=5000)
        if poll_res == 0:
            logger.error(
                "Never got reply from logger. Cannot proceed setting up Nexus."
            )
            try:
                with open("log_server.log", "r") as file:
                    logger.debug(file.read())
            except Exception as e:
                logger.error(e)
            self.destroy_nexus()
            logger.error("exiting after destroy")
            exit(1)
        logger_info: LogInfoMsg = self.logger_in_socket.recv_pyobj()
        self.logger_pull_port = logger_info.pull_port
        self.logger_pub_port = logger_info.pub_port
        self.logger_in_socket.send_pyobj(
            LogInfoReplyMsg(logger_info.name, "OK", "registered logger information")
        )
        logger.debug("logger replied with setup message")

    def start_message_broker(self):
        spawn_context = get_context("spawn")
        self.p_broker = spawn_context.Process(
            target=bootstrap_broker, args=("localhost", self.broker_in_port)
        )
        logger.debug("broker created")
        self.p_broker.start()
        time.sleep(1)
        logger.debug("broker started")
        if not self.p_broker.is_alive():
            logger.error(
                "Broker process failed to start. "
                "Please see the log file for more information. "
                "The improv server will now exit."
            )
            self.quit()
            raise Exception("Could not start message broker server.")
        logger.debug("broker is alive")
        poll_res = self.broker_in_socket.poll(timeout=5000)
        if poll_res == 0:
            logger.error("Never got reply from broker. Cannot proceed.")
            try:
                with open("broker_server.log", "r") as file:
                    logger.debug(file.read())
            except Exception as e:
                logger.error(e)
            self.destroy_nexus()
            logger.debug("exiting after destroy")
            exit(1)
        broker_info: BrokerInfoMsg = self.broker_in_socket.recv_pyobj()
        self.broker_sub_port = broker_info.sub_port
        self.broker_pub_port = broker_info.pub_port
        self.broker_in_socket.send_pyobj(
            BrokerInfoReplyMsg(broker_info.name, "OK", "registered broker information")
        )
        logger.debug("broker replied with setup message")

    def _shutdown_broker(self):
        """Internal method to kill the subprocess
        running the message broker
        """
        if self.p_broker:
            try:
                self.p_broker.terminate()
                self.p_broker.join(timeout=5)
                if self.p_broker.exitcode is None:
                    self.p_broker.kill()
                    logger.error("Killed broker process")
                else:
                    logger.info(
                        "Broker shutdown successful with exit code {}".format(
                            self.p_broker.exitcode
                        )
                    )
            except Exception as e:
                logger.exception(f"Unable to close broker {e}")

    def _shutdown_logger(self):
        """Internal method to kill the subprocess
        running the logger
        """
        if self.p_logger:
            try:
                self.p_logger.terminate()
                self.p_logger.join(timeout=5)
                if self.p_logger.exitcode is None:
                    self.p_logger.kill()
                    logger.error("Killed logger process")
                else:
                    logger.info("Logger shutdown successful")
            except Exception as e:
                logger.exception(f"Unable to close logger: {e}")

    def _shutdown_harvester(self):
        """Internal method to kill the subprocess
        running the logger
        """
        if self.p_harvester:
            try:
                self.p_harvester.terminate()
                self.p_harvester.join(timeout=5)
                if self.p_harvester.exitcode is None:
                    self.p_harvester.kill()
                    logger.error("Killed harvester process")
                else:
                    logger.info("Harvester shutdown successful")
            except Exception as e:
                logger.exception(f"Unable to close harvester: {e}")

    def set_up_sockets(self, actor_in_port):

        logger.debug("Connecting to output")
        cfg = self.config.settings  # this could be self.settings instead
        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(SocketOption.LINGER, 0)
        self.out_socket = self.zmq_context.socket(PUB)
        self.out_socket.bind("tcp://*:%s" % cfg["output_port"])
        out_port_string = self.out_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
        cfg["output_port"] = int(out_port_string.split(":")[-1])

        logger.debug("Connecting to control")
        self.in_socket = self.zmq_context.socket(REP)
        self.in_socket.bind("tcp://*:%s" % cfg["control_port"])
        in_port_string = self.in_socket.getsockopt_string(SocketOption.LAST_ENDPOINT)
        cfg["control_port"] = int(in_port_string.split(":")[-1])

        logger.debug("Connecting to actor comm socket")
        self.actor_in_socket = self.zmq_context.socket(REP)
        self.actor_in_socket.bind(f"tcp://*:{actor_in_port}")
        in_port_string = self.actor_in_socket.getsockopt_string(
            SocketOption.LAST_ENDPOINT
        )
        self.actor_in_socket_port = int(in_port_string.split(":")[-1])

        logger.debug("Setting up sync server startup socket")
        self.zmq_sync_context = zmq_sync.Context()
        self.zmq_sync_context.setsockopt(SocketOption.LINGER, 0)

        self.logger_in_socket = self.zmq_sync_context.socket(REP)
        self.logger_in_socket.bind("tcp://*:0")
        logger_in_port_string = self.logger_in_socket.getsockopt_string(
            SocketOption.LAST_ENDPOINT
        )
        self.logger_in_port = int(logger_in_port_string.split(":")[-1])

        self.broker_in_socket = self.zmq_sync_context.socket(REP)
        self.broker_in_socket.bind("tcp://*:0")
        broker_in_port_string = self.broker_in_socket.getsockopt_string(
            SocketOption.LAST_ENDPOINT
        )
        self.broker_in_port = int(broker_in_port_string.split(":")[-1])

    def start_improv_services(self, log_server_pub_port, store_size):
        logger.debug("Starting logger")
        self.start_logger(log_server_pub_port)
        logger.addHandler(
            log.ZmqLogHandler("localhost", self.logger_pull_port, self.zmq_sync_context)
        )

        logger.debug("starting broker")
        self.start_message_broker()

        logger.debug("Parsing redis persistence")
        self.configure_redis_persistence()

        logger.debug("Starting redis server")
        # default size should be system-dependent
        self._start_store_interface(store_size)
        logger.info("Redis server started")

        self.out_socket.send_string("StoreInterface started")

        if self.config.settings["harvest_data_from_memory"]:
            logger.debug("starting harvester")
            self.start_harvester()

        # connect to store and subscribe to notifications
        logger.info("Create new store object")
        self.store = StoreInterface(server_port_num=self.store_port)
        logger.info(f"Redis server connected on port {self.store_port}")
        logger.info("all services started")

    def apply_cli_config_overrides(
        self, store_size, control_port, output_port, actor_in_port
    ):
        if store_size is not None:
            self.config.settings["store_size"] = store_size
        if control_port is not None:
            self.config.settings["control_port"] = control_port
        if output_port is not None:
            self.config.settings["output_port"] = output_port
        if actor_in_port is not None:
            self.config.settings["actor_in_port"] = actor_in_port

    def start_harvester(self):
        spawn_context = get_context("spawn")
        self.p_harvester = spawn_context.Process(
            target=bootstrap_harvester,
            args=(
                "localhost",
                self.broker_in_port,
                "localhost",
                self.store_port,
                "localhost",
                self.broker_pub_port,
                "localhost",
                self.logger_pull_port,
            ),
        )
        self.p_harvester.start()
        time.sleep(1)
        if not self.p_harvester.is_alive():
            logger.error(
                "Harvester process failed to start. "
                "Please see the log file for more information. "
                "The improv server will now exit."
            )
            self.quit()
            raise Exception("Could not start harvester server.")

        harvester_info: HarvesterInfoMsg = self.broker_in_socket.recv_pyobj()
        self.broker_in_socket.send_pyobj(
            HarvesterInfoReplyMsg(
                harvester_info.name, "OK", "registered harvester information"
            )
        )
        logger.info("Harvester server started")

    async def serve(self, serve_function, *args, **kwargs):
        await serve_function(*args, **kwargs)

    async def poll_kernel(self):
        try:
            done, pending = await asyncio.wait(
                self.tasks, return_when=concurrent.futures.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            pass

        # sort through tasks to see where we got input from
        # (so we can choose a handler)
        for i, t in enumerate(self.tasks):
            if i == 0:
                if t in done:
                    self.tasks[i] = asyncio.create_task(self.process_actor_message())
            elif t in done:
                self.tasks[i] = asyncio.create_task(self.remote_input())

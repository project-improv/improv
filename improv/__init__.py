import logging
import sys
import os.path
import signal
import asyncio
import click
import subprocess
import zmq.asyncio as zmq
from zmq import REP
from zmq.log.handlers import PUBHandler
from contextlib import redirect_stdout
from improv.tui import TUI
from multiprocessing import Process, get_context
from improv.nexus import Nexus

DEFAULT_OUTPUT_PORT = 5555
DEFAULT_LOGGING_PORT = 5556 
DEFAULT_CONTROL_PORT = 5557

@click.command()
@click.option('-a', '--actor-path', type=click.Path(exists=True, resolve_path=True), multiple=True, default=[''], help="search path to add to sys.path when looking for actors; defaults to the directory containing CONFIGFILE")
@click.argument('configfile', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('control_port', type=click.INT, default=DEFAULT_CONTROL_PORT)
@click.argument('output_port', type=click.INT, default=DEFAULT_OUTPUT_PORT)
@click.argument('logging_port', type=click.INT, default=DEFAULT_LOGGING_PORT)
def default_invocation(configfile, control_port, output_port, logging_port, actor_path):
    """
    Function provided as an entry point for command-line usage. 

    \b
    CONFIGFILE    YAML file specifying improv pipeline 
    CONTROL_PORT  port on which control signals are sent to/from server
    OUTPUT_PORT   port on which messages from server are broadcast
    LOGGING_PORT  port on which logging messages are broadcast
    """

    if not actor_path:
        sys.path.append(os.path.dirname(configfile))
    else:
        sys.path.extend(actor_path)
    
    app = TUI(DEFAULT_CONTROL_PORT, DEFAULT_LOGGING_PORT, DEFAULT_CONTROL_PORT)

    server = subprocess.Popen(['improv-server', configfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    app.run()

    server.wait()  # wait for improv server to successfully close


    

@click.command()
@click.argument('configfile', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('control_port', type=click.INT, default=DEFAULT_CONTROL_PORT)
@click.argument('output_port', type=click.INT, default=DEFAULT_OUTPUT_PORT)
@click.argument('logging_port', type=click.INT, default=DEFAULT_LOGGING_PORT)
def run_server(configfile, control_port, output_port, logging_port):
    """
    Function provided as an entry point for command line usage. Runs the improv
    server in headless mode.
    
    \b
    CONFIGFILE    YAML file specifying improv pipeline
    CONTROL_PORT  port on which control signals are sent to/from server
    OUTPUT_PORT   port on which messages from server are broadcast
    LOGGING_PORT  port on which logging messages are broadcast
    """
    zmq_log_handler = PUBHandler('tcp://*:%s' % logging_port)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s %(message)s',
                        handlers=[logging.FileHandler("global.log"),
                                  zmq_log_handler])

    server = Nexus()
    server.createNexus(file=configfile, control_port=control_port, output_port=output_port)
    server.startNexus()
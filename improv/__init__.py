import logging
import sys
import os.path
import re
import click
import subprocess
import zmq.asyncio as zmq
from zmq import REP
from zmq.log.handlers import PUBHandler
from contextlib import redirect_stdout
from improv.tui import TUI
from multiprocessing import Process, get_context
from improv.nexus import Nexus

class IPAddressParamType(click.types.ParamType):
    name = "TCP address"

    def convert(self, value, param, ctx):
        # first check if it's a full IP address
        regex = r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
        if ':' in value:
            [address, port] = value.split(':')
            match = re.match(regex, address)
            if not match or not all([0 <= part < 256 for part in address.split('.')]):
                self.fail("{address!r} is not a valid address.".format(address=address), param, ctx)
            else:
                ip = address
            
        else:  # assume it's just a port
            ip = "127.0.0.1"  # localhost
            port = value

        # validate port number
        if 0 <= int(value) < 2**16:  # valid port numbers are [0, 2**16)
            return(ip + ":" + port)
        else:
            self.fail("{value!r} is not a valid port number.".format(value=value), param, ctx)
        
    def __repr__(self):
        return "TCP"

IPTYPE = IPAddressParamType()
DEFAULT_CONTROL_PORT = "5555"
DEFAULT_OUTPUT_PORT = "5556"
DEFAULT_LOGGING_PORT = "5557"

@click.command()
@click.option('-a', '--actor-path', type=click.Path(exists=True, resolve_path=True), multiple=True, default=[''], help="search path to add to sys.path when looking for actors; defaults to the directory containing CONFIGFILE")
@click.argument('configfile', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('control_port', type=IPTYPE, default=DEFAULT_CONTROL_PORT)
@click.argument('output_port', type=IPTYPE, default=DEFAULT_OUTPUT_PORT)
@click.argument('logging_port', type=IPTYPE, default=DEFAULT_LOGGING_PORT)
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
    
    app = TUI(control_port, output_port, logging_port)

    server = subprocess.Popen(['improv-server', configfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    app.run()

    server.wait()  # wait for improv server to successfully close


    

@click.command()
@click.argument('configfile', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('control_port', type=IPTYPE, default=DEFAULT_CONTROL_PORT)
@click.argument('output_port', type=IPTYPE, default=DEFAULT_OUTPUT_PORT)
@click.argument('logging_port', type=IPTYPE, default=DEFAULT_LOGGING_PORT)
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
    zmq_log_handler = PUBHandler('tcp://%s' % logging_port)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s %(message)s',
                        handlers=[logging.FileHandler("global.log"),
                                  zmq_log_handler])

    server = Nexus()
    server.createNexus(file=configfile, control_port=control_port, output_port=output_port)
    server.startNexus()
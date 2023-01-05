import logging
import os.path
import re
import argparse
import subprocess
from zmq.log.handlers import PUBHandler
from improv.tui import TUI
from improv.nexus import Nexus

MAX_PORT = 2**16 - 1
DEFAULT_CONTROL_PORT = "5555"
DEFAULT_OUTPUT_PORT = "5556"
DEFAULT_LOGGING_PORT = "5557"

def file_exists(fname):
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError("{} not found".format(fname))
    return fname

def path_exists(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("{} not found".format(path))
    return path

def is_valid_port(port):
    p = int(port)
    if 0 <= p < MAX_PORT:
        return p
    else:
        raise argparse.ArgumentTypeError("Port {} invalid. Ports must be in [0, {}).".format(p, MAX_PORT))

def is_valid_ip_addr(addr):
        regex = r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
        if ':' in addr:
            [address, port] = addr.split(':')
            match = re.match(regex, address)
            part_list = address.split('.')
            if not match or len(part_list) != 4 or not all([0 <= int(part) < 256 for part in part_list]):
                raise argparse.ArgumentTypeError("{address!r} is not a valid address.".format(address=address))
            else:
                ip = address
        
        else:  # assume it's just a port
            ip = "127.0.0.1"  # localhost
            port = addr
        
        port = str(is_valid_port(port))

        return(ip + ":" + port)

def parse_cli_args(args):
    parser = argparse.ArgumentParser(description='Command line tool for improv.')

    subparsers = parser.add_subparsers(title="subcommands", help="for launching individual components", required=False)

    run_parser = subparsers.add_parser('run', description="Start the improv client and server together")
    run_parser.add_argument('-c', '--control-port', type=is_valid_port, default=DEFAULT_CONTROL_PORT, help="local port on which control are sent to/from server")
    run_parser.add_argument('-o', '--output-port', type=is_valid_port, default=DEFAULT_OUTPUT_PORT, help="local port on which server output messages are broadcast")
    run_parser.add_argument('-l', '--logging-port', type=is_valid_port, default=DEFAULT_LOGGING_PORT, help="local port on which logging messages are broadcast")
    run_parser.add_argument('-f', '--logfile', default="global.log", help="name of log file")
    run_parser.add_argument('-a', '--actor-path', type=path_exists, action='append', default=[], help="search path to add to sys.path when looking for actors; defaults to the directory containing configfile")
    run_parser.add_argument('configfile', type=file_exists, help="YAML file specifying improv pipeline")
    run_parser.set_defaults(func=run)

    client_parser = subparsers.add_parser('client', description="Start the improv client")
    client_parser.add_argument('-c', '--control-port', type=is_valid_ip_addr, default=DEFAULT_CONTROL_PORT, help="address on which control signals are sent to the server")
    client_parser.add_argument('-s', '--server-port', type=is_valid_ip_addr, default=DEFAULT_OUTPUT_PORT, help="address on which messages from the server are received")
    client_parser.add_argument('-l', '--logging-port', type=is_valid_ip_addr, default=DEFAULT_LOGGING_PORT, help="address on which logging messages are broadcast")
    client_parser.set_defaults(func=run_client)

    server_parser = subparsers.add_parser('server', description="Start the improv server")
    server_parser.add_argument('-c', '--control-port', type=is_valid_port, default=DEFAULT_CONTROL_PORT, help="local port on which control signals are received")
    server_parser.add_argument('-o', '--output-port', type=is_valid_port, default=DEFAULT_OUTPUT_PORT, help="local port on which output messages are broadcast")
    server_parser.add_argument('-l', '--logging-port', type=is_valid_port, default=DEFAULT_LOGGING_PORT, help="local port on which logging messages are broadcast")
    server_parser.add_argument('-f', '--logfile', default="global.log", help="name of log file")
    server_parser.add_argument('-a', '--actor-path', type=path_exists, action='append', default=[], help="search path to add to sys.path when looking for actors; defaults to the directory containing configfile")
    server_parser.add_argument('configfile', type=file_exists, help="YAML file specifying improv pipeline")
    server_parser.set_defaults(func=run_server)

    return parser.parse_args(args)


def default_invocation(): 
    """
    Function provided as an entry point for command-line usage. 
    """
    args = parse_cli_args(sys.argv[1:])
    args.func(args)


def run_client(args):
    app = TUI(args.control_port, args.server_port, args.logging_port)

    app.run()

def run_server(args):
    """
    Runs the improv server in headless mode.
    """
    zmq_log_handler = PUBHandler('tcp://*:%s' % args.logging_port)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s %(message)s',
                        handlers=[logging.FileHandler(args.logfile),
                                  zmq_log_handler])

    server = Nexus()
    server.createNexus(file=args.configfile, control_port=args.control_port, output_port=args.output_port)
    print("Server running on (control, output, log) ports ({}, {}, {})...".format(args.control_port, args.output_port, args.logging_port))
    server.startNexus()


def run(args):
    apath_opts = []
    for p in args.actor_path:
        if p:
            apath_opts.append('-a')
            apath_opts.append(p)

    server_opts = ['improv', 'server', 
                            '-c', str(args.control_port), 
                            '-o', str(args.output_port),
                            '-l', str(args.logging_port),
                            '-f', args.logfile,
    ]
    server_opts.extend(apath_opts)
    server_opts.append(args.configfile)

    server = subprocess.Popen(server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    args.server_port = args.output_port
    run_client(args)

    server.wait()


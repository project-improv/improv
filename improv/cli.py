import logging
import os.path
import re
import argparse
import signal
import subprocess
import sys
import psutil
import time
import datetime
from improv.tui import TUI
from improv.nexus import Nexus

MAX_PORT = 2**16 - 1


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
        raise argparse.ArgumentTypeError(
            "Port {} invalid. Ports must be in [0, {}).".format(p, MAX_PORT)
        )


def is_valid_ip_addr(addr):
    regex = r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
    if ":" in addr:
        [address, port] = addr.split(":")
        match = re.match(regex, address)
        part_list = address.split(".")
        if (
            not match
            or len(part_list) != 4
            or not all([0 <= int(part) < 256 for part in part_list])
        ):
            raise argparse.ArgumentTypeError(
                "{address!r} is not a valid address.".format(address=address)
            )
        else:
            ip = address

    else:  # assume it's just a port
        ip = "127.0.0.1"  # localhost
        port = addr

    port = str(is_valid_port(port))
    return ip + ":" + port


def parse_cli_args(args):
    parser = argparse.ArgumentParser(description="Command line tool for improv.")

    subparsers = parser.add_subparsers(
        title="subcommands", help="for launching individual components", required=True
    )

    run_parser = subparsers.add_parser(
        "run", description="Start the improv client and server together"
    )
    run_parser.add_argument(
        "-c",
        "--control-port",
        type=is_valid_port,
        help="local port on which control are sent to/from server",
    )
    run_parser.add_argument(
        "-o",
        "--output-port",
        type=is_valid_port,
        help="local port on which server output messages are broadcast",
    )
    run_parser.add_argument(
        "-l",
        "--logging-port",
        type=is_valid_port,
        help="local port on which logging messages are broadcast",
    )
    run_parser.add_argument(
        "-f", "--logfile", default="global.log", help="name of log file"
    )
    run_parser.add_argument(
        "-a",
        "--actor-path",
        type=path_exists,
        action="append",
        default=[],
        help="search path to add to sys.path when looking for actors; \
              defaults to the directory containing configfile",
    )
    run_parser.add_argument(
        "configfile", type=file_exists, help="YAML file specifying improv pipeline"
    )
    run_parser.set_defaults(func=run)

    client_parser = subparsers.add_parser(
        "client", description="Start the improv client"
    )
    client_parser.add_argument(
        "-c",
        "--control-port",
        type=is_valid_ip_addr,
        help="address on which control signals are sent to the server",
    )
    client_parser.add_argument(
        "-s",
        "--server-port",
        type=is_valid_ip_addr,
        help="address on which messages from the server are received",
    )
    client_parser.add_argument(
        "-l",
        "--logging-port",
        type=is_valid_ip_addr,
        help="address on which logging messages are broadcast",
    )
    client_parser.set_defaults(func=run_client)

    server_parser = subparsers.add_parser(
        "server", description="Start the improv server"
    )
    server_parser.add_argument(
        "-c",
        "--control-port",
        type=is_valid_port,
        help="local port on which control signals are received",
    )
    server_parser.add_argument(
        "-o",
        "--output-port",
        type=is_valid_port,
        help="local port on which output messages are broadcast",
    )
    server_parser.add_argument(
        "-l",
        "--logging-port",
        type=is_valid_port,
        help="local port on which logging messages are broadcast",
    )
    server_parser.add_argument(
        "-f", "--logfile", default="global.log", help="name of log file"
    )
    server_parser.add_argument(
        "-a",
        "--actor-path",
        type=path_exists,
        action="append",
        default=[],
        help="search path to add to sys.path when looking for actors; \
            defaults to the directory containing configfile",
    )
    server_parser.add_argument(
        "configfile", type=file_exists, help="YAML file specifying improv pipeline"
    )
    server_parser.set_defaults(func=run_server)

    list_parser = subparsers.add_parser(
        "list", description="List running improv processes"
    )
    list_parser.set_defaults(func=run_list)

    cleanup_parser = subparsers.add_parser(
        "cleanup", description="Kill all processes returned by 'improv list'"
    )
    cleanup_parser.set_defaults(func=run_cleanup)

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

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(message)s",
        handlers=[logging.FileHandler("improv-debug.log")],
    )

    if not args.actor_path:
        sys.path.append(os.path.dirname(args.configfile))
    else:
        sys.path.extend(args.actor_path)

    server = Nexus()
    control_port, output_port, log_port = server.create_nexus(
        file=args.configfile,
        control_port=args.control_port,
        output_port=args.output_port,
        log_server_pub_port=args.logging_port,
        logfile=args.logfile,
    )
    curr_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{curr_dt} Server running on (control, output, log) ports "
        f"({control_port}, {output_port}, {log_port}).\n"
        f"Press Ctrl-C to quit."
    )
    try:
        server.start_nexus(server.poll_queues, poll_function=server.poll_kernel)
    except Exception as e:
        print(f"CLI-started server run encountered uncaught error {e}")
        logging.error(f"CLI-started server run encountered uncaught error {e}")
        raise e

    if args.actor_path:
        for p in args.actor_path:
            sys.path.remove(p)
    else:
        sys.path.remove(os.path.dirname(args.configfile))


def run_list(args, printit=True):
    out_list = []
    pattern = re.compile(r"(improv (run|client|server)|redis-server)")
    #    mp_pattern = re.compile(r"-c from multiprocessing") # TODO is this right?
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["cmdline"]:
            cmdline = " ".join(proc.info["cmdline"])
            if re.search(pattern, cmdline):  # or re.search(mp_pattern, cmdline):
                out_list.append(proc)
                if printit:
                    print(f"{proc.pid} {proc.name()} {cmdline}")

    return out_list


def run_cleanup(args, headless=False):
    proc_list = run_list(args, printit=False)
    if proc_list:
        if not headless:
            print(f"The following {len(proc_list)} processes will be killed:")
            for proc in proc_list:
                cmdline = " ".join(proc.info["cmdline"])
                print(f"{proc.pid} {proc.name()} {cmdline}")
            res = input("Is that okay [y/N]? ")
        else:
            res = "y"

        if res.lower() == "y":
            for proc in proc_list:
                try:
                    if not proc.status() == psutil.STATUS_STOPPED:
                        logging.info(
                            f"process {proc.pid} {proc.name()}"
                            f" has status {proc.status()}. Interrupting."
                        )
                        proc.send_signal(signal.SIGINT)
                except psutil.NoSuchProcess:
                    pass
            gone, alive = psutil.wait_procs(proc_list, timeout=3)
            for p in alive:
                try:
                    p.terminate()
                    p.wait(timeout=10)
                except psutil.TimeoutExpired as e:
                    logging.warning(f"{e}: Process did not exit on time.")
                    try:
                        p.kill()
                    except psutil.NoSuchProcess as e:
                        logging.warning(
                            f"{e}: Process exited after wait timeout"
                            f" but before kill signal attempted."
                        )
                        # this happens sometimes because Nexus uses gracious
                        # timeout periods.
                except psutil.NoSuchProcess as e:
                    logging.warning(
                        f"{e}: Process exited after wait timeout"
                        f" but before kill signal attempted."
                    )
                    # this happens sometimes because Nexus uses gracious
                    # timeout periods.

    else:
        if not headless:
            print("No running processes found.")


def run(args, timeout=10):
    apath_opts = []
    for p in args.actor_path:
        if p:
            apath_opts.append("-a")
            apath_opts.append(p)

    server_opts = [
        "improv",
        "server",
        "-f",
        args.logfile,
    ]

    if args.control_port:
        server_opts.append("-c")
        server_opts.append(str(args.control_port))

    if args.output_port:
        server_opts.append("-o")
        server_opts.append(str(args.output_port))

    if args.logging_port:
        server_opts.append("-l")
        server_opts.append(str(args.logging_port))

    server_opts.extend(apath_opts)
    server_opts.append(args.configfile)

    print(" ".join(server_opts))

    with open("improv-debug.log", mode="a+") as logfile:
        server = subprocess.Popen(server_opts, stdout=logfile, stderr=logfile)

    # wait for server to start up
    ports = get_server_ports(args, timeout)
    if ports:
        control_port, output_port, logging_port = ports
        args.logging_port = logging_port
        args.control_port = control_port
        args.server_port = output_port
        run_client(args)

    try:
        wait_timeout = 60
        print(f"Waiting {wait_timeout} seconds for Nexus to complete shutdown.")
        server.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        print("Cleaning up the hard way. May have exited dirty.")
        server.terminate()
        server.wait(timeout)
        run_cleanup(args, headless=True)


def get_server_ports(args, timeout):
    # save current datetime so we can see when server has started up
    curr_dt = datetime.datetime.now().replace(microsecond=0)

    increment = 0.05
    time_now = 0
    ports = None
    while time_now < timeout:
        server_start_time = _server_start_logged("improv-debug.log")
        if server_start_time and server_start_time >= curr_dt:
            ports = _get_ports("improv-debug.log")
            if ports:
                break

        time.sleep(increment)
        time_now += increment

    if not server_start_time:
        print(
            f"Unable to read server start time from {'improv-debug.log'}.\n"
            "This may be because the server could not be started or "
            "did not log its activity."
        )
    elif not ports:
        print(f"Unable to read ports from {'improv-debug.log'}.")

    return ports


def _server_start_logged(logfile):
    # read logfile to make sure new session and ports are logged
    with open(logfile, mode="r") as logfile:
        contents = logfile.read()
        pattern = re.compile(
            r"\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}(?= Server running on)"
        )
        matches = pattern.findall(contents)
        if matches:
            return datetime.datetime.strptime(matches[-1], "%Y-%m-%d %H:%M:%S")
        else:
            return None


def _get_ports(logfile):
    # read logfile to get ports
    with open(logfile, mode="r") as logfile:
        contents = logfile.read()
        return _read_log_contents_for_ports(contents)


def _read_log_contents_for_ports(logfile_contents):
    pattern = re.compile(r"(?<=\(control, output, log\) ports \()\d*, \d*, \d*")

    # get most recent match (log file may contain old runs)
    port_str_list = pattern.findall(logfile_contents)
    if port_str_list:
        port_str = port_str_list[-1]
        return (int(p) for p in port_str.split(", "))
    else:
        return None

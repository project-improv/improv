import pytest
import os
import datetime
from collections import namedtuple
import subprocess
import asyncio
import signal
import improv.cli as cli

from test_nexus import ports

SERVER_WARMUP = 16
SERVER_TIMEOUT = 16


@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    yield None
    os.chdir(prev)


@pytest.fixture
async def server(setdir, ports):
    """
    Sets up a server using minimal.yaml in the configs folder.
    Requires the actor path command line argument and so implicitly
    tests that as well.
    """
    os.chdir("configs")

    control_port, output_port, logging_port = ports

    # start server
    server_opts = [
        "improv",
        "server",
        "-c",
        str(control_port),
        "-o",
        str(output_port),
        "-l",
        str(logging_port),
        "-a",
        "..",
        "-f",
        "testlog",
        "minimal.yaml",
    ]

    server = subprocess.Popen(
        server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await asyncio.sleep(SERVER_WARMUP)
    yield server
    server.wait(SERVER_TIMEOUT)
    try:
        os.remove("testlog")
    except FileNotFoundError:
        pass


@pytest.fixture
async def cli_args(setdir, ports):
    logfile = "tmp.log"
    control_port, output_port, logging_port = ports
    config_file = "configs/minimal.yaml"
    Args = namedtuple(
        "cli_args",
        "control_port output_port logging_port logfile configfile actor_path",
    )

    args = Args(control_port, output_port, logging_port, logfile, config_file, [])
    return args


def test_configfile_required(setdir):
    with pytest.raises(SystemExit):
        cli.parse_cli_args(["run"])

    with pytest.raises(SystemExit):
        cli.parse_cli_args(["server"])

    with pytest.raises(SystemExit):
        cli.parse_cli_args(["server", "does_not_exist.yaml"])


def test_multiple_actor_path(setdir):
    args = cli.parse_cli_args(
        ["run", "-a", "actors", "-a", "configs", "configs/blank_file.yaml"]
    )
    assert len(args.actor_path) == 2

    args = cli.parse_cli_args(
        ["server", "-a", "actors", "-a", "configs", "configs/blank_file.yaml"]
    )
    assert len(args.actor_path) == 2


@pytest.mark.parametrize(
    ("mode", "flag", "expected"),
    [
        ("run", "-c", "6000"),
        ("run", "-o", "6000"),
        ("run", "-l", "6000"),
        ("server", "-c", "6000"),
        ("server", "-o", "6000"),
        ("server", "-l", "6000"),
        ("client", "-c", "6000"),
        ("client", "-s", "6000"),
        ("client", "-l", "6000"),
    ],
)
def test_can_override_ports(mode, flag, expected, setdir):
    file = "configs/blank_file.yaml"
    localhost = "127.0.0.1:"
    params = {
        "-c": "control_port",
        "-o": "output_port",
        "-s": "server_port",
        "-l": "logging_port",
    }

    if mode in ["run", "server"]:
        args = cli.parse_cli_args([mode, flag, expected, file])
        assert vars(args)[params[flag]] == int(expected)
    else:
        args = cli.parse_cli_args([mode, flag, expected])
        assert vars(args)[params[flag]] == localhost + expected


@pytest.mark.parametrize(
    ("mode", "flag", "expected"),
    [
        ("run", "-c", "127.0.0.1:6000"),
        ("run", "-o", "-6000"),
        ("run", "-l", str(cli.MAX_PORT + 1)),
        ("server", "-c", "127.0.0.1:6000"),
        ("server", "-o", "-6000"),
        ("server", "-l", str(cli.MAX_PORT + 1)),
    ],
)
def test_non_port_is_error(mode, flag, expected):
    file = "configs/blank_file.yaml"
    with pytest.raises(SystemExit):
        cli.parse_cli_args([mode, flag, expected, file])


@pytest.mark.parametrize(
    ("mode", "flag", "expected"),
    [
        ("client", "-c", "111.127.0.0.1:6000"),
        ("client", "-s", "127.0.1:6000"),
        ("client", "-s", "-6000"),
        ("client", "-l", str(cli.MAX_PORT + 1)),
        ("client", "-l", "127.0.0.1"),
        ("client", "-l", "127.0.0.1:"),
        ("client", "-l", "127.0.0.1:" + str(cli.MAX_PORT)),
    ],
)
def test_non_ip_is_error(mode, flag, expected):
    with pytest.raises(SystemExit):
        cli.parse_cli_args([mode, flag, expected])


@pytest.mark.parametrize(
    ("mode", "flag", "expected"),
    [
        ("client", "-c", "127.0.0.1:6000"),
        ("client", "-s", "155.4.4.3:4000"),
    ],
)
def test_can_override_ip(mode, flag, expected):
    params = {
        "-c": "control_port",
        "-o": "output_port",
        "-s": "server_port",
        "-l": "logging_port",
    }
    args = cli.parse_cli_args([mode, flag, expected])
    assert vars(args)[params[flag]] == expected


async def test_sigint_kills_server(server):
    server.send_signal(signal.SIGINT)


async def test_improv_list_nonempty(server):
    proc_list = cli.run_list("", printit=False)
    assert len(proc_list) > 0
    server.send_signal(signal.SIGINT)


async def test_improv_kill_empties_list(server):
    proc_list = cli.run_list("", printit=False)
    assert len(proc_list) > 0
    cli.run_cleanup("", headless=True)
    proc_list = cli.run_list("", printit=False)
    assert len(proc_list) == 0


async def test_improv_run_writes_stderr_to_log(setdir, ports):
    os.chdir("configs")
    control_port, output_port, logging_port = ports

    # start server
    server_opts = [
        "improv",
        "run",
        "-c",
        str(control_port),
        "-o",
        str(output_port),
        "-l",
        str(logging_port),
        "-a",
        "..",
        "-f",
        "testlog",
        "blank_file.yaml",
    ]
    server = subprocess.Popen(
        server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await asyncio.sleep(SERVER_WARMUP)
    server.kill()
    server.wait(SERVER_TIMEOUT)
    with open("testlog") as log:
        contents = log.read()
    assert "Traceback" in contents
    os.remove("testlog")
    cli.run_cleanup("", headless=True)


async def test_get_ports_from_logfile(setdir):
    test_control_port = 53349
    test_output_port = 53350
    test_logging_port = 53351

    logfile = "tmp.log"

    with open(logfile, "w") as log:
        log.write(
            "Server running on (control, output, log) ports (53345, 53344, 53343)."
        )
        log.write(
            f"Server running on (control, output, log) ports ({test_control_port}, "
            f"{test_output_port}, {test_logging_port})."
        )

    control_port, output_port, logging_port = cli._get_ports(logfile)

    os.remove(logfile)

    assert control_port == test_control_port
    assert output_port == test_output_port
    assert logging_port == test_logging_port


async def test_no_server_start_in_logfile_raises_error(setdir, cli_args, capsys):
    with open(cli_args.logfile, mode="w") as f:
        f.write("this is some placeholder text")

    cli.get_server_ports(cli_args, timeout=1)

    captured = capsys.readouterr()
    assert "Unable to read server start time" in captured.out

    os.remove(cli_args.logfile)
    cli.run_cleanup("", headless=True)


async def test_no_ports_in_logfile_raises_error(setdir, cli_args, capsys):
    curr_dt = datetime.datetime.now().replace(microsecond=0)
    with open(cli_args.logfile, mode="w") as f:
        f.write(f"{curr_dt} Server running on (control, output, log) ports XXX\n")

    cli.get_server_ports(cli_args, timeout=1)
    captured = capsys.readouterr()
    assert f"Unable to read ports from {cli_args.logfile}." in captured.out

    os.remove(cli_args.logfile)
    cli.run_cleanup("", headless=True)

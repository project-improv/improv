import time

import pytest
import os
import asyncio
import subprocess

import improv.tui as tui
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

SERVER_WARMUP = 10
UNUSED_TCP_PORT = 10567


@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    os.chdir("../demos")
    yield None
    os.chdir(prev)


@pytest.fixture
def ip():
    """Fixture to provide an IP test input."""

    pytest.ip = "127.0.0.1"
    return pytest.ip


@pytest.fixture
def unused_tcp_port():
    global UNUSED_TCP_PORT
    pytest.unused_tcp_port = UNUSED_TCP_PORT
    yield pytest.unused_tcp_port
    UNUSED_TCP_PORT += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dir", "configfile", "logfile"),
    [
        ("minimal", "minimal.yaml", "testlog"),
    ],
)
async def test_simple_boot_and_quit(dir, configfile, logfile, setdir, ports):
    os.chdir(dir)

    control_port, output_port, logging_port, actor_in_port = ports

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
        "-f",
        logfile,
        configfile,
    ]

    with open(logfile, mode="a+") as log:
        server = subprocess.Popen(server_opts, stdout=log, stderr=log)
        time.sleep(5)
        print(log.readlines())
    await asyncio.sleep(SERVER_WARMUP)

    # initialize client
    app = tui.TUI(control_port, output_port, logging_port)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*"setup", "enter")
        await pilot.pause(0.5)
        await pilot.press(*"quit", "enter")
        await pilot.pause(2)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(15)
    # os.remove(logfile)  # later, might want to read this file and check for messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dir", "configfile", "logfile", "datafile"),
    [
        ("minimal", "minimal.yaml", "testlog", "sample_generator_data.npy"),
        ("minimal", "minimal_persistence.yaml", "testlog", "test_persistence.csv"),
    ],
)
async def test_stop_output(dir, configfile, logfile, datafile, setdir, ports):
    os.chdir(dir)

    control_port, output_port, logging_port, actor_in_port = ports

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
        "-f",
        logfile,
        configfile,
    ]

    with open(logfile, mode="a+") as log:
        server = subprocess.Popen(server_opts, stdout=log, stderr=log)
    await asyncio.sleep(SERVER_WARMUP)

    # initialize client
    app = tui.TUI(control_port, output_port, logging_port)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*"setup", "enter")
        await pilot.pause(0.5)
        await pilot.press(*"run", "enter")
        await pilot.pause(1)
        await pilot.press(*"stop", "enter")
        await pilot.pause(2)
        await pilot.press(*"quit", "enter")
        await pilot.pause(3)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)

    # check that the file written by Generator's stop function got written
    os.path.isfile(datafile)

    # then remove that file and logile
    os.remove(datafile)
    os.remove(logfile)  # later, might want to read this file and check for messages


@pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dir", "configfile", "logfile", "datafile"),
    [
        ("minimal", "minimal_spawn.yaml", "testlog", "sample_generator_data.npy"),
    ],
)
async def test_stop_output_spawn(dir, configfile, logfile, datafile, setdir, ports):
    os.chdir(dir)

    control_port, output_port, logging_port, actor_in_port = ports

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
        "-f",
        logfile,
        configfile,
    ]

    with open(logfile, mode="a+") as log:
        server = subprocess.Popen(server_opts, stdout=log, stderr=log)
    await asyncio.sleep(SERVER_WARMUP)

    # initialize client
    app = tui.TUI(control_port, output_port, logging_port)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*"setup", "enter")
        await pilot.pause(0.5)
        await pilot.press(*"run", "enter")
        await pilot.pause(1)
        await pilot.press(*"stop", "enter")
        await pilot.pause(2)
        await pilot.press(*"quit", "enter")
        await pilot.pause(3)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)

    # check that the file written by Generator's stop function got written
    os.path.isfile(datafile)

    # then remove that file and logile
    os.remove(datafile)
    os.remove(logfile)  # later, might want to read this file and check for messages

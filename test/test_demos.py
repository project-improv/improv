import pytest
import os
import asyncio
import subprocess
import improv.tui as tui
from demos.basic.actors.zmqActor import ZmqPSActor, ZmqRRActor
import concurrent.futures

from test_nexus import ports

SERVER_WARMUP = 10


@pytest.fixture()
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    os.chdir("../demos")
    yield None
    os.chdir(prev)


@pytest.fixture()
def ip():
    """Fixture to provide an IP test input."""

    pytest.ip = "127.0.0.1"
    return pytest.ip


@pytest.mark.parametrize(
    ("dir", "configfile", "logfile"), [("minimal", "minimal.yaml", "testlog")]
)
async def test_simple_boot_and_quit(dir, configfile, logfile, setdir, ports):
    os.chdir(dir)

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
        await pilot.press(*"quit", "enter")
        await pilot.pause(2)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)
    os.remove(logfile)  # later, might want to read this file and check for messages


@pytest.mark.parametrize(
    ("dir", "configfile", "logfile", "datafile"),
    [("minimal", "minimal.yaml", "testlog", "sample_generator_data.npy")],
)
async def test_stop_output(dir, configfile, logfile, datafile, setdir, ports):
    os.chdir(dir)

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
        await pilot.pause(5)
        await pilot.press(*"run", "enter")
        await pilot.pause(5)
        await pilot.press(*"stop", "enter")
        await pilot.pause(5)
        await pilot.press(*"quit", "enter")
        await pilot.pause(5)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)

    # check that the file written by Generator's stop function got written
    os.path.isfile(datafile)

    # then remove that file and logile
    os.remove(datafile)
    os.remove(logfile)  # later, might want to read this file and check for messages


def test_zmq_ps(ip, unused_tcp_port):
    """Tests if we can set the zmq PUB/SUB socket and send message."""

    act1 = ZmqPSActor("act1", "/tmp/store")
    act2 = ZmqPSActor("act2", "/tmp/store")
    act1.setSendSocket(ip, unused_tcp_port)
    act2.setRecvSocket(ip, unused_tcp_port)
    msg = "hello"
    act1.sendMsg(msg)
    recvmsg = act2.recvMsg()
    assert recvmsg == msg


def test_zmq_rr(ip, unused_tcp_port):
    """Tests if we can set the zmq REQ/REP socket and send message."""

    act1 = ZmqRRActor("act1", "/tmp/store")
    act2 = ZmqRRActor("act2", "/tmp/store")
    act1.setReqSocket(ip, unused_tcp_port)
    act2.setRepSocket(ip, unused_tcp_port)
    msg = "hello"
    reply = "world"

    def handle_request():
        return act1.requestMsg(msg)

    def handle_reply():
        return act2.replyMsg(reply)

    # Use a ThreadPoolExecutor to run handle_request()
    # and handle_reply() in separate threads.

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(handle_request)
        future2 = executor.submit(handle_reply)

        # Ensure the request is sent before the reply.
        request_result = future1.result()
        reply_result = future2.result()

    # Check if the received message is equal to the original message.
    assert reply_result == msg
    # Check if the reply is correct.
    assert request_result == reply


def test_zmq_rr_timeout(ip, unused_tcp_port):
    """Test for requestMsg where we timeout or fail to send"""
    act1 = ZmqRRActor("act1", "/tmp/store")
    act1.setReqSocket(ip, unused_tcp_port)
    msg = "hello"
    replymsg = act1.requestMsg(msg)
    assert replymsg is None

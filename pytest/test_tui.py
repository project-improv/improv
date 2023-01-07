import pytest
import asyncio
import improv.tui as tui
import logging 
import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP
from zmq.log.handlers import PUBHandler

CONTROL_PORT = "5555"
OUTPUT_PORT = "5556"
LOGGING_PORT = "5557" 

@pytest.fixture
def logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    zmq_log_handler = PUBHandler('tcp://*:%s' % LOGGING_PORT)
    logger.addHandler(zmq_log_handler)
    return logger

@pytest.fixture
def sockets():
    context = zmq.Context()
    ctrl_socket = context.socket(REP)
    ctrl_socket.bind("tcp://*:%s" % CONTROL_PORT)
    out_socket = context.socket(PUB)
    out_socket.bind("tcp://*:%s" % OUTPUT_PORT)
    return (ctrl_socket, out_socket)


@pytest.fixture
async def backend(sockets):
    """
    Fake program to be controlled by TUI.
    """
    socket = sockets[0]  # control socket
    msg = await socket.recv_multipart()
    yield socket.send_string("received")

@pytest.fixture
async def publish(sockets):
    """
    Set up a fake server to publish messages for testing.
    """
    pubsocket = sockets[1]  # output socket
    yield pubsocket.send_string("test output socket")

@pytest.fixture
async def quitter(sockets):
    """
    Set up a fake server to publish messages for testing.
    """
    pubsocket = sockets[1]  # output socket
    yield pubsocket.send_string("QUIT")

@pytest.fixture
async def app():
    mock = tui.TUI(CONTROL_PORT, OUTPUT_PORT, LOGGING_PORT)
    yield mock
    await asyncio.sleep(.05)  # give pending tasks a chance to cancel


async def test_console_panel(app, capfd):
    async with app.run_test() as pilot:
        assert app.query_one("#console").id == 'console'

async def test_quit_from_socket(app, quitter, capfd):
    async with app.run_test() as pilot:
        await quitter
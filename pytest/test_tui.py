import pytest
import time
import improv.tui as tui
import logging 
import zmq.asyncio as zmq
from zmq import PUB, REP
from zmq.log.handlers import PUBHandler

CONTROL_PORT = 6555
OUTPUT_PORT = 6556
LOGGING_PORT = 6557 
SERVER_COUNTER = 0

@pytest.fixture
def ports():
    global SERVER_COUNTER
    yield (CONTROL_PORT + SERVER_COUNTER, OUTPUT_PORT + SERVER_COUNTER, LOGGING_PORT + SERVER_COUNTER)
    SERVER_COUNTER += 3

@pytest.fixture
def logger(ports):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    zmq_log_handler = PUBHandler('tcp://*:%s' % ports[2])
    logger.addHandler(zmq_log_handler)
    yield logger
    logger.removeHandler(zmq_log_handler)

@pytest.fixture
async def sockets(ports):
    with zmq.Context() as context:
        ctrl_socket = context.socket(REP)
        ctrl_socket.bind("tcp://*:%s" % ports[0])
        out_socket = context.socket(PUB)
        out_socket.bind("tcp://*:%s" % ports[1])
        yield (ctrl_socket, out_socket)

@pytest.fixture
async def app(ports):
    mock = tui.TUI(*ports)
    yield mock
    time.sleep(0.5)

async def test_console_panel_receives_broadcast(app, sockets, logger):
    async with app.run_test() as pilot:
        await sockets[1].send_string("received")
        await pilot.pause(0.1)
        console = pilot.app.get_widget_by_id("console")
        console.history[0] == 'received'

async def test_quit_from_socket(app, sockets):
    async with app.run_test() as pilot:
        await sockets[1].send_string("QUIT")
        await pilot.pause(0.1)
        assert not pilot.app._running

async def test_log_panel_receives_logging(app, logger):
    async with app.run_test() as pilot:
        logger.info('test')
        await pilot.pause(0.1)
        log_window = pilot.app.get_widget_by_id("log")
        # assert log_window.history[0] == 'INFO'
        assert 'test' in log_window.history[0] 

async def test_input_box_echoed_to_console(app):
    async with app.run_test() as pilot:
        await pilot.press(*'foo', 'enter')
        console = pilot.app.get_widget_by_id("console")
        assert console.history[0] == 'foo'

async def test_quit_screen(app):
    async with app.run_test() as pilot:
        await pilot.press('ctrl+c', 'tab', 'tab', 'enter')
        assert pilot.app._running

        await pilot.press('ctrl+c', 'tab', 'enter')
        await pilot.pause(0.5)
        assert not pilot.app._running


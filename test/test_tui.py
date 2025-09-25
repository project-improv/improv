import pytest
import time
import improv.tui as tui
from improv.messaging import ActorSignalReplyMsg
import logging
import zmq.asyncio as zmq
from zmq import PUB, REP
from zmq.log.handlers import PUBHandler


@pytest.fixture
def logger(ports):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    zmq_log_handler = PUBHandler("tcp://*:%s" % ports[2])
    logger.addHandler(zmq_log_handler)
    yield logger
    logger.removeHandler(zmq_log_handler)
    zmq_log_handler.close()


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
    mock = tui.TUI(*ports, testing=True)
    yield mock
    time.sleep(0.5)


async def test_console_panel_receives_broadcast(app, sockets):
    async with app.run_test() as pilot:
        await sockets[1].send_string("received")
        await pilot.pause(0.1)
        console = pilot.app.get_widget_by_id("console")
        assert console.history[0] == "received"


async def test_quit_from_socket(app, sockets):
    async with app.run_test() as pilot:
        await sockets[1].send_string("QUIT")
        await pilot.pause(0.1)
        assert not pilot.app._running


async def test_log_panel_receives_logging(app, logger):
    async with app.run_test() as pilot:
        logger.info("test")
        await pilot.pause(0.1)
        log_window = pilot.app.get_widget_by_id("log")
        assert "test" in log_window.history[0]


async def test_input_box_echoed_to_console(app, sockets):
    async with app.run_test() as pilot:
        await pilot.press(*"foo", "enter")
        request = await sockets[0].recv_pyobj()
        reply_obj = ActorSignalReplyMsg("TUI", "OK", "foo")
        await sockets[0].send_pyobj(reply_obj)
        console = pilot.app.get_widget_by_id("console")
        assert request.signal == "foo"
        assert console.history[0] == "foo"


async def test_quit_screen(app):
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c", "tab", "tab", "enter")
        assert pilot.app._running

        await pilot.press("ctrl+c", "tab", "enter")
        await pilot.pause(0.5)
        assert not pilot.app._running


async def test_turn_on_print_debug_msg(app, logger):
    async with app.run_test() as pilot:
        logger.debug("test debug message")
        await pilot.pause(0.1)
        log_window = pilot.app.get_widget_by_id("log")
        assert len(log_window.history) == 0

        await pilot.press("ctrl+p")
        await pilot.pause(0.1)
        logger.debug("test debug message")
        await pilot.pause(0.1)
        log_window = pilot.app.get_widget_by_id("log")
        assert "debug" in log_window.history[0]

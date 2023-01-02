import sys
import asyncio
import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextLog, Input
from improv.actor import Actor, Signal
from improv.link import Link
from multiprocessing import Process, get_context
import logging; logger = logging.getLogger(__name__)
from zmq.log.handlers import PUBHandler
logger.setLevel(logging.INFO)  

class SocketLog(TextLog):
    def __init__(self, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        context = zmq.Context()
        self.socket = context.socket(SUB)
        self.socket.connect("tcp://localhost:%s" % str(port))
        self.socket.setsockopt(SUBSCRIBE, b"")

    async def poll(self):
        ready = await self.socket.poll(10)
        if ready:
            msg = await self.socket.recv()
            self.write(msg.decode('utf-8'))

    async def on_mount(self) -> None:
        """Event handler called when widget is added to the app."""
        self.set_interval(1/60, self.poll)

class TUI(App):
    """
    View class for the text user interface. Implemented as a Textual app.
    """
    def __init__(self, console_port=5555, logging_port=5556):
        super().__init__()
        self.console_port = console_port
        self.logging_port = logging_port
        logger.info('Text interface initialized')

    CSS_PATH = "tui.css"
    BINDINGS = [
        ("ctrl-c", "quit", "Quit")
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SocketLog(self.console_port, id="console")
        yield Input()
        yield SocketLog(self.logging_port)
        yield Footer()
    
    def on_key(self, event) -> None:
        logger.info(str(event))

    def on_input_submitted(self, message):
        self.query_one(Input).value = ""
        self.query_one("#console").write(message.value)
        # self.comm_queue.put([message.value])

            
if __name__ == '__main__':
    CONSOLE_PORT = 5555
    LOGGING_PORT = 5556 
    zmq_log_handler = PUBHandler('tcp://*:%s' % LOGGING_PORT)
    logger.addHandler(zmq_log_handler)

    async def publish():
        """
        Set up a fake server to publish messages for testing.
        """
        context = zmq.Context()
        pubsocket = context.socket(PUB)
        pubsocket.bind("tcp://*:%s" % CONSOLE_PORT)

        counter = 0
        while counter < 5:
            await pubsocket.send_string("test " + str(counter))
            await asyncio.sleep(1)
            counter += 1
    
    async def main_loop():
        app = TUI(CONSOLE_PORT, LOGGING_PORT)
        await asyncio.gather(app.run_async(), publish())

    asyncio.run(main_loop())
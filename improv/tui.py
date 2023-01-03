import asyncio
import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextLog, Input
# from improv.actor import Actor, Signal
from improv.link import Link
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
        try:
            ready = await self.socket.poll(10)
            if ready:
                msg = await self.socket.recv()
                self.write(msg.decode('utf-8'))
        except asyncio.CancelledError:
            pass

    async def on_mount(self) -> None:
        """Event handler called when widget is added to the app."""
        self.set_interval(1/60, self.poll)

class TUI(App, inherit_bindings=False):
    """
    View class for the text user interface. Implemented as a Textual app.
    """
    def __init__(self, console_port=5555, logging_port=5556, output_port=5557):
        super().__init__()
        self.console_port = console_port
        self.logging_port = logging_port

        context = zmq.Context()
        self.out_socket = context.socket(REQ)
        self.out_socket.connect("tcp://localhost:%s" % output_port)

        logger.info('Text interface initialized')

    CSS_PATH = "tui.css"
    BINDINGS = [
               ("tab", "focus_next", "Focus Next")
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SocketLog(self.console_port, id="console")
        yield Input()
        yield SocketLog(self.logging_port, id="log")
        yield Footer()

    async def poll_controller(self):
        try:
            ready = await self.out_socket.poll(10)
            if ready:
                reply = await self.out_socket.recv_multipart()
                msg = reply[0].decode('utf-8')
                if msg == 'QUIT':
                    self.exit()
                else:
                    self.query_one("#console").write(msg)

        except asyncio.CancelledError:
            pass


    async def on_mount(self):
        self.set_interval(1/60, self.poll_controller)
        self.set_focus(self.query_one(Input))
    
    async def on_key(self, event) -> None:
        logger.info(str(event))

    async def on_input_submitted(self, message):
        self.query_one(Input).value = ""
        self.query_one("#console").write(message.value)
        await self.out_socket.send_string(message.value)
        
    
            
if __name__ == '__main__':
    CONSOLE_PORT = 5555
    LOGGING_PORT = 5556 
    OUTPUT_PORT = 5557 
    zmq_log_handler = PUBHandler('tcp://*:%s' % LOGGING_PORT)
    logger.addHandler(zmq_log_handler)

    context = zmq.Context()
    socket = context.socket(REP)
    socket.bind("tcp://*:%s" % OUTPUT_PORT)
    pubsocket = context.socket(PUB)
    pubsocket.bind("tcp://*:%s" % CONSOLE_PORT)

    async def backend():
        """
        Fake program to be controlled by TUI.
        """
        while True:
            msg = await socket.recv_multipart()
            await socket.send_string("Awaiting input:")

    async def publish():
        """
        Set up a fake server to publish messages for testing.
        """
        counter = 0
        while True:
            await pubsocket.send_string("test " + str(counter))
            await asyncio.sleep(1)
            counter += 1
    
    async def main_loop():
        app = TUI(CONSOLE_PORT, LOGGING_PORT, OUTPUT_PORT)

        # the following construct ensures both the (infinite) fake servers are killed once the tui finishes
        finished, unfinished = await asyncio.wait([app.run_async(), publish(), backend()], return_when=asyncio.FIRST_COMPLETED)

        for task in unfinished:
            task.cancel()

    asyncio.run(main_loop())
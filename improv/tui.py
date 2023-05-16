import asyncio
import zmq.asyncio as zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER
from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Grid, Container
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    TextLog,
    Input,
    Button,
    Static,
    Label,
    Placeholder,
)
from textual.message import Message
import logging

logger = logging.getLogger(__name__)
from zmq.log.handlers import PUBHandler

logger.setLevel(logging.INFO)


class SocketLog(TextLog):
    def __init__(self, port, context, *args, **kwargs):
        if "formatter" in kwargs:
            self.format = kwargs["formatter"]
            kwargs.pop("formatter")
        else:
            self.format = self._simple_formatter

        super().__init__(*args, **kwargs)
        self.socket = context.socket(SUB)
        self.socket.connect("tcp://%s" % str(port))
        self.socket.setsockopt_string(SUBSCRIBE, "")
        self.history = []
        self.print_debug = False

    class Echo(Message):
        def __init__(self, sender, value) -> None:
            super().__init__()
            self.value = value

    def write(self, content, width=None, expand=False, shrink=True):
        TextLog.write(self, content, width, expand, shrink)
        self.history.append(content)

    @staticmethod
    def _simple_formatter(parts):
        """
        format messages from zmq
        message is a list of message parts
        """
        return " ".join([p.decode("utf-8").replace("\n", " ") for p in parts])

    async def poll(self):
        try:
            ready = await self.socket.poll(10)
            if ready:
                parts = await self.socket.recv_multipart()
                msg_type = parts[0].decode("utf-8")
                if msg_type != "DEBUG" or self.print_debug:
                    msg = self.format(parts)
                    self.write(msg)
                    self.post_message(self.Echo(self, msg))
        except asyncio.CancelledError:
            pass

    async def on_mount(self) -> None:
        """Event handler called when widget is added to the app."""
        self.poller = self.set_interval(1 / 60, self.poll)


class QuitScreen(Screen):
    def compose(self) -> ComposeResult:
        quit_str = (
            "Are you sure you want to quit? "
            "The server has not been stopped, and the process "
            "may continue to run in the background. "
            "To exit safely, enter 'quit' into the command console. "
        )

        yield Grid(
            Static(quit_str, id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    async def on_key(self, event) -> None:
        # make sure button-related key presses don't bubble up to main window
        if event.key == "enter":
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()


class HelpScreen(Screen):
    def compose(self):
        cmd_table = Table()
        cmd_table.add_column("Command", justify="left")
        cmd_table.add_column("Function", justify="left")
        cmd_table.add_row("setup", "Prepare all actors to run")
        cmd_table.add_row("run", "Start the experiment")
        cmd_table.add_row("pause", "???")
        cmd_table.add_row("stop", "???")
        cmd_table.add_row(
            "quit", "Stop everything and terminate this client and the server."
        )

        yield Container(
            Static(cmd_table, id="help_table"),
            Container(Button("OK", id="ok_btn"), id="ok"),
            id="help_screen",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.pop_screen()


class TUI(App, inherit_bindings=False):
    """
    View class for the text user interface. Implemented as a Textual app.
    """

    def __init__(self, control_port, output_port, logging_port):
        super().__init__()
        self.title = "improv console"
        self.control_port = TUI._sanitize_addr(control_port)
        self.output_port = TUI._sanitize_addr(output_port)
        self.logging_port = TUI._sanitize_addr(logging_port)

        self.context = zmq.Context()
        self.control_socket = self.context.socket(REQ)
        self.control_socket.connect("tcp://%s" % self.control_port)

        logger.info("Text interface initialized")

    CSS_PATH = "tui.css"
    BINDINGS = [
        ("tab", "focus_next", "Focus Next"),
        ("ctrl+c", "request_quit", "Emergency Quit"),
        ("ctrl+p", "set_debug", "Toggle Debug Info"),
        ("question_mark", "help", "Help"),
    ]

    def action_set_debug(self):
        log_window = self.get_widget_by_id("log")
        log_window.print_debug = not log_window.print_debug

    @staticmethod
    def _sanitize_addr(input):
        if isinstance(input, int):
            return "localhost:%s" % str(input)
        elif ":" in input:
            return input
        else:
            return "localhost:%s" % input

    @staticmethod
    def format_log_messages(parts):
        type_list = ["debug", "info", "warning", "error", "critical", "exception"]
        msg_type = parts[0].decode("utf-8")
        msg = SocketLog._simple_formatter(parts[1:])
        if msg_type == "DEBUG":
            msg = "[bold black on white]" + msg + "[/]"
        elif msg_type == "INFO":
            msg = "[white]" + msg + "[/]"
        elif msg_type == "WARNING":
            msg = ":warning-emoji:  [yellow]" + msg + "[/]"
        elif msg_type == "ERROR":
            msg = ":heavy_exclamation_mark: [#e40000]" + msg + "[/]"
        elif msg_type == "CRITICAL":
            msg = ":collision::scream: [bold red]" + msg + "[/]"

        return msg

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Grid(
            Header("improv console"),
            Label("[white]Log Messages[/]"),
            SocketLog(
                self.logging_port,
                self.context,
                formatter=self.format_log_messages,
                markup=True,
                id="log",
            ),
            Label("Command History"),
            SocketLog(self.output_port, self.context, id="console"),
            Input(id="input"),
            Footer(),
            id="main",
        )

    async def send_to_controller(self, msg):
        """
        Safe version of send/receive with controller.
        Based on the Lazy Pirate pattern [here](https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)
        """
        REQUEST_TIMEOUT = 2500
        REQUEST_RETRIES = 3

        retries_left = REQUEST_RETRIES

        try:
            logger.info(f"Sending {msg} to controller.")
            await self.control_socket.send_string(msg)
            reply = None

            while True:
                ready = await self.control_socket.poll(REQUEST_TIMEOUT)

                if ready:
                    reply = await self.control_socket.recv_multipart()
                    reply = reply[0].decode("utf-8")
                    logger.info(f"Received {reply} from controller.")
                    break
                else:
                    retries_left -= 1
                    logger.warning("No response from server.")

                # try to close and reconnect
                self.control_socket.setsockopt(LINGER, 0)
                self.control_socket.close()
                if retries_left == 0:
                    logger.error("Server seems to be offline. Giving up.")
                    break

                logger.info("Attempting to reconnect to server...")

                self.control_socket = self.context.socket(REQ)
                self.control_socket.connect("tcp://%s" % self.control_port)

                logger.info(f"Resending {msg} to controller.")
                await self.control_socket.send_string(msg)

        except asyncio.CancelledError:
            pass

        return reply

    async def on_mount(self):
        self.set_focus(self.query_one(Input))

    async def on_input_submitted(self, message):
        self.query_one(Input).value = ""
        self.query_one("#console").write(message.value)
        reply = await self.send_to_controller(message.value)
        self.query_one("#console").write(reply)

    async def on_socket_log_echo(self, message):
        if message.sender.id == "console" and message.value == "QUIT":
            logger.info("Got QUIT; will try to exit")
            self.exit()

    def action_request_quit(self):
        self.push_screen(QuitScreen())

    def action_help(self):
        self.push_screen(HelpScreen())


if __name__ == "__main__":
    CONTROL_PORT = "5555"
    OUTPUT_PORT = "5556"
    LOGGING_PORT = "5557"

    import random

    zmq_log_handler = PUBHandler("tcp://*:%s" % LOGGING_PORT)
    logger.addHandler(zmq_log_handler)
    logger.setLevel(logging.DEBUG)

    context = zmq.Context()
    socket = context.socket(REP)
    socket.bind("tcp://*:%s" % CONTROL_PORT)
    pubsocket = context.socket(PUB)
    pubsocket.bind("tcp://*:%s" % OUTPUT_PORT)

    async def backend():
        """
        Fake program to be controlled by TUI.
        """
        while True:
            msg = await socket.recv_multipart()
            if msg[0].decode("utf-8") == "quit":
                await socket.send_string("QUIT")
            else:
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

    async def log():
        """
        Send fake logging events for testing.
        """
        counter = 0
        type_list = ["debug", "info", "warning", "error", "critical", "exception"]
        while True:
            this_type = random.choice(type_list)
            getattr(logger, this_type)("log message " + str(counter))
            # logger.info("log message " + str(counter))
            await asyncio.sleep(1.2)
            counter += 1

    async def main_loop():
        app = TUI(CONTROL_PORT, OUTPUT_PORT, LOGGING_PORT)

        # the following construct ensures both the (infinite) fake servers are killed once the tui finishes
        finished, unfinished = await asyncio.wait(
            [app.run_async(), publish(), backend(), log()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in unfinished:
            task.cancel()

    asyncio.run(main_loop())

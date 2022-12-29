from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextLog, Input

import logging; logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LogWindow(TextLog):
    def on_mount(self) -> None:
        """Event handler called when widget is added to the app."""
        tui_handler = logging.StreamHandler(self)
        tui_handler.terminator = ""  # prevents extra newline
        logger.addHandler(tui_handler)

class TUIView(App):
    """
    View class for the text user interface. Implemented as a Textual app.
    """

    CSS_PATH = "tui.css"
    # TODO: take this as class input so it can be specified in the YAML file

    BINDINGS = [
        ("ctrl-c", "quit", "Quit")
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TextLog(id="cmdhistory")
        yield Input()
        yield LogWindow()
        yield Footer()

    def on_key(self, event) -> None:
        logger.info(event)

    def on_input_submitted(self, message):
        self.query_one("#cmdhistory").write(message.value)
        self.query_one(Input).value = ""

        # TODO: App should take a comm channel as input for sending back to Nexus


if __name__ == "__main__":
    app = TUIView()
    app.run()
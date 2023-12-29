:py:mod:`improv.tui`
====================

.. py:module:: improv.tui


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   improv.tui.SocketLog
   improv.tui.QuitScreen
   improv.tui.HelpScreen
   improv.tui.TUI




Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.tui.logger
   improv.tui.CONTROL_PORT


.. py:data:: logger

   

.. py:class:: SocketLog(port, context, *args, **kwargs)

   Bases: :py:obj:`textual.widgets.TextLog`

   A widget for logging text.

   .. py:class:: Echo(sender, value)

      Bases: :py:obj:`textual.message.Message`

      Base class for a message.


   .. py:method:: write(content, width=None, expand=False, shrink=True)

      Write text or a rich renderable.

      :param content: Rich renderable (or text).
      :param width: Width to render or ``None`` to use optimal width.
      :param expand: Enable expand to widget width, or ``False`` to use `width`.
      :param shrink: Enable shrinking of content to fit width.


   .. py:method:: poll()
      :async:


   .. py:method:: on_mount() -> None
      :async:

      Event handler called when widget is added to the app.



.. py:class:: QuitScreen(name: str | None = None, id: str | None = None, classes: str | None = None)

   Bases: :py:obj:`textual.screen.Screen`

   A widget for the root of the app.

   .. py:method:: compose() -> textual.app.ComposeResult

      Called by Textual to create child widgets.

      Extend this to build a UI.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Container(
              Tree(), Viewer()
          )
          yield Footer()
      ```


   .. py:method:: on_key(event) -> None
      :async:


   .. py:method:: on_button_pressed(event: textual.widgets.Button.Pressed) -> None



.. py:class:: HelpScreen(name: str | None = None, id: str | None = None, classes: str | None = None)

   Bases: :py:obj:`textual.screen.Screen`

   A widget for the root of the app.

   .. py:method:: compose()

      Called by Textual to create child widgets.

      Extend this to build a UI.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Container(
              Tree(), Viewer()
          )
          yield Footer()
      ```


   .. py:method:: on_button_pressed(event: textual.widgets.Button.Pressed) -> None



.. py:class:: TUI(control_port, output_port, logging_port)

   Bases: :py:obj:`textual.app.App`

   View class for the text user interface. Implemented as a Textual app.

   .. py:attribute:: CSS_PATH
      :value: 'tui.css'

      

   .. py:attribute:: BINDINGS
      :value: [('tab', 'focus_next', 'Focus Next'), ('ctrl+c', 'request_quit', 'Emergency Quit'), ('ctrl+p',...

      

   .. py:method:: action_set_debug()


   .. py:method:: format_log_messages(parts)
      :staticmethod:


   .. py:method:: compose() -> textual.app.ComposeResult

      Create child widgets for the app.


   .. py:method:: send_to_controller(msg)
      :async:

      Safe version of send/receive with controller.
      Based on the Lazy Pirate pattern [here]
      (https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)


   .. py:method:: on_mount()
      :async:


   .. py:method:: on_input_submitted(message)
      :async:


   .. py:method:: on_socket_log_echo(message)
      :async:


   .. py:method:: action_request_quit()


   .. py:method:: action_help()



.. py:data:: CONTROL_PORT
   :value: '5555'

   


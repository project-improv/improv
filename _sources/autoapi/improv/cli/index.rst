:py:mod:`improv.cli`
====================

.. py:module:: improv.cli


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   improv.cli.file_exists
   improv.cli.path_exists
   improv.cli.is_valid_port
   improv.cli.is_valid_ip_addr
   improv.cli.parse_cli_args
   improv.cli.default_invocation
   improv.cli.run_client
   improv.cli.run_server
   improv.cli.run_list
   improv.cli.run_cleanup
   improv.cli.run
   improv.cli.get_server_ports



Attributes
~~~~~~~~~~

.. autoapisummary::

   improv.cli.MAX_PORT
   improv.cli.DEFAULT_CONTROL_PORT
   improv.cli.DEFAULT_OUTPUT_PORT
   improv.cli.DEFAULT_LOGGING_PORT


.. py:data:: MAX_PORT

   

.. py:data:: DEFAULT_CONTROL_PORT
   :value: '0'

   

.. py:data:: DEFAULT_OUTPUT_PORT
   :value: '0'

   

.. py:data:: DEFAULT_LOGGING_PORT
   :value: '0'

   

.. py:function:: file_exists(fname)


.. py:function:: path_exists(path)


.. py:function:: is_valid_port(port)


.. py:function:: is_valid_ip_addr(addr)


.. py:function:: parse_cli_args(args)


.. py:function:: default_invocation()

   Function provided as an entry point for command-line usage.


.. py:function:: run_client(args)


.. py:function:: run_server(args)

   Runs the improv server in headless mode.


.. py:function:: run_list(args, printit=True)


.. py:function:: run_cleanup(args, headless=False)


.. py:function:: run(args, timeout=10)


.. py:function:: get_server_ports(args, timeout)



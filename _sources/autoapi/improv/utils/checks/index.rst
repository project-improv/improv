:py:mod:`improv.utils.checks`
=============================

.. py:module:: improv.utils.checks

.. autoapi-nested-parse::

   The script checks the validity of the YAML configuration file.

   Example usage:

       $ python checks.py [file_name].yaml

       $ python checks.py good_config.yaml
       No loops.

       $ python checks.py bad_config.yaml
       Loop(s) found.
       Processor to Analysis to Acquirer



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   improv.utils.checks.check_if_connections_acyclic



.. py:function:: check_if_connections_acyclic(path_to_yaml)

   Check if connections in the YAML configuration file do not form a loop.

   Print 'No loops.' if the connections are acyclic.
   Print 'Loop(s) found.' followed by the loop path if loops are found.

   :param path_to_yaml: Path to the YAML file.
   :type path_to_yaml: str

   :returns: Whether the connections are acyclic.
   :rtype: bool



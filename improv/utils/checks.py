"""
The script checks the validity of the YAML configuration file.

Example usage:

    $ python checks.py [file_name].yaml

"""

import sys

import networkx as nx
import yaml


def check_if_connections_acyclic(path_to_yaml):
    """
    Check if connections in the YAML configuration file do not form a loop.

    Print 'No loops.' if the connections are acyclic.
    Print 'Loop(s) found.' followed by the loop path if loops are found.

    Args:
        path_to_yaml (str): Path to the YAML file.

    Returns:
        bool: Whether the connections are acyclic.

    """
    with open(path_to_yaml) as f:
        raw = yaml.safe_load(f)["connections"]

    # Need to keep only module names
    connections = {}
    for connection, values in raw.items():
        for source in values["sources"]:
            source_name = source.split(".")[0]
            if source_name not in connections.keys():
                connections[source_name] = [
                    sink_name.split(".")[0] for sink_name in values["sinks"]
                ]
            else:
                connections[source_name] = connections[source_name] + [
                    sink_name.split(".")[0] for sink_name in values["sinks"]
                ]

    g = nx.DiGraph(connections)
    dag = nx.is_directed_acyclic_graph(g)

    if dag:
        print("No loops.")
        return True

    print("Loop(s) found.")
    loops = nx.algorithms.cycles.simple_cycles(g)
    for loop in loops:
        print(*loop, sep=" to ")
    return False


if __name__ == "__main__":
    check_if_connections_acyclic(sys.argv[1])

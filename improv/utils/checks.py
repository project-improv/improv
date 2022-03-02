
"""
The script checks the validity of the YAML configuration file.

Example usage:

    $ python checks.py [file_name].yaml

    $ python checks.py good_config.yaml
    No loops.

    $ python checks.py bad_config.yaml
    Loop(s) found.
    Processor to Analysis to Acquirer

"""
import os
import sys

import networkx as nx
import yaml
import matplotlib.pyplot as plt
import sys


def check_if_connections_acyclic(path_to_yaml):
    """
    Check if connections in the YAML configuration file do not form a loop.

    Print 'No loops.' if the connections are acyclic.
    Print 'Loop(s) found.' followed by the loop path if loops are found.

    :param path_to_yaml: Path to the YAML file.
    :type path_to_yaml: str

    :return: Whether the connections are acyclic.
    :rtype: bool

    """
    with open(path_to_yaml) as f:
        raw = yaml.safe_load(f)['connections']
    # Need to keep only module names
    connections = {}
    keys = []
    i = 0
    for key, values in raw.items():
        new_key = key.split('.')[0]
        new_values = [value.split('.')[0] for value in values]
        connections[new_key] = new_values
        keys.append(new_key)


    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Graph - Shapes', fontsize=10)
    print(connections)
    g = nx.DiGraph()
    for x in keys:
        g.add_node(x)

    for x in keys:
        for y in connections[x]:
            g.add_edge(x, y)

    dag = nx.is_directed_acyclic_graph(g)

    if dag:
        print('No loops.')
        pos = nx.circular_layout(g)
        nx.draw(g, pos, node_size=15000, node_color='#ADD8E6', font_size=15, font_weight='bold', with_labels = True, width = 10, arrowsize = 60, min_source_margin = 80, min_target_margin = 80)
        plt.tight_layout()
        demo = ""
        if("basic" in path_to_yaml):
            demo = "basic_demo"
        elif("live" in path_to_yaml):
            demo = "live_demo"
        elif("naumann" in path_to_yaml):
            demo = "naumann_demo"
        elif("pandas" in path_to_yaml):
            demo = "pandas_demo"
        elif("suite2p" in path_to_yaml):
            demo = "suite2p"
        else:
            demo = path_to_yaml

        plt.savefig(demo + "_graph.png", format="PNG")
        return True

    print('Loop(s) found.')
    loops = nx.algorithms.cycles.simple_cycles(g)
    for loop in loops:
        print(*loop, sep=' to ')
    return False


if __name__ == '__main__':
    check_if_connections_acyclic(sys.argv[1])

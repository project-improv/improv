import sys
import os.path

def default_invocation():
    import argparse
    import logging

    from improv.nexus import Nexus

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="name of a YAML configuration file")
    parser.add_argument("--actor-path", help="search path to add to sys.path when looking for actors; defaults to the directory containing <config>")
    args = parser.parse_args()

    if not args.actor_path:
        sys.path.append(os.path.dirname(args.config))
    else:
        sys.path.append(os.path.abspath(args.actor_path))

    loadfile = args.config

    nexus = Nexus("Sample")
    nexus.createNexus(file=loadfile)
    nexus.startNexus()
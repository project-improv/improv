
def default_invocation():
    import argparse
    import logging

    from improv.nexus import Nexus

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of a YAML configuration file")
    args = parser.parse_args()

    loadfile = args.config_file

    nexus = Nexus("Sample")
    nexus.createNexus(file=loadfile)
    nexus.startNexus()
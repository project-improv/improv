import logging
import sys
import os.path
import click
from improv.nexus import Nexus

@click.command()
@click.argument('configfile', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--actor-path', type=click.Path(exists=True, resolve_path=True), default='', help="search path to add to sys.path when looking for actors; defaults to the directory containing CONFIGFILE")
def default_invocation(configfile, actor_path):
    """
    Function provided as an entry point for command-line usage. Invoke using
    improv <YAML config file>
    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not actor_path:
        sys.path.append(os.path.dirname(configfile))
    else:
        sys.path.append(actor_path)
    click.echo(configfile)

    nexus = Nexus("Sample")
    nexus.createNexus(file=configfile)
    nexus.startNexus()
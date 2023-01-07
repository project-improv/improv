import click
from click.testing import CliRunner
from improv.cli import default_invocation
import pytest
import asyncio

@pytest.fixture()
def runner():
    """
    Fixture to create a new command line test runner.     
    """
    yield CliRunner()

    # this is needed because CliRunner is sensitive to global state and our 
    # tests may end up closing the global asyncio event loop
    # to be safe, we get a brand new event loop
    asyncio.set_event_loop(asyncio.new_event_loop())

def test_sample_demo(runner):
    result = runner.invoke(default_invocation, ["configs/sample_config.yaml", "--actor-path=actors"], input="run\nquit")
    assert result.exit_code == 0

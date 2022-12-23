import click
from click.testing import CliRunner
from improv import default_invocation

def test_sample_demo():
    runner = CliRunner()
    result = runner.invoke(default_invocation, ["configs/sample_config.yaml", "--actor-path=actors"], input="run\nquit")
    assert result.exit_code == 0
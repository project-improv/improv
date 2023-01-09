import pytest
import os
import improv.cli as cli

@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    yield None
    os.chdir(prev)

def test_configfile_required(setdir):
    with pytest.raises(SystemExit):
        cli.parse_cli_args(['run'])

    with pytest.raises(SystemExit):
        cli.parse_cli_args(['server'])
    
    with pytest.raises(SystemExit):
        cli.parse_cli_args(['server', 'does_not_exist.yaml'])


def test_multiple_actor_path(setdir):
    args = cli.parse_cli_args(['run', '-a', 'actors', '-a', 'configs', 'configs/blank_file.yaml'])
    assert len(args.actor_path) == 2

    args = cli.parse_cli_args(['server', '-a', 'actors', '-a', 'configs', 'configs/blank_file.yaml'])
    assert len(args.actor_path) == 2

@pytest.mark.parametrize("mode,flag,expected", [('run', '-c', "6000"), 
                                                ('run', '-o', "6000"),
                                                ('run', '-l', "6000"),
                                                ('server', '-c', "6000"), 
                                                ('server', '-o', "6000"),
                                                ('server', '-l', "6000"),
                                                ('client', '-c', "6000"), 
                                                ('client', '-s', "6000"),
                                                ('client', '-l', "6000"),
                                                ])
def test_can_override_ports(mode, flag, expected, setdir):
    file = 'configs/blank_file.yaml'
    localhost = "127.0.0.1:"
    params = {'-c': 'control_port',
                '-o': 'output_port',
                '-s': 'server_port',
                '-l': 'logging_port'
            }

    if mode in ['run', 'server']:
        args = cli.parse_cli_args([mode, flag, expected, file])
        assert vars(args)[params[flag]] == int(expected)
    else:
        args = cli.parse_cli_args([mode, flag, expected])
        assert vars(args)[params[flag]] == localhost + expected

@pytest.mark.parametrize("mode,flag,expected", [('run', '-c', "127.0.0.1:6000"), 
                                                ('run', '-o', "-6000"),
                                                ('run', '-l', str(cli.MAX_PORT + 1)),
                                                ('server', '-c', "127.0.0.1:6000"), 
                                                ('server', '-o', "-6000"),
                                                ('server', '-l', str(cli.MAX_PORT + 1)),
                                                ])
def test_non_port_is_error(mode, flag, expected):
    file = 'configs/blank_file.yaml'
    with pytest.raises(SystemExit):
        cli.parse_cli_args([mode, flag, expected, file])

@pytest.mark.parametrize("mode,flag,expected", [('client', '-c', "111.127.0.0.1:6000"), 
                                                ('client', '-s', "127.0.1:6000"),
                                                ('client', '-s', "-6000"),
                                                ('client', '-l', str(cli.MAX_PORT + 1)),
                                                ('client', '-l', "127.0.0.1"),
                                                ('client', '-l', "127.0.0.1:"),
                                                ('client', '-l', "127.0.0.1:" + str(cli.MAX_PORT)),
                                                ])
def test_non_ip_is_error(mode, flag, expected):
    with pytest.raises(SystemExit):
        cli.parse_cli_args([mode, flag, expected])

@pytest.mark.parametrize("mode,flag,expected", [('client', '-c', "127.0.0.1:6000"), 
                                                ('client', '-s', "155.4.4.3:4000"),
                                                ])
def test_can_override_ip(mode, flag, expected):
    params = {'-c': 'control_port',
                '-o': 'output_port',
                '-s': 'server_port',
                '-l': 'logging_port'
            }
    args = cli.parse_cli_args([mode, flag, expected])
    assert vars(args)[params[flag]] == expected


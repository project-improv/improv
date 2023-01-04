import pytest
import improv.cli as cli

def test_configfile_required():
    with pytest.raises(SystemExit):
        cli.parse_cli_args(['run'])

    with pytest.raises(SystemExit):
        cli.parse_cli_args(['server'])
    
    with pytest.raises(SystemExit):
        cli.parse_cli_args(['server', 'does_not_exist.yaml'])


def test_multiple_actor_path():
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
def test_can_override_ports(mode, flag, expected):
    file = 'configs/blank_file.yaml'
    params = {'-c': 'control_port',
                '-o': 'output_port',
                '-s': 'server_port',
                '-l': 'logging_port'
            }

    if mode in ['run', 'server']:
        args = cli.parse_cli_args([mode, flag, expected, file])
    else:
        args = cli.parse_cli_args([mode, flag, expected])
    assert vars(args)[params[flag]] == int(expected)


# TODO: test ports vs addresses
# TODO: test individual functions
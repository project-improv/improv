import pytest
import os
import sys
import subprocess
import asyncio
import signal
import improv.cli as cli

CONTROL_PORT = 5555
OUTPUT_PORT = 5556
LOGGING_PORT = 5557 

@pytest.fixture
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    yield None
    os.chdir(prev)

@pytest.fixture
async def server(setdir):
    """
    Sets up a server using minimal.yaml in the configs folder. 
    Requires the actor path command line argument and so implicitly 
    tests that as well.
    """
    os.chdir('configs')
    #start server
    server_opts = ['improv', 'server', 
                            '-c', str(CONTROL_PORT), 
                            '-o', str(OUTPUT_PORT),
                            '-l', str(LOGGING_PORT),
                            '-a', '..',
                            '-f', 'testlog', 'minimal.yaml',
    ]
    server = subprocess.Popen(server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(1)
    yield server
    server.wait()
    os.remove('testlog')

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

async def test_sigint_kills_server(server):
    server.send_signal(signal.SIGINT)

async def test_improv_list_nonempty(server):
    proc_list = cli.run_list('', printit=False)
    assert len(proc_list) > 0
    server.send_signal(signal.SIGINT)

async def test_improv_kill_empties_list(server):
    proc_list = cli.run_list('', printit=False)
    assert len(proc_list) > 0
    cli.run_cleanup('', headless=True)
    proc_list = cli.run_list('', printit=False)
    assert len(proc_list) == 0
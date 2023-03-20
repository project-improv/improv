import pytest
import os
import asyncio
import subprocess
import improv.tui as tui

from test_nexus import ports

@pytest.fixture()
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__)) 
    os.chdir('../demos')
    yield None
    os.chdir(prev)

@pytest.mark.parametrize("dir,configfile,logfile", [('minimal','minimal.yaml', 'testlog')])
async def test_simple_boot_and_quit(dir, configfile, logfile, setdir, ports):
    os.chdir(dir)

    control_port, output_port, logging_port = ports

    #start server
    server_opts = ['improv', 'server', 
                            '-c', str(control_port), 
                            '-o', str(output_port),
                            '-l', str(logging_port),
                            '-f', logfile, configfile,
    ]

    with open(logfile, mode='a+') as log:
        server = subprocess.Popen(server_opts, stdout=log, stderr=log)

    # initialize client
    app = tui.TUI(control_port, output_port, logging_port)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*'setup', 'enter')
        await pilot.pause(.5)
        await pilot.press(*'quit', 'enter')
        await pilot.pause(8)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)
    os.remove(logfile)  # later, might want to read this file and check for messages

@pytest.mark.parametrize("dir,configfile,logfile,datafile", [('minimal', 'minimal.yaml', 'testlog', 'sample_generator_data.npy')])
async def test_stop_output(dir, configfile, logfile, datafile, setdir, ports):
    os.chdir(dir)

    control_port, output_port, logging_port = ports

    #start server
    server_opts = ['improv', 'server', 
                            '-c', str(control_port), 
                            '-o', str(output_port),
                            '-l', str(logging_port),
                            '-f', logfile, configfile,
    ]

    with open(logfile, mode='a+') as log:
        server = subprocess.Popen(server_opts, stdout=log, stderr=log)

    # initialize client
    app = tui.TUI(control_port, output_port, logging_port)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*'setup', 'enter')
        await pilot.pause(.5)
        await pilot.press(*'run', 'enter')
        await pilot.pause(1)
        await pilot.press(*'stop', 'enter')
        await pilot.pause(2)
        await pilot.press(*'quit', 'enter')
        await pilot.pause(8)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait(10)

    # check that the file written by Generator's stop function got written
    os.path.isfile(datafile)

    # then remove that file and logile
    os.remove(datafile)
    os.remove(logfile)  # later, might want to read this file and check for messages

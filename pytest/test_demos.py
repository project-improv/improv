import pytest
import os
import asyncio
import subprocess
import improv.tui as tui

CONTROL_PORT = 6555
OUTPUT_PORT = 6556
LOGGING_PORT = 6557 

@pytest.fixture()
def setdir():
    prev = os.getcwd()
    os.chdir(os.path.dirname(__file__)) 
    os.chdir('../demos')
    yield None
    os.chdir(prev)

@pytest.mark.parametrize("dir,configfile,logfile", [('minimal','minimal.yaml', 'testlog')])
async def test_simple_boot_and_quit(dir, configfile, logfile, setdir):
    os.chdir(dir)

    #start server
    server_opts = ['improv', 'server', 
                            '-c', str(CONTROL_PORT), 
                            '-o', str(OUTPUT_PORT),
                            '-l', str(LOGGING_PORT),
                            '-f', logfile, configfile,
    ]
    server = subprocess.Popen(server_opts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # initialize client
    app = tui.TUI(CONTROL_PORT, OUTPUT_PORT, LOGGING_PORT)

    # run client
    async with app.run_test() as pilot:
        print("running pilot")
        await pilot.press(*'setup', 'enter')
        await pilot.pause(0.5)
        await pilot.press(*'quit', 'enter')
        await pilot.pause(.8)
        assert not pilot.app._running

    # wait on server to fully shut down
    server.wait()
    os.remove(logfile)  # later, might want to read this file and check for messages
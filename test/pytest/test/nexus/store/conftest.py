import pytest
import subprocess
from multiprocessing import Process

@pytest.fixture(scope="module")
# scope=function, class, module, package or session?
# function: the default scope, the fixture is destroyed at the end of the test.
#
# class: the fixture is destroyed during teardown of the last test in the class.
#
# module: the fixture is destroyed during teardown of the last test in the module.
#
# package: the fixture is destroyed during teardown of the last test in the package.
# session: the fixture is destroyed at the end of the test session.

def set_up(self):
    ''' Start the server
    '''
    print('Setting up Plasma store.')
    self.p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def tear_down(self):
    ''' Kill the server
    '''
    print('Tearing down Plasma store.')
    self.p.kill()
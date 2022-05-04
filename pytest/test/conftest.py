import pytest
import subprocess
import asyncio
from nexus.actor import RunManager, AsyncRunManager
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

class StoreDependentTestCase():
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

class ActorDependentTestCase():
    def set_up(self):
        ''' Start the server
        '''
        print('Setting up Plasma store.')
        self.p = subprocess.Popen(
            ['plasma_store','-s', '/tmp/store', '-m', str(10000000)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tear_down(self):
        ''' Kill the server
        '''
        print('Tearing down Plasma store.')
        self.p.kill()

    def run_setup(self):
        print('Set up = True.')
        self.is_set_up = True

    def run_method(self):
        # Accurate print statement?
        print('Running method.')
        self.run_num += 1

    def process_setup(self):
        print('Processing setup.')
        pass

    def process_run(self):
        print('Processing run - ran.')
        self.q_comm.put('ran')

    def create_process(self, q_sig, q_comm):
        print('Creating process.')
        with RunManager('test', self.process_run, self.process_setup, q_sig, q_comm) as rm:
            print(rm)

    async def createAsyncProcess(self, q_sig, q_comm):
        print('Creating asyn process.')
        async with AsyncRunManager('test', self.process_run, self.process_setup, q_sig, q_comm) as rm:
            print(rm)

    async def a_put(self, signal, time):
        print('Async put.')
        await asyncio.sleep(time)
        self.q_sig.put_async(signal)
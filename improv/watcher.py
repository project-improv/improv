from multiprocessing import Process, Queue, Manager, cpu_count, set_start_method
import numpy as np
import pyarrow.plasma as plasma
import asyncio
import subprocess
import signal
import time
from queue import Empty
import numpy as np
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import asyncio
import concurrent
from pyarrow.plasma import ObjectNotAvailable
from improv.actor import Actor, Spike, RunManager, AsyncRunManager
from improv.store import ObjectNotFoundError
from pyarrow.plasma import ObjectNotAvailable

class BasicWatcher(Actor):
    '''
    Actor that monitors stored objects from the other actors
    and saves objects that have been flagged by those actors
    '''

    def __init__(self, *args, inputs= None):
        super().__init__(*args)

        self.watchin= inputs
        

    def setup(self):
        '''
        set up tasks and polling based on inputs which will
        be used for asynchronous polling of input queues
        '''
        self.numSaved= 0
        self.tasks= []
        self.polling= self.watchin
        self.setUp= False

    def run(self):
        '''
        continually run the watcher to check all of the 
        input queues for objects to save
        '''


        with RunManager(self.name, self.watchrun, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

        print('watcher saved '+ str(self.numSaved)+ ' objects')

    def watchrun(self):
        '''
        set up async loop for polling
        '''
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.watch())

    async def watch(self):
        '''
        function for asynchronous polling of input queues
        loops through each of the queues in watchin and checks
        if an object is present and then saves the object if found
        '''

        if self.setUp== False:
            for q in self.polling:
                self.tasks.append(asyncio.ensure_future(q.get_async()))
            self.setUp = True

        done, pending= await asyncio.wait(self.tasks, return_when= concurrent.futures.FIRST_COMPLETED)

        for i,t in enumerate(self.tasks):
            if t in done or self.polling[i].status == 'done':
                r = self.polling[i].result # r is array with id and name of object
                actorID = self.polling[i].getStart() # name of actor asking watcher to save the object
                try:
                    obj= self.client.getID(r[0])
                    np.save('output/saved/'+actorID+r[1], obj)
                except ObjectNotFoundError as e:
                    logger.info(e.message)
                    pass
                self.tasks[i] = (asyncio.ensure_future(self.polling[i].get_async()))

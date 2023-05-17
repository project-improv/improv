import numpy as np
import asyncio
# import pyarrow.plasma as plasma
# from multiprocessing import Process, Queue, Manager, cpu_count, set_start_method
# import subprocess
# import signal
# import time
from queue import Empty
import logging

import concurrent
# from pyarrow.plasma import ObjectNotAvailable
from improv.actor import Actor, Signal, RunManager
from improv.store import ObjectNotFoundError
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasicWatcher(Actor):
    """
    Actor that monitors stored objects from the other actors
    and saves objects that have been flagged by those actors
    """

    def __init__(self, *args, inputs=None):
        super().__init__(*args)

        self.watchin = inputs

    def setup(self):
        """
        set up tasks and polling based on inputs which will
        be used for asynchronous polling of input queues
        """
        self.numSaved = 0
        self.tasks = []
        self.polling = self.watchin
        self.setUp = False

    def run(self):
        """
        continually run the watcher to check all of the
        input queues for objects to save
        """

        with RunManager(
            self.name, self.watchrun, self.setup, self.q_sig, self.q_comm
        ) as rm:
            logger.info(rm)

        print("watcher saved " + str(self.numSaved) + " objects")

    def watchrun(self):
        """
        set up async loop for polling
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.watch())

    async def watch(self):
        """
        function for asynchronous polling of input queues
        loops through each of the queues in watchin and checks
        if an object is present and then saves the object if found
        """

        if self.setUp is False:
            for q in self.polling:
                self.tasks.append(asyncio.create_task(q.get_async()))
            self.setUp = True

        done, pending = await asyncio.wait(
            self.tasks, return_when=concurrent.futures.FIRST_COMPLETED
        )

        for i, t in enumerate(self.tasks):
            if t in done or self.polling[i].status == "done":
                r = self.polling[i].result  # r is array with id and name of object
                actorID = self.polling[
                    i
                ].getStart()  # name of actor asking watcher to save the object
                try:
                    obj = self.client.getID(r[0])
                    np.save("output/saved/" + actorID + r[1], obj)
                except ObjectNotFoundError as e:
                    logger.info(e.message)
                    pass
                self.tasks[i] = asyncio.create_task(self.polling[i].get_async())


class Watcher:
    """Monitors the store as separate process
    TODO: Facilitate Watcher being used in multiple processes (shared list)
    """

    # Related to subscribe - could be private, i.e., _subscribe
    def __init__(self, name, client):
        self.name = name
        self.client = client
        self.flag = False
        self.saved_ids = []

        self.client.subscribe()
        self.n = 0

    def setLinks(self, links):
        self.q_sig = links

    def run(self):
        while True:
            if self.flag:
                try:
                    self.checkStore2()
                except Exception as e:
                    logger.error("Watcher exception during run: {}".format(e))
                    # break
            try:
                signal = self.q_sig.get(timeout=0.005)
                if signal == Signal.run():
                    self.flag = True
                    logger.warning("Received run signal, begin running")
                elif signal == Signal.quit():
                    logger.warning("Received quit signal, aborting")
                    break
                elif signal == Signal.pause():
                    logger.warning("Received pause signal, pending...")
                    self.flag = False
                elif signal == Signal.resume():  # currently treat as same as run
                    logger.warning("Received resume signal, resuming")
                    self.flag = True
            except Empty:
                pass  # no signal from Nexus

    # def checkStore(self):
    #     notification_info = self.client.notify()
    #     recv_objid, recv_dsize, recv_msize = notification_info
    #     obj = self.client.getID(recv_objid)
    #     try:
    #         self.saveObj(obj)
    #         self.n += 1
    #     except Exception as e:
    #         logger.error('Watcher error: {}'.format(e))

    def saveObj(self, obj, name):
        with open(
            "/media/hawkwings/Ext Hard Drive/dump/dump" + name + ".pkl", "wb"
        ) as output:
            pickle.dump(obj, output)

    def checkStore2(self):
        objs = list(self.client.get_all().keys())
        ids_to_save = list(set(objs) - set(self.saved_ids))

        # with Pool() as pool:
        #     saved_ids = pool.map(saveObjbyID, ids_to_save)
        # print('Saved :', len(saved_ids))
        # self.saved_ids.extend(saved_ids)

        for id in ids_to_save:
            self.saveObj(self.client.getID(id), str(id))
            self.saved_ids.append(id)


# def saveObjbyID(id):
#     client = plasma.connect('/tmp/store')
#     obj = client.get(id)
#     with open('/media/hawkwings/Ext\ Hard\ Drive/dump/dump'+str(id)+'.pkl', 'wb') as output:
#         pickle.dump(obj, output)
#     return id

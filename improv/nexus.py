import asyncio
import concurrent
import functools
import multiprocessing
import signal
import sys
import os
import time
import subprocess

import numpy as np
import logging
import pyarrow.plasma as plasma

from multiprocessing import Process, Queue, Manager, cpu_count, set_start_method
from PyQt5 import QtGui, QtWidgets
from importlib import import_module
from threading import Thread
from queue import Empty, Full
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from improv.watcher import BasicWatcher
from improv import store
from improv.actor import Spike
from improv.tweak import Tweak
from improv.link import Link, MultiLink

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s %(message)s',
                    handlers=[logging.FileHandler("global.log"),
                              logging.StreamHandler()])

# TODO: Set up limbo.notify in async function (?)

class Nexus():
    ''' Main server class for handling objects in RASP
    '''
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def createNexus(self, file=None, use_hdd=False):
        self._startStore(40000000000) #default size should be system-dependent; this is 40 GB

        #connect to store and subscribe to notifications
        self.limbo = store.Limbo()
        self.limbo.subscribe()

        # LMDB storage
        self.use_hdd = use_hdd
        if self.use_hdd:
            self.lmdb_name = f'lmdb_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.limbo_dict = dict()

        self.comm_queues = {}
        self.sig_queues = {}
        self.data_queues = {}
        self.actors = {}
        self.flags = {}
        self.processes = []

        #self.startWatcher()

        self.loadTweak(file=file)

        self.flags.update({'quit':False, 'run':False, 'load':False})
        self.allowStart = False

    def startNexus(self):

        ''' Puts all actors in separate processes and begins polling
            to listen to comm queues
        '''
        for name,m in self.actors.items(): # m accesses the specific actor class instance
            if 'GUI' not in name: #GUI already started
                p = Process(target=self.runActor, name=name, args=(m,))
                if 'Watcher' not in name:
                    if 'daemon' in self.tweak.actors[name].options: # e.g. suite2p creates child processes.
                        p.daemon = self.tweak.actors[name].options['daemon']
                        logger.info('Setting daemon to {} for {}'.format(p.daemon,name))
                    else: 
                        p.daemon = True #default behavior
                self.processes.append(p)

        self.start()

        if self.tweak.hasGUI:
            loop = asyncio.get_event_loop()

            signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
            for s in signals:
                loop.add_signal_handler(
                    s, lambda s=s: self.stop_polling(s, loop)) #TODO
            try:
                res = loop.run_until_complete(self.pollQueues()) #TODO: in Link executor, complete all tasks
            except asyncio.CancelledError:
                logging.info("Loop is cancelled")
            
            try:
                logging.info(f"Result of run_until_complete: {res}") 
            except:
                logging.info("Res failed to await")

            logging.info(f"Current loop: {asyncio.get_event_loop()}") 
            
            loop.stop()
            loop.close()
            logger.info('Shutdown loop')
        else:
            pass

    def loadTweak(self, file=None):
        ''' For each connection:
            create a Link with a name (purpose), start, and end
            Start links to one actor's name, end to the other.
            Nexus gives start_actor the Link as a q_in,
              and end_actor the Link as a q_out
            Nexus maintains dict of name and associated Link.
            Nexus also has list of Links that it is itself connected to
              for communication purposes.
            OR
            For each connection, create 2 Links. Nexus acts as intermediary.
        '''
        #TODO load from file or user input, as in dialogue through FrontEnd?

        self.tweak = Tweak(configFile = file)
        self.tweak.createConfig()

        # create all data links requested from Tweak config
        self.createConnections()

        if self.tweak.hasGUI:
            # Have to load GUI first (at least with Caiman)
            name = self.tweak.gui.name
            m = self.tweak.gui # m is TweakModule
            # treat GUI uniquely since user communication comes from here
            try:
                visualClass = m.options['visual']
                # need to instantiate this actor
                visualActor = self.tweak.actors[visualClass]
                self.createActor(visualClass, visualActor)
                # then add links for visual
                for k,l in {key:self.data_queues[key] for key in self.data_queues.keys() if visualClass in key}.items():
                    self.assignLink(k, l)

                #then give it to our GUI
                self.createActor(name, m)
                self.actors[name].setup(visual=self.actors[visualClass])

                self.p_GUI = Process(target=self.actors[name].run, name=name)
                self.p_GUI.daemon = True
                self.p_GUI.start()

            except Exception as e:
                logger.error('Exception in setting up GUI {}'.format(name)+': {}'.format(e))

        else:
            # have fake GUI for communications
            q_comm = Link('GUI_comm', 'GUI', self.name)
            self.comm_queues.update({q_comm.name:q_comm})

        # First set up each class/actor
        for name,actor in self.tweak.actors.items():
            if name not in self.actors.keys():
                #Check for actors being instantiated twice
                self.createActor(name, actor)

        # Second set up each connection b/t actors
        for name,link in self.data_queues.items():
            self.assignLink(name, link)

        if self.tweak.settings['use_watcher'] is not None:

            watchin = []

            for name in self.tweak.settings['use_watcher']:
                watch_link= Link(name+'_watch', name, 'Watcher')
                self.assignLink(name+'.watchout', watch_link)
                watchin.append(watch_link)

            self.createWatcher(watchin)

        #TODO: error handling for if a user tries to use q_in without defining it

    def createWatcher(self, watchin):
        watcher= BasicWatcher('Watcher', inputs=watchin)
        watcher.setStore(store.Limbo(watcher.name))
        q_comm = Link('Watcher_comm', watcher.name, self.name)
        q_sig = Link('Watcher_sig', self.name, watcher.name)
        self.comm_queues.update({q_comm.name:q_comm})
        self.sig_queues.update({q_sig.name:q_sig})
        watcher.setCommLinks(q_comm, q_sig)

        self.actors.update({watcher.name: watcher})

    def createActor(self, name, actor):
        ''' Function to instantiate actor, add signal and comm Links,
            and update self.actors dictionary
        '''
        # Instantiate selected class
        mod = import_module(actor.packagename)
        clss = getattr(mod, actor.classname)
        instance = clss(actor.name, **actor.options)

        # Add link to Limbo store
        limbo = self.createLimbo(actor.name)
        instance.setStore(limbo)

        # Add signal and communication links
        limbo_arg = [None, None]
        if self.use_hdd:
            limbo_arg = [limbo, self.createLimbo('default')]

        q_comm = Link(actor.name+'_comm', actor.name, self.name)
        q_sig = Link(actor.name+'_sig', self.name, actor.name)
        self.comm_queues.update({q_comm.name:q_comm})
        self.sig_queues.update({q_sig.name:q_sig})
        instance.setCommLinks(q_comm, q_sig)

        # Update information
        self.actors.update({name:instance})

    def createConnections(self):
        ''' Assemble links (multi or other)
            for later assignment
        '''
        for source,drain in self.tweak.connections.items():
            name = source.split('.')[0]
            #current assumption is connection goes from q_out to something(s) else
            if len(drain) > 1: #we need multiasyncqueue 
                link, endLinks = MultiLink(name+'_multi', source, drain)
                self.data_queues.update({source:link})
                for i,e in enumerate(endLinks):
                    self.data_queues.update({drain[i]:e})
            else: #single input, single output
                d = drain[0]
                d_name = d.split('.') #TODO: check if .anything, if not assume q_in
                link = Link(name+'_'+d_name[0], source, d)
                self.data_queues.update({source:link})
                self.data_queues.update({d:link})

    def assignLink(self, name, link):
        ''' Function to set up Links between actors
            for data location passing
            Actor must already be instantiated

            #NOTE: Could use this for reassigning links if actors crash?
            #TODO: Adjust to use default q_out and q_in vs being specified
        '''
        #logger.info('Assigning link {}'.format(name))
        classname = name.split('.')[0]
        linktype = name.split('.')[1]
        if linktype == 'q_out':
            self.actors[classname].setLinkOut(link)
        elif linktype == 'q_in':
            self.actors[classname].setLinkIn(link)
        elif linktype == 'watchout':
            self.actors[classname].setLinkWatch(link)
        else:
            self.actors[classname].addLink(linktype, link)

    def createLimbo(self, name):
        """ Creates Limbo w/ or w/out LMDB functionality based on {self.use_hdd}. """
        if not self.use_hdd:
            return store.Limbo(name)
        else:
            if name not in self.limbo_dict:
                self.limbo_dict[name] = store.Limbo(name, use_hdd=True, lmdb_name=self.lmdb_name)
            return self.limbo_dict[name]

    def runActor(self, actor):
        '''Run the actor continually; used for separate processes
            #TODO: hook into monitoring here?
        '''
        actor.run()

    def startWatcher(self):
        self.watcher = store.Watcher('watcher', self.createLimbo('watcher'))
        limbo = self.createLimbo('watcher') if self.use_hdd else None
        q_sig = Link('watcher_sig', self.name, 'watcher')
        self.watcher.setLinks(q_sig)
        self.sig_queues.update({q_sig.name:q_sig})

        self.p_watch = Process(target=self.watcher.run, name='watcher_process')
        self.p_watch.daemon = True
        self.p_watch.start()
        self.processes.append(self.p_watch)

    def start(self):
        logger.info('Starting processes')
        self.t = time.time()

        for p in self.processes:
            logger.info(p)
            p.start()


    def setup(self):
        for q in self.sig_queues.values():
            try:
                q.put_nowait(Spike.setup())
            except Full:
                logger.warning('Signal queue'+q.name+'is full')

    def run(self):
        if self.allowStart:
            for q in self.sig_queues.values():
                try:
                    q.put_nowait(Spike.run())
                except Full:
                    logger.warning('Signal queue'+q.name+'is full')
                    #queue full, keep going anyway TODO: add repeat trying as async task

    def quit(self):
        logger.warning('Killing child processes')

        for q in self.sig_queues.values():
            try:
                q.put_nowait(Spike.quit())
            except Full as f:
                logger.warning('Signal queue '+q.name+' full, cannot tell it to quit: {}'.format(f))

        self.processes.append(self.p_GUI)
        #self.processes.append(self.p_watch)

        for p in self.processes:
            # if p.is_alive():
            #     p.terminate()
            p.terminate()
            p.join()

        logger.warning('Actors terminated')
        print('total time ', time.time()-self.t)

        self.destroyNexus()

    async def pollQueues(self):
        """ Listens to links and processes their signals.
        
        For every communications queue connected to Nexus, a task is
        created that gets from the queue. Throughout runtime, when these
        queues output a signal, they are processed by other functions.
        At the end of runtime (when the gui has been closed), polling is 
        stopped.
        
        Returns:
            "Shutting down" (string): Notifies start() that pollQueues has completed.
        """

        self.listing = [] #TODO: Remove or rewrite
        self.actorStates = dict.fromkeys(self.actors.keys())
        if not self.tweak.hasGUI:  # Since Visual is not started, it cannot send a ready signal.
            try:
                del self.actorStates['Visual']
            except:
                pass
        polling = list(self.comm_queues.values())
        pollingNames = list(self.comm_queues.keys())
        self.tasks = []
        for q in polling:
            self.tasks.append(asyncio.ensure_future(q.get_async()))

        while not self.flags['quit']:
            done, pending = await asyncio.wait(self.tasks, return_when=concurrent.futures.FIRST_COMPLETED)
            #TODO: actually kill pending tasks
            for i,t in enumerate(self.tasks):
                if t in done or polling[i].status == 'done': #catch tasks that complete await wait/gather
                    r = polling[i].result
                    if 'GUI' in pollingNames[i]:
                        self.processGuiSignal(r, pollingNames[i])
                    else:
                        self.processActorSignal(r, pollingNames[i])
                    self.tasks[i] = (asyncio.ensure_future(polling[i].get_async()))

        self.stop_polling("quit", asyncio.get_running_loop(), polling)
        logger.warning('Shutting down polling')
        return "Shutting Down"

    def processGuiSignal(self, flag, name):
        '''Receive flags from the Front End as user input
            TODO: Not all needed
        '''
        name = name.split('_')[0]
        logger.info('Received signal from GUI: '+flag[0])
        if flag[0]:
            if flag[0] == Spike.run():
                logger.info('Begin run!')
                #self.flags['run'] = True
                self.run()
            elif flag[0] == Spike.setup():
                logger.info('Running setup')
                self.setup()
            elif flag[0] == Spike.ready():
                logger.info('GUI ready')
                self.actorStates[name] = flag[0]
            elif flag[0] == Spike.quit():
                logger.warning('Quitting the program!')
                self.flags['quit'] = True
                self.quit()
            elif flag[0] == Spike.load():
                logger.info('Loading Tweak config from file '+flag[1])
                self.loadTweak(flag[1])
            elif flag[0] == Spike.pause():
                logger.info('Pausing processes')
                # TODO. Alsoresume, reset
        else:
            logger.error('Signal received from Nexus but cannot identify {}'.format(flag))

    def processActorSignal(self, sig, name):
        if sig is not None:
            logger.info('Received signal '+str(sig[0])+' from '+name)
            if sig[0]==Spike.ready():
                self.actorStates[name.split('_')[0]] = sig[0]
                if all(val==Spike.ready() for val in self.actorStates.values()):
                    self.allowStart = True      #TODO: replace with q_sig to FE/Visual
                    logger.info('Allowing start')

                    #TODO: Maybe have flag for auto-start, else require explict command
                    # if not self.tweak.hasGUI:
                    #     self.run()

    def destroyNexus(self):
        ''' Method that calls the internal method
            to kill the process running the store (plasma server)
        '''
        logger.warning('Destroying Nexus')
        self._closeStore()
        logger.warning('Killed the central store')

    def _closeStore(self):
        ''' Internal method to kill the subprocess
            running the store (plasma sever)
        '''
        try:
            self.p_Limbo.kill()
            logger.info('Store closed successfully')
        except Exception as e:
            logger.exception('Cannot close store {0}'.format(e))

    def _startStore(self, size):
        ''' Start a subprocess that runs the plasma store
            Raises a RuntimeError exception size is undefined
            Raises an Exception if the plasma store doesn't start
        '''
        if size is None:
            raise RuntimeError('Server size needs to be specified')
        try:
            self.p_Limbo = subprocess.Popen(['plasma_store',
                              '-s', '/tmp/store',
                              '-m', str(size),
                              '-e', 'hashtable://test'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
            logger.info('Store started successfully')
        except Exception as e:
            logger.exception('Store cannot be started: {0}'.format(e))

    def stop_polling(self, stop_signal, loop, queues):
        """ Cancels outstanding tasks and fills their last request.

        Puts a string into all active queues, then cancels their 
        corresponding tasks. These tasks are not fully cancelled until 
        the next run of the event loop.

        Args:
            stop_signal (signal.signal): Signal for signal handler.
            loop (loop): Event loop for signal handler.
            queues (AsyncQueue): Comm queues for links.
        """ 

        logging.info("Received shutdown order")

        logging.info(f"Stop signal: {stop_signal}")
        shutdown_message = "SHUTDOWN"
        [q.put(shutdown_message) for q in queues]
        logging.info('Canceling outstanding tasks')
        try:
            asyncio.gather(*self.tasks)
        except asyncio.CancelledError: 
            logging.info("Gather is cancelled")

        [task.cancel() for task in self.tasks]

        cur_task = asyncio.current_task()
        cur_task.cancel()
        tasks = [task for task in self.tasks if not task.done()]
        [t.cancel() for t in tasks]
        [t.cancel() for t in tasks] #necessary in order to start cancelling tasks other than the first one

        try:
            cur_task.cancel() 
        except asyncio.CancelledError:
            logging.info("cur_task cancelled")
        
        logging.info('Polling has stopped.')


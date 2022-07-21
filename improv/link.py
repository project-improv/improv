import asyncio
import logging
from multiprocessing import Manager, cpu_count, set_start_method
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s %(message)s',
                    handlers=[logging.FileHandler("global.log"),
                              logging.StreamHandler()])

def Link(name, start, end):
    """ Function to construct a queue that Nexus uses for
    inter-process (actor) signaling and information passing.

    A Link has an internal queue that can be synchronous (put, get)
    as inherited from multiprocessing.Manager.Queue
    or asynchronous (put_async, get_async) using async executors.
    
    Args:
        See AsyncQueue constructor

    Returns:
        AsyncQueue: queue for communicating between actors and with Nexus
    """

    m = Manager()
    q = AsyncQueue(m.Queue(maxsize=0), name, start, end)
    return q

class AsyncQueue(object):
    """ Single-output and asynchronous queue class.

    Attributes:
        queue: 
        real_executor: 
        cancelled_join: boolean
        name:
        start:
        end:
        status:
        result:
        num:
        dict:
    """

    def __init__(self, q, name, start, end):
        """ Constructor for the queue class.

        Args:
            q (Queue): A queue from the Manager class
            name (str): String description of this queue
            start (str): The producer (input) actor for the queue
            end (str): The consumer (output) actor for the queue
            
        """

        self.queue = q
        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end

        self.status = 'pending'
        self.result = None
        
    def getStart(self):
        """ Gets the starting actor.
        
        The starting actor is the actor that is at the tail of the link.
        This actor is the one that gives output.

        Returns:
            start (Actor): The starting actor. 
        """

        return self.start

    def getEnd(self):
        """ Gets the ending actor.
        
        The ending actor is the actor that is at the head of the link. 
        This actor is the one that takes input.
        
        Returns:
            end (Actor): The ending actor.
        """

        return self.end

    @property
    def _executor(self):
        if not self.real_executor:
            self.real_executor = ThreadPoolExecutor(max_workers=cpu_count())
        return self.real_executor

    def __getstate__(self):
        """ Gets a dictionary of attributes. 
        
        This function gets a dictionary, with keys being the names of 
        the attributes, and values being the values of the attributes.
        
        Returns:
            self_dict (dict): A dictionary containing attributes.
        """

        self_dict = self.__dict__
        self_dict['_real_executor'] = None
        return self_dict

    def __getattr__(self, name):
        """ Gets the attribute specified by "name".

        Args:
            name (str): Name of the attribute to be returned. 

        Raises:
            AttributeError: Restricts the available attributes to a
            specific list. This error is raised if a different attribute
             of the queue is requested.
        
        TODO: 
            Don't raise this?
        
        Returns:
            (object): Value of the attribute specified by "name".
        """

        if name in ['qsize', 'empty', 'full',
                    'get', 'get_nowait', 'close']:
            return getattr(self.queue, name)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                    (self.__class__.__name__, name))

    def __repr__(self):
        """ String representation for Link.
        
        Returns:
            (str): "Link" followed by the name given in the constructor.
        """

        return 'Link '+self.name

    def put(self, item):
        """ Function wrapper for put.

        Args:
            item (object): Any item that can be sent through a queue
        """

        self.queue.put(item)

    def put_nowait(self, item):
        """ Function wrapper for put without waiting

        Args:
            item (object): Any item that can be sent through a queue
        """

        self.queue.put_nowait(item)

    async def put_async(self, item):
        """ Coroutine for an asynchronous put

        It adds the put request to the event loop and awaits.

        Args:
            item (object): Any item that can be sent through a queue

        Returns:
            Awaitable or result of the put
        """

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(self._executor, self.put, item)
        return res

    async def get_async(self):
        """ Coroutine for an asynchronous get

        It adds the get request to the event loop and awaits, setting 
        the status to pending. Once the get has returned, it returns the
        result of the get and sets its status as done.

        Returns:
            Awaitable or result of the get.
        
        Exceptions:
            Explicitly passes any exceptions to not hinder execution.
            Errors are logged with the get_async tag.
        """

        loop = asyncio.get_event_loop()
        self.status = 'pending'
        try:
            self.result = await loop.run_in_executor(self._executor, self.get)
            self.status = 'done'
            return self.result
        except Exception as e:
            logger.exception('Error in get_async: {}'.format(e))
            pass

    def cancel_join_thread(self):
        """ Function wrapper for cancel_join_thread.
        """

        self._cancelled_join = True
        self._queue.cancel_join_thread()

    def join_thread(self):
        """ Function wrapper for join_thread.
        """

        self._queue.join_thread()
        if self._real_executor and not self._cancelled_join:
            self._real_executor.shutdown()


def MultiLink(name, start, end):
    """ Function to generate links for the multi-output queue case.

    Args:
        See constructor for AsyncQueue or MultiAsyncQueue

    Returns:
        MultiAsyncQueue: Producer end of the queue
        List: AsyncQueues for consumers
    """
    
    m = Manager()

    q_out = []
    for endpoint in end:
        q = AsyncQueue(m.Queue(maxsize=0), name, start, endpoint)
        q_out.append(q)

    q = MultiAsyncQueue(m.Queue(maxsize=0), q_out, name, start, end)

    return q, q_out

class MultiAsyncQueue(AsyncQueue):
    """ Extension of AsyncQueue to have multiple endpoints.

    Inherits from AsyncQueue. 
    A single producer queue's 'put' is copied to multiple consumer's 
    queues, q_in is the producer queue, q_out are the consumer queues.
    
    TODO: 
        Test the async nature of this group of queues
    """

    def __init__(self, q_in, q_out, name, start, end):
        self.queue = q_in
        self.output = q_out

        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end[0]
        self.status = 'pending'
        self.result = None

    def __repr__(self):
        return 'MultiLink ' + self.name

    def __getattr__(self, name):
        # Remove put and put_nowait and define behavior specifically
        #TODO: remove get capability?
        if name in ['qsize', 'empty', 'full',
                    'get', 'get_nowait', 'close']:
            return getattr(self.queue, name)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                    (self.__class__.__name__, name))

    def put(self, item):
        for q in self.output:
            q.put(item)

    def put_nowait(self, item):
        for q in self.output:
            q.put_nowait(item)
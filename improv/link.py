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

def Link(name, start, end, limbo):
    """ Abstract constructor for a queue that Nexus uses for
    inter-process (actor) signaling and information passing.

    A Link has an internal queue that can be synchronous (put, get)
    as inherited from multiprocessing.Manager.Queue
    or asynchronous (put_async, get_async) using async executors.

    Attributes:
        m: Manager from multiprocessing
        q: Queue from the Manager m

    Returns:
        AsyncQueue: queue for communicating between actors and with Nexus
    """

    m = Manager()
    q = AsyncQueue(m.Queue(maxsize=0), name, start, end, limbo)
    return q

class AsyncQueue(object):
    """ Multi-output and asynchronous queue class.

        Attributes:
            queue: 
            real_executor: 
            cancelled_join: boolean
            name:
            start:
            end:
            status:
            result:
            limbo:
            num:
            dict:
    """

    def __init__(self, q, name, start, end, limbo):
        """ Constructor for the queue class.

        Args:
            q (Queue): A queue from the Manager class
            name (str): String description of this queue
            start (str): The producer (input) actor for the queue
            end (str): The consumer (output) actor for the queue
            limbo (improv.store.Limbo): Connection to the store for logging in case of future replay

        #TODO: Rewrite to avoid limbo and use logger files instead
        """
        self.queue = q
        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end

        self.status = 'pending'
        self.result = None
        
        self.limbo = limbo
        self.num = 0

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    @property
    def _executor(self):
        if not self.real_executor:
            self.real_executor = ThreadPoolExecutor(max_workers=cpu_count())
        return self.real_executor

    def __getstate__(self):
        self_dict = self.__dict__
        self_dict['_real_executor'] = None
        return self_dict

    def __getattr__(self, name):
        if name in ['qsize', 'empty', 'full',
                    'get', 'get_nowait', 'close']:
            return getattr(self.queue, name)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                    (self.__class__.__name__, name))

    def __repr__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return 'Link '+self.name #+' From: '+self.start+' To: '+self.end

    def put(self, item):
        self.log_to_limbo(item)
        self.queue.put(item)

    def put_nowait(self, item):
        self.log_to_limbo(item)
        self.queue.put_nowait(item)

    async def put_async(self, item):
        loop = asyncio.get_event_loop()
        self.log_to_limbo(item)
        res = await loop.run_in_executor(self._executor, self.put, item)
        return res

    async def get_async(self):
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
        self._cancelled_join = True
        self._queue.cancel_join_thread()

    def join_thread(self):
        self._queue.join_thread()
        if self._real_executor and not self._cancelled_join:
            self._real_executor.shutdown()

    def log_to_limbo(self, item):
        if self.limbo is not None:
            self.limbo.put(item, f'q__{self.start}__{self.num}')
            self.num += 1


def MultiLink(name, start, end, limbo):
    ''' End is a list

        Return a MultiAsyncQueue as q (for producer) and list of AsyncQueues as q_out (for consumers)
    '''
    m = Manager()

    q_out = []
    for endpoint in end:
        q = AsyncQueue(m.Queue(maxsize=0), name, start, endpoint, limbo=limbo)
        q_out.append(q)

    q = MultiAsyncQueue(m.Queue(maxsize=0), q_out, name, start, end)

    return q, q_out

class MultiAsyncQueue(AsyncQueue):
    ''' Extension of AsyncQueue created by Link to have multiple endpoints.
        A single producer queue's 'put' is copied to multiple consumer's queues
        q_in is the producer queue, q_out are the consumer queues

        #TODO: test the async nature of this group of queues
    '''
    def __init__(self, q_in, q_out, name, start, end):
        self.queue = q_in
        self.output = q_out

        self.real_executor = None
        self.cancelled_join = False

        self.name = name
        self.start = start
        self.end = end[0] #somewhat arbitrary endpoint naming
        self.status = 'pending'
        self.result = None

    def __repr__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return 'MultiLink '+self.name

    def __getattr__(self, name):
        # Remove put and put_nowait and define behavior specifically
        #TODO: remove get capability
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
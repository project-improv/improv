import concurrent
import time
import asyncio
import math

from improv.link import Link
from improv.store import Store
import subprocess

def clean_list_print(lst):
    print("\n=======================\n")
    for el in lst:
        print(el)
        print("\n")
    print("\n=======================\n")

def setup_store():
    """ Fixture to set up the store subprocess with 10 mb.

    This fixture runs a subprocess that instantiates the store with a 
    memory of 10 megabytes. It specifies that "/tmp/store/" is the 
    location of the store socket.

    Yields:
        Store: An instance of the store.

    TODO:
        Figure out the scope.
    """

    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    store = Store(store_loc = "/tmp/store")
    return store

async def pollQueues(links):
    tasks = []
    for link in links:
        tasks.append(asyncio.create_task(link.get_async()))
    
    links_cpy = links
    t_0 = time.perf_counter()
    t_1 = time.perf_counter()
    print("time get")
    cur_time = 0
    while(t_1 - t_0 < 5):

        #need to add something to the queue such that asyncio.wait returns

        links[0].put("Message")
        done, pending = await asyncio.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)
        for i, t in enumerate(tasks):
            if t in done:
                pass
                tasks[i] = asyncio.create_task(links_cpy[i].get_async())

        t_1 = time.perf_counter()
        
        if (math.floor(t_1 - t_0) != cur_time):
            print(math.floor(t_1 - t_0))
            cur_time = math.floor(t_1 - t_0)
    
    print("All tasks prior to stop polling: \n")
    clean_list_print([task for task in tasks])

    loop = asyncio.get_running_loop()
    return stop_polling(tasks, loop, links)



def start():
    links = [Link(f"Link {i}", f"start {i}", f"end {i}", setup_store()) for i in range(4)]
    loop = asyncio.get_event_loop()
    print('RUC loop')
    res = loop.run_until_complete(pollQueues(links))
    print(f"RES: {res}")

    print("**********************\nAll tasks at the end of execution:")
    clean_list_print(res)
    print("**********************")
    print(f"Loop: {loop}")
    loop.close()
    print(f"Loop: {loop}")

def stop_polling(tasks, loop, links):
    #asyncio.gather(*tasks)
    print("Cancelling")

    [lnk.put("msg") for lnk in links]

    # [task.cancel() for task in tasks]
    # [task.cancel() for task in tasks]

    # [lnk.put("msg") for lnk in links]

    print("All tasks: \n")
    clean_list_print([task for task in tasks])
    print("Pending:\n")
    clean_list_print([task for task in tasks if not task.done()])
    print("Cancelled: \n")
    clean_list_print([task for task in tasks if task.cancelled()])
    print("Pending and cancelled: \n")
    clean_list_print([task for task in tasks if not task.done() and task.cancelled()]) 

    return [task for task in tasks] 

if __name__ == "__main__":
    start()
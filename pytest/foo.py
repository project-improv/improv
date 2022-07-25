from signal import SIGTERM, SIGINT
import asyncio
import subprocess

from improv.link import Link
from improv.actor import Actor
from improv.store import Limbo

def setup():
    p = subprocess.Popen(
        ['plasma_store', '-s', '/tmp/store', '-m', str(10000000)],\
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lmb = Limbo(store_loc = "/tmp/store")
    
    acts = [Actor(f"Test {i}") for i in range(2)]
    lnk = Link("Link 1", acts[0].name, acts[1].name, lmb)
    return lnk


async def sayhi(n=1):
    for i in range(n):
        print("HI ")

    await str(n)
    
async def main():
    lnk = setup()
    tasks = []
    tasks.append(asyncio.create_task(lnk.get_async()))

    print("HELLOHELLO")
    done, pending = await asyncio.wait(tasks, return_when = asyncio.FIRST_COMPLETED)
    

    print(str((done, pending)))

    # lnk.put("ASK")

    # try:
    #     await asyncio.wait_for(asyncio.create_task(lnk.get_async()), 2.0)
    # except:
    #     pass

    # try:
    #     task = asyncio.create_task(lnk.get_async())
    #     res = await asyncio.wait_for(task, 1.0)
    # except asyncio.TimeoutError:
    #     res1 = "FAILED"

    # res = await asyncio.wait(task, return_when = asyncio.FIRST_COMPLETED)

    # task = asyncio.create_task(lnk.get_async())
    # asyncio.wait(list(task), return_when=asyncio.FIRST_COMPLETED)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print("BYEBYE")
    loop.stop()
    loop.close()
    
    # tsk = asyncio.ensure_future(main())

    # for signal in [SIGINT, SIGTERM]:
    #     loop.add_signal_handler(signal, tsk.cancel)
    # try:
    #     loop.run_until_complete(tsk)
    # finally:
    #     loop.close()

    



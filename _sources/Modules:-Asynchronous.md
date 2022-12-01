Asynchronous code based on `asyncio` could be run as a module using `nexus.module.AsyncRunManager`. This page explains the code from `analysis_async.py`. This module receives raw frame data and performs a fake analysis.

### Initialization
The `run` function is called upon initialization like a typical module. Here, instead of starting a `RunManager`, we have to start a new event loop to run `self.arun`. Furthermore, there is a new `asyncio`-compatible `Queue`, which keeps incoming objects in a FIFO manner. As expected, the `runMethod` (`self.get_frame`) is no longer called repeatedly but will initially be added to the event loop and awaited.

```python
def run(self):
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    self.loop = asyncio.get_event_loop()

    self.aqueue = asyncio.Queue()
    self.loop.run_until_complete(self.arun())

async def arun(self):
    async with AsyncRunManager(self.name, self.get_frame, self.setup, self.q_sig, self.q_comm) as rm:
        logger.info(rm)
```
### Receiving Frames
When it's first started, `get_frame` also starts `analysis`. Like other modules, it gets the object ID from the `Queue`. Note that the method is now `get_async` not `get`. As a result, the `get_async` function is only taking the control when there's a new object and no longer blocks code execution. The object is then passed into our asynchronous `Queue`.

```python
async def get_frame(self):
    asyncio.ensure_future(self.analysis(), loop=self.loop)

    while True:
        if self.aqueue.qsize() > 0:
            asyncio.ensure_future(self.analysis(), loop=self.loop)

        obj_id = await self.q_in.get_async()
        if obj_id is not None:
            frame = self.client.getID(obj_id[0][str(self.frame_number)])
            await self.aqueue.put([frame, self.frame_number, time.time()])
            self.frame_number += 1
```
### Analysis
This function simulates an I/O intensive task or outsourcing of analysis to other programs or processes. When objects begin to pile-up in the asynchronous `Queue`, `get_frame` adds a new `analysis` future into the event loop, allowing in concurrent processing.

```python
async def analysis(self):
    while True:
        frame = await self.aqueue.get()
        t = 0.15 * random() + 0.1
        await asyncio.sleep(t)
```
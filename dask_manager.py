import asyncio
from dask.distributed import Scheduler, Worker, Client
from contextlib import AsyncExitStack
import yaml

        
class Manager:

    def __init__(self, workers):
        self.workers = workers

    def distributed_run(self, fnpool):
        async def f():
            async with Scheduler() as sched:
                async with AsyncExitStack() as stack:
                    ws = []
                    for i in range(self.workers):
                        ws.append(await stack.enter_async_context(Worker(sched.address)))
                    async with Client(sched.address, asynchronous=True) as client:
                        futures = []
                        for i in range(len(fnpool)):
                            futures.append(client.submit(fnpool[i]))
                        result = await client.gather(futures)
                        return result  # savefn for each experiment so dask doesnt write over
        return asyncio.get_event_loop().run_until_complete(f())


# Copyright (c) 2021 Damon Wischik. See LICENSE for permissions.

import asyncio, heapq
from eventsim import EventSimulator
import time
from asyncio import futures

class ProcessorSharingQueue:
    def __init__(self, service_rate=1, loop=None):
        self._service_rate = service_rate
        self._queue = []
        self._loop = loop if loop else asyncio.get_event_loop()
        self._done = None
        self._work_done = 0
        self._last_time = self._loop.time()
    def process(self, work):
        t = self._advance_clock()
        fut = self._loop.create_future()
        w = work / self._service_rate
        heapq.heappush(self._queue, (self._work_done+w, t, fut))
        if self._done:
            self._done.cancel()
        self._schedule()
        return fut
    def complete(self):
        t = self._advance_clock()
        (_, tstart, fut) = heapq.heappop(self._queue)
        fut.set_result(t - tstart)
        self._schedule()
    def _advance_clock(self):
        t = self._loop.time()
        if self._queue:
            self._work_done += (t - self._last_time) / len(self._queue)
        self._last_time = t
        return t
    def _schedule(self):
        if not self._queue:
            self._done = None
        else:
            w,_,_ = self._queue[0]
            dt = (w - self._work_done) * len(self._queue)
            self._done = self._loop.call_later(dt, self.complete)


class FIFOQueue:
    def __init__(self, service_rate=1, loop=None):
        self._service_rate = service_rate
        self._queue = []
        self._loop = loop if loop else asyncio.get_event_loop()
        self._done = None
    def process(self, work):
        fut = self._loop.create_future()
        w = work / self._service_rate
        self._queue.append((w, fut))
        if not self._done:
            print(f"will finish at {self._loop.time()} + {w} seconds")
            self._done = self._loop.call_later(w, self.complete)
        return fut
    def complete(self):
        w,fut = self._queue[0]
        fut.set_result(w)
        self._queue = self._queue[1:]
        if self._queue:
            w,_ = self._queue[0]
            self._done = self._loop.call_later(w, self.complete)
        else:
            self._done = None


#------------------------------------------------

# loop = EventSimulator()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# asyncio._set_running_loop(loop)

async def simulated_sleep(delay, loop=None):
    future = loop.create_future()
    loop.call_later(delay, lambda: future.set_result(None))
    await future

async def process(text: str):
    # This is compute heavy task
    start = time.time()
    while True:
        if time.time() - start >= 1:
            break
    print(f'{text} finished at {loop.time()}')

async def generate():
    for r in range(10):
        print("Request {} created at {}".format(r, loop.time()))
        asyncio.create_task(process("Request {}".format(r)))
        await asyncio.sleep(0.5)
        
# asyncio.create_task(generate())

q = FIFOQueue(service_rate=1)
            
async def queueing_job(i=1):
    print(loop.time(), "Start job {}".format(i))
    await asyncio.sleep(i)
    print(loop.time(), "Sending job {}".format(i))
    xmit = q.process(work=4)
    await xmit
    print(loop.time(), "Done job {} in time {}".format(i, xmit.result()))

# t1 = asyncio.ensure_future(queueing_job(1))
# t2 = asyncio.ensure_future(queueing_job(3))

# loop.run_forever()
loop.run_until_complete(generate())
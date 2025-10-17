import asyncio, time
from typing import Callable, Any
from contextlib import asynccontextmanager

async def timeit_async(func: Callable[..., Any], *args, **kwargs):
    t0 = time.time()
    res = await func(*args, **kwargs)
    td = (time.time() - t0) * 1000
    return res, int(td)

async def with_timeout(coro, timeout_sec, default=None):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except Exception as e:
        return default

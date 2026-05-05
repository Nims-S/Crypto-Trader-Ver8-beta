from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PromptJob:
    name: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


async def batch_prompts_async(
    jobs: Iterable[PromptJob],
    *,
    client: Callable[[str], Any],
    max_concurrency: int = 4,
) -> dict[str, Any]:
    jobs_list = list(jobs)
    if not jobs_list:
        return {}

    sem = asyncio.Semaphore(max(1, int(max_concurrency or 1)))

    async def _worker(job: PromptJob):
        async with sem:
            if inspect.iscoroutinefunction(client):
                result = await client(job.prompt)
            else:
                result = await asyncio.to_thread(client, job.prompt)
            return job.name, result

    results = await asyncio.gather(*(_worker(job) for job in jobs_list))
    return {k: v for k, v in results}


def batch_prompts_sync(
    jobs: Iterable[PromptJob],
    *,
    client: Callable[[str], Any],
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Thread-safe sync wrapper."""

    def runner(out):
        out.append(asyncio.run(batch_prompts_async(jobs, client=client, max_concurrency=max_concurrency)))

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            result = []
            t = threading.Thread(target=runner, args=(result,))
            t.start()
            t.join()
            return result[0]
    except RuntimeError:
        pass

    return asyncio.run(batch_prompts_async(jobs, client=client, max_concurrency=max_concurrency))

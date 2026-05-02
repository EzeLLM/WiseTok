"""
asyncio patterns: TaskGroup, gather, semaphore-limited concurrency,
async context managers, async generators, Queue producer/consumer.
"""
import asyncio
from typing import List, AsyncGenerator, Any
from contextlib import asynccontextmanager
import random
import time


class AsyncResourcePool:
	"""Async context manager for resource pooling."""

	def __init__(self, resource_name: str, capacity: int = 5):
		self.resource_name = resource_name
		self.capacity = capacity
		self.semaphore = asyncio.Semaphore(capacity)
		self.active_count = 0

	@asynccontextmanager
	async def acquire(self):
		"""Acquire a resource from the pool."""
		async with self.semaphore:
			self.active_count += 1
			print(f"[{self.resource_name}] Acquired ({self.active_count}/{self.capacity})")
			try:
				yield self
			finally:
				self.active_count -= 1
				print(f"[{self.resource_name}] Released ({self.active_count}/{self.capacity})")


async def slow_network_request(url: str, delay: float = 1.0) -> dict:
	"""Simulate a network request."""
	print(f"  → Fetching {url}")
	await asyncio.sleep(delay)
	return {"url": url, "status": 200, "data": f"Response from {url}"}


async def fetch_with_timeout(url: str, timeout: float = 2.0) -> dict:
	"""Fetch with timeout handling."""
	try:
		result = await asyncio.wait_for(
			slow_network_request(url, random.uniform(0.5, 1.5)),
			timeout=timeout
		)
		return result
	except asyncio.TimeoutError:
		print(f"  ✗ Timeout for {url}")
		return {"url": url, "status": 408, "error": "Timeout"}


async def bounded_fetch_batch(
	urls: List[str],
	max_concurrent: int = 3
) -> List[dict]:
	"""Fetch multiple URLs with bounded concurrency using semaphore."""
	semaphore = asyncio.Semaphore(max_concurrent)

	async def fetch_bounded(url: str) -> dict:
		async with semaphore:
			return await fetch_with_timeout(url)

	tasks = [fetch_bounded(url) for url in urls]
	results = await asyncio.gather(*tasks, return_exceptions=True)
	return results


async def fetch_with_taskgroup(urls: List[str]) -> List[dict]:
	"""Fetch using TaskGroup (Python 3.11+)."""
	results = []

	try:
		async with asyncio.TaskGroup() as tg:
			async def fetch_and_append(url: str):
				result = await fetch_with_timeout(url)
				results.append(result)

			for url in urls:
				tg.create_task(fetch_and_append(url))
	except ExceptionGroup as eg:
		print(f"  ✗ ExceptionGroup: {eg.exceptions}")

	return results


async def async_data_generator(
	count: int = 5,
	delay: float = 0.5
) -> AsyncGenerator[dict, None]:
	"""Async generator that yields data over time."""
	for i in range(count):
		await asyncio.sleep(delay)
		yield {
			"id": i,
			"timestamp": time.time(),
			"value": random.randint(1, 100)
		}


async def consume_async_generator():
	"""Consume data from async generator."""
	print("\nConsuming async generator:")
	async for item in async_data_generator(5, 0.3):
		print(f"  Got item: {item['id']} = {item['value']}")


class AsyncQueue:
	"""Producer/consumer pattern with asyncio.Queue."""

	def __init__(self, queue_size: int = 10):
		self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
		self.stats = {"produced": 0, "consumed": 0}

	async def producer(self, producer_id: int, count: int = 5):
		"""Producer task."""
		for i in range(count):
			item = {"producer": producer_id, "item": i, "value": random.randint(1, 100)}
			await self.queue.put(item)
			self.stats["produced"] += 1
			print(f"  [P{producer_id}] Produced item {i}")
			await asyncio.sleep(random.uniform(0.1, 0.3))

	async def consumer(self, consumer_id: int):
		"""Consumer task."""
		while True:
			try:
				item = await asyncio.wait_for(self.queue.get(), timeout=5.0)
				if item is None:  # Sentinel value
					break
				self.stats["consumed"] += 1
				print(f"  [C{consumer_id}] Consumed: {item}")
				await asyncio.sleep(random.uniform(0.1, 0.4))
				self.queue.task_done()
			except asyncio.TimeoutError:
				break

	async def run_test(self, num_producers: int = 2, num_consumers: int = 2):
		"""Run producers and consumers concurrently."""
		print("\nRunning queue test:")
		async with asyncio.TaskGroup() as tg:
			# Start producers
			for p_id in range(num_producers):
				tg.create_task(self.producer(p_id, count=3))

			# Start consumers
			for c_id in range(num_consumers):
				tg.create_task(self.consumer(c_id))

		# Wait for queue to drain
		await self.queue.join()
		print(f"  Queue stats: produced={self.stats['produced']}, consumed={self.stats['consumed']}")


async def retry_with_backoff(
	coro,
	max_retries: int = 3,
	backoff_base: float = 1.0
) -> Any:
	"""Retry a coroutine with exponential backoff."""
	for attempt in range(1, max_retries + 1):
		try:
			return await coro()
		except Exception as e:
			if attempt == max_retries:
				raise
			wait_time = backoff_base ** (attempt - 1)
			print(f"  ⚠ Attempt {attempt} failed, retrying in {wait_time}s: {e}")
			await asyncio.sleep(wait_time)


async def main():
	print("=== Asyncio Patterns Demo ===\n")

	# 1. Bounded concurrency with semaphore
	print("1. Bounded concurrent fetches (max 3):")
	urls = [f"https://api.example.com/endpoint{i}" for i in range(6)]
	results = await bounded_fetch_batch(urls, max_concurrent=3)
	print(f"   Fetched {len(results)} URLs\n")

	# 2. TaskGroup (Python 3.11+)
	print("2. Using TaskGroup:")
	urls_small = [f"https://api.example.com/endpoint{i}" for i in range(3)]
	results_tg = await fetch_with_taskgroup(urls_small)
	print(f"   TaskGroup collected {len(results_tg)} results\n")

	# 3. Async generator consumption
	await consume_async_generator()

	# 4. Queue producer/consumer
	queue_test = AsyncQueue(queue_size=10)
	await queue_test.run_test(num_producers=2, num_consumers=2)

	# 5. Resource pool with async context manager
	print("\nResource pool example:")
	pool = AsyncResourcePool("DatabaseConnection", capacity=3)

	async def use_resource(task_id: int):
		async with pool.acquire():
			await asyncio.sleep(random.uniform(0.5, 1.5))

	async with asyncio.TaskGroup() as tg:
		for i in range(7):
			tg.create_task(use_resource(i))

	# 6. Retry with backoff
	print("\nRetry with backoff:")
	call_count = [0]

	async def unreliable_operation():
		call_count[0] += 1
		if call_count[0] < 2:
			raise ConnectionError("Connection failed")
		return "Success on attempt 2"

	result = await retry_with_backoff(unreliable_operation, max_retries=3, backoff_base=0.5)
	print(f"   Result: {result}")


if __name__ == "__main__":
	asyncio.run(main())

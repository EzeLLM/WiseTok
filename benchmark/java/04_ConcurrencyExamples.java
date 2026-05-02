package com.example.wisetok.concurrent;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.IntStream;

/**
 * Demonstrates Java concurrency primitives:
 * ExecutorService, ReentrantLock, synchronized, AtomicInteger, ConcurrentHashMap, CompletableFuture.
 */
public class ConcurrencyExamples {

    /**
     * ExecutorService with fixed thread pool for parallel tokenizer merges.
     */
    public static List<Integer> parallelMerges(List<TokenPair> pairs, int numThreads) {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Integer>> futures = new ArrayList<>();

        for (TokenPair pair : pairs) {
            futures.add(executor.submit(() -> {
                // Simulate merge operation
                Thread.sleep(10);
                return pair.getId() * 2;
            }));
        }

        List<Integer> results = new ArrayList<>();
        for (Future<Integer> future : futures) {
            try {
                results.add(future.get(5, TimeUnit.SECONDS));
            } catch (TimeoutException e) {
                future.cancel(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();
        try {
            executor.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        return results;
    }

    /**
     * ForkJoinPool for recursive merge tree processing.
     */
    public static class MergeTreeTask extends RecursiveTask<Long> {
        private static final int THRESHOLD = 100;
        private final int[] tokens;
        private final int start, end;

        public MergeTreeTask(int[] tokens, int start, int end) {
            this.tokens = tokens;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Long compute() {
            if (end - start <= THRESHOLD) {
                // Base case: sequential merge
                long sum = 0;
                for (int i = start; i < end; i++) {
                    sum += tokens[i];
                }
                return sum;
            } else {
                // Recursive case: split and merge
                int mid = (start + end) / 2;
                MergeTreeTask leftTask = new MergeTreeTask(tokens, start, mid);
                MergeTreeTask rightTask = new MergeTreeTask(tokens, mid, end);
                leftTask.fork();
                long rightResult = rightTask.compute();
                long leftResult = leftTask.join();
                return leftResult + rightResult;
            }
        }
    }

    /**
     * ReentrantLock for synchronizing access to shared merge state.
     */
    public static class TokenizerState {
        private final ReentrantLock mergeCountLock = new ReentrantLock();
        private int mergeCount = 0;

        public void incrementMergeCount() {
            mergeCountLock.lock();
            try {
                mergeCount++;
            } finally {
                mergeCountLock.unlock();
            }
        }

        public int getMergeCount() {
            mergeCountLock.lock();
            try {
                return mergeCount;
            } finally {
                mergeCountLock.unlock();
            }
        }

        /**
         * Trylock with timeout for non-blocking acquisition.
         */
        public boolean tryIncrementWithTimeout(long timeoutMs) {
            try {
                if (mergeCountLock.tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
                    try {
                        mergeCount++;
                        return true;
                    } finally {
                        mergeCountLock.unlock();
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }

    /**
     * Synchronized methods and blocks.
     */
    public static class SynchronizedTokenStore {
        private final Map<String, Integer> tokenCounts = new HashMap<>();

        public synchronized void addToken(String token) {
            tokenCounts.merge(token, 1, Integer::sum);
        }

        public synchronized Integer getCount(String token) {
            return tokenCounts.getOrDefault(token, 0);
        }

        public void processTokens(List<String> tokens) {
            // Minimize synchronized block scope
            synchronized (this) {
                for (String token : tokens) {
                    tokenCounts.merge(token, 1, Integer::sum);
                }
            }
        }
    }

    /**
     * AtomicInteger for lock-free counting.
     */
    public static class AtomicTokenCounter {
        private final AtomicInteger count = new AtomicInteger(0);
        private final AtomicLong totalBytes = new AtomicLong(0L);

        public void incrementCount() {
            count.incrementAndGet();
        }

        public void addBytes(long bytes) {
            totalBytes.addAndGet(bytes);
        }

        public int getCount() {
            return count.get();
        }

        public long getTotalBytes() {
            return totalBytes.get();
        }

        public int compareAndSwap(int expect, int update) {
            return count.compareAndExchange(expect, update);
        }

        public void reset() {
            count.set(0);
            totalBytes.set(0);
        }
    }

    /**
     * ConcurrentHashMap for thread-safe, non-blocking map operations.
     */
    public static class ConcurrentTokenPairCounts {
        private final ConcurrentHashMap<String, AtomicLong> pairCounts = new ConcurrentHashMap<>();

        public void recordPair(String pair) {
            pairCounts.computeIfAbsent(pair, k -> new AtomicLong(0))
                    .incrementAndGet();
        }

        public long getPairCount(String pair) {
            AtomicLong count = pairCounts.get(pair);
            return count != null ? count.get() : 0L;
        }

        public Map<String, Long> getAllCounts() {
            ConcurrentHashMap<String, Long> result = new ConcurrentHashMap<>();
            pairCounts.forEach((k, v) -> result.put(k, v.get()));
            return result;
        }

        /**
         * Bulk operation with forEach.
         */
        public void forEachPair(java.util.function.BiConsumer<String, Long> action) {
            pairCounts.forEach((pair, count) -> action.accept(pair, count.get()));
        }
    }

    /**
     * ReadWriteLock for many-readers, few-writers scenario.
     */
    public static class ReadWriteTokenizer {
        private final ReadWriteLock lock = new ReentrantReadWriteLock();
        private Map<String, Integer> merges = new HashMap<>();

        public void addMerge(String pair, Integer id) {
            lock.writeLock().lock();
            try {
                merges.put(pair, id);
            } finally {
                lock.writeLock().unlock();
            }
        }

        public Integer getMergeId(String pair) {
            lock.readLock().lock();
            try {
                return merges.get(pair);
            } finally {
                lock.readLock().unlock();
            }
        }

        public int getTotalMerges() {
            lock.readLock().lock();
            try {
                return merges.size();
            } finally {
                lock.readLock().unlock();
            }
        }
    }

    /**
     * CompletableFuture with multiple completion strategies.
     */
    public static CompletableFuture<List<String>> trainAsync(List<String> corpus) {
        return CompletableFuture.supplyAsync(() -> {
            // Simulate training
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return corpus.subList(0, Math.min(5, corpus.size()));
        }).exceptionally(ex -> {
            System.err.println("Training failed: " + ex.getMessage());
            return Collections.emptyList();
        });
    }

    /**
     * CompletableFuture.allOf for concurrent batch operations.
     */
    public static CompletableFuture<Integer> trainMultipleTokenizers(List<List<String>> corpora) {
        List<CompletableFuture<Integer>> futures = new ArrayList<>();

        for (List<String> corpus : corpora) {
            futures.add(
                CompletableFuture.supplyAsync(() -> corpus.size())
                    .thenApply(size -> size * 2)
            );
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> futures.stream()
                    .mapToInt(f -> f.join())
                    .sum());
    }

    /**
     * Demonstrate barrier/latch coordination.
     */
    public static void demonstrateCountDownLatch(int numWorkers) throws InterruptedException {
        CountDownLatch startSignal = new CountDownLatch(1);
        CountDownLatch doneSignal = new CountDownLatch(numWorkers);

        for (int i = 0; i < numWorkers; i++) {
            new Thread(() -> {
                try {
                    startSignal.await();  // Wait for start signal
                    // Do work
                    doneSignal.countDown();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
        }

        Thread.sleep(100);
        startSignal.countDown();  // Signal all workers to start
        doneSignal.await();  // Wait for all workers to finish
    }

    /**
     * Demonstrate CyclicBarrier for phased execution.
     */
    public static void demonstrateCyclicBarrier(int numPhases, int numWorkers) throws InterruptedException {
        CyclicBarrier barrier = new CyclicBarrier(numWorkers, () -> {
            System.out.println("Phase completed");
        });

        ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
        for (int i = 0; i < numWorkers; i++) {
            executor.submit(() -> {
                try {
                    for (int phase = 0; phase < numPhases; phase++) {
                        // Do work
                        barrier.await();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }

        executor.shutdown();
    }

    static class TokenPair {
        private final int id;
        private final String left;
        private final String right;

        TokenPair(int id, String left, String right) {
            this.id = id;
            this.left = left;
            this.right = right;
        }

        int getId() { return id; }
    }
}

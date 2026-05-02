package com.example.wisetok.util;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Demonstrates advanced Java generics with bounded type parameters,
 * Stream API (map, filter, flatMap, reduce, collect), Optional, and CompletableFuture chains.
 */
public class GenericsAndStreams {

    /**
     * Generic method with bounded wildcard: ? extends T
     * Process a collection of items that extend a base type.
     */
    public static <T> List<String> processItems(Collection<? extends T> items) {
        return items.stream()
                .map(Object::toString)
                .collect(Collectors.toList());
    }

    /**
     * Generic method with lower bound: ? super T
     * Accepts consumer that can accept T or any supertype of T.
     */
    public static <T> void consumeItems(List<T> items, java.util.function.Consumer<? super T> consumer) {
        items.forEach(consumer);
    }

    /**
     * Bounded type parameter: <T extends Number>
     * Work with numeric types only.
     */
    public static <T extends Number> double sumNumbers(List<T> numbers) {
        return numbers.stream()
                .mapToDouble(Number::doubleValue)
                .sum();
    }

    /**
     * Multiple bounds: <T extends Comparable<T> & Cloneable>
     * Type T must implement both Comparable and Cloneable.
     */
    public static <T extends Comparable<T> & Cloneable> T findMax(List<T> items) {
        return items.stream()
                .max(Comparator.naturalOrder())
                .orElse(null);
    }

    /**
     * Nested generic bounds for complex data structures.
     */
    public static <K, V extends Comparable<V>> Map<K, V> sortByValue(Map<K, V> map) {
        return map.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(Collectors.toLinkedHashMap(Map.Entry::getKey, Map.Entry::getValue,
                    (e1, e2) -> e1, LinkedHashMap::new));
    }

    /**
     * Stream API: filter and map operations.
     */
    public static List<String> filterAndTransform(List<String> items, String prefix) {
        return items.stream()
                .filter(s -> !s.isEmpty())
                .filter(s -> s.startsWith(prefix))
                .map(String::toUpperCase)
                .sorted()
                .collect(Collectors.toList());
    }

    /**
     * Stream API: flatMap to flatten nested structures.
     */
    public static List<String> flattenTokens(List<List<String>> nested) {
        return nested.stream()
                .flatMap(List::stream)
                .filter(s -> s.length() > 0)
                .distinct()
                .collect(Collectors.toList());
    }

    /**
     * Stream API: reduce to combine elements.
     */
    public static String concatenateWithSeparator(List<String> items, String separator) {
        return items.stream()
                .reduce((a, b) -> a + separator + b)
                .orElse("");
    }

    /**
     * Stream API: collect into custom structure.
     */
    public static Map<Integer, List<String>> groupByLength(List<String> items) {
        return items.stream()
                .collect(Collectors.groupingBy(
                    String::length,
                    Collectors.toList()
                ));
    }

    /**
     * Optional handling and chaining.
     */
    public static Optional<String> findLongestToken(List<String> tokens) {
        return tokens.stream()
                .max(Comparator.comparingInt(String::length));
    }

    /**
     * Optional with orElse, orElseThrow, orElseGet.
     */
    public static String getTokenOrDefault(Optional<String> token, String defaultValue) {
        return token
                .map(String::toUpperCase)
                .filter(s -> !s.isEmpty())
                .orElse(defaultValue);
    }

    /**
     * CompletableFuture chains: thenApply, thenCompose, allOf.
     */
    public static CompletableFuture<Integer> trainTokenizerAsync(List<String> corpus) {
        return CompletableFuture.supplyAsync(() -> corpus.size())
                .thenApply(size -> size * 2)  // thenApply: transform result
                .thenCompose(count ->
                    CompletableFuture.supplyAsync(() -> count + 1000))  // thenCompose: chain futures
                .exceptionally(ex -> {
                    System.err.println("Training failed: " + ex.getMessage());
                    return 0;
                });
    }

    /**
     * CompletableFuture.allOf for parallel async operations.
     */
    @SafeVarargs
    public static <T> CompletableFuture<List<T>> allOf(CompletableFuture<T>... futures) {
        return CompletableFuture.allOf(futures)
                .thenApply(v -> Arrays.stream(futures)
                        .map(CompletableFuture::join)
                        .collect(Collectors.toList()));
    }

    /**
     * Practical example: merge token pairs asynchronously.
     */
    public static CompletableFuture<String> mergeTokenPair(String left, String right) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(10);
                return left + right;
            } catch (InterruptedException e) {
                return null;
            }
        });
    }

    /**
     * Chaining multiple async operations.
     */
    public static CompletableFuture<Integer> processTokenSequence(List<String> tokens) {
        return tokens.stream()
                .reduce(
                    CompletableFuture.completedFuture(0),
                    (acc, token) -> acc.thenCombine(
                        CompletableFuture.supplyAsync(() -> token.length()),
                        Integer::sum
                    ),
                    (cf1, cf2) -> cf1.thenCombine(cf2, Integer::sum)
                );
    }

    /**
     * Variance example: covariant (extends) vs contravariant (super).
     */
    public static void demonstrateVariance() {
        List<? extends String> covariant = Arrays.asList("a", "b", "c");
        covariant.stream().forEach(System.out::println);

        List<? super String> contravariant = new ArrayList<>(Arrays.asList("x", "y"));
        contravariant.add("z");
    }

    /**
     * TypeToken pattern for working with generics at runtime.
     */
    public static class TypeToken<T> {
        private final Class<?> type;

        public TypeToken() {
            this.type = ((java.lang.reflect.ParameterizedType) getClass()
                    .getGenericSuperclass()).getActualTypeArguments()[0];
        }

        public Class<?> getRawType() {
            return type;
        }
    }
}

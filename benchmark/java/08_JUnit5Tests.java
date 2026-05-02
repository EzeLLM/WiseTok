package com.example.wisetok.test;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.InjectMocks;
import org.mockito.MockitoAnnotations;
import org.mockito.ArgumentCaptor;

import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * JUnit 5 tests demonstrating:
 * @ParameterizedTest, @ValueSource, @MethodSource, @DisplayName,
 * Mockito (@Mock, @InjectMocks, when/thenReturn, verify, ArgumentCaptor),
 * and AssertJ assertions.
 */
@DisplayName("Tokenizer Training Test Suite")
public class TokenizerTrainingTests {

    @Mock
    private TokenizerRepository tokenizerRepository;

    @Mock
    private TokenizerService tokenizerService;

    @InjectMocks
    private TokenizerTrainer trainer;

    private Tokenizer testTokenizer;
    private List<String> testCorpus;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        testTokenizer = new Tokenizer();
        testTokenizer.setId(1L);
        testTokenizer.setName("test-tokenizer");
        testTokenizer.setMergeCount(1000);

        testCorpus = Arrays.asList(
            "hello world",
            "hello tokenizer",
            "world tokenizer"
        );
    }

    @Test
    @DisplayName("Should create tokenizer with valid input")
    void testCreateTokenizer() {
        // Arrange
        when(tokenizerRepository.save(any(Tokenizer.class)))
                .thenReturn(testTokenizer);

        // Act
        Tokenizer result = tokenizerRepository.save(testTokenizer);

        // Assert
        assertNotNull(result);
        assertEquals("test-tokenizer", result.getName());
        assertEquals(1000, result.getMergeCount());
        verify(tokenizerRepository, times(1)).save(any(Tokenizer.class));
    }

    @Test
    @DisplayName("Should throw exception for null tokenizer")
    void testCreateTokenizerWithNull() {
        assertThrows(IllegalArgumentException.class, () -> {
            if (testTokenizer.getName() == null) {
                throw new IllegalArgumentException("Tokenizer name cannot be null");
            }
        });
    }

    @ParameterizedTest
    @ValueSource(ints = {100, 1000, 10000, 50000})
    @DisplayName("Should accept valid merge counts")
    void testValidMergeCounts(int mergeCount) {
        testTokenizer.setMergeCount(mergeCount);
        assertThat(testTokenizer.getMergeCount())
                .isPositive()
                .isGreaterThanOrEqualTo(100)
                .isLessThanOrEqualTo(500000);
    }

    @ParameterizedTest
    @ValueSource(strings = {"a", "", "   "})
    @DisplayName("Should reject invalid tokenizer names")
    void testInvalidNames(String name) {
        testTokenizer.setName(name);
        assertThat(testTokenizer.getName())
                .as("Tokenizer name %s is invalid", name)
                .hasSizeLessThan(3)
                .or()
                .isBlank();
    }

    @ParameterizedTest
    @CsvSource({
        "gpt4,          50000,  valid",
        "custom-regex,  30000,  valid",
        "byte-level,    10000,  valid"
    })
    @DisplayName("Should train tokenizers with different patterns")
    void testTrainWithDifferentPatterns(String pattern, int merges, String expected) {
        testTokenizer.setPattern(pattern);
        testTokenizer.setMergeCount(merges);

        assertThat(testTokenizer.getPattern()).isNotBlank();
        assertThat(testTokenizer.getMergeCount()).isPositive();
        assertEquals("valid", expected);
    }

    @ParameterizedTest
    @MethodSource("provideMergeOperations")
    @DisplayName("Should apply merge operations correctly")
    void testApplyMerges(String left, String right, String expected) {
        String result = left + right;
        assertEquals(expected, result);
        assertThat(result).hasSize(expected.length());
    }

    static Stream<org.junit.jupiter.params.provider.Arguments> provideMergeOperations() {
        return Stream.of(
            org.junit.jupiter.params.provider.Arguments.of("hel", "lo", "hello"),
            org.junit.jupiter.params.provider.Arguments.of("world", "!", "world!"),
            org.junit.jupiter.params.provider.Arguments.of("to", "ken", "token")
        );
    }

    @Test
    @DisplayName("Should encode text using trained tokenizer")
    void testEncodeText() {
        // Arrange
        String input = "hello world";
        List<Integer> expected = Arrays.asList(104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100);

        when(tokenizerService.encode(any(Tokenizer.class), anyString()))
                .thenReturn(expected);

        // Act
        List<Integer> result = tokenizerService.encode(testTokenizer, input);

        // Assert
        assertThat(result).containsExactlyElementsOf(expected);
        assertThat(result).hasSize(11);
        verify(tokenizerService).encode(testTokenizer, input);
    }

    @Test
    @DisplayName("Should retrieve tokenizer from repository")
    void testGetTokenizer() {
        // Arrange
        when(tokenizerRepository.findById(1L))
                .thenReturn(Optional.of(testTokenizer));

        // Act
        Optional<Tokenizer> result = tokenizerRepository.findById(1L);

        // Assert
        assertThat(result).isPresent();
        assertThat(result.get()).isEqualTo(testTokenizer);
        verify(tokenizerRepository, times(1)).findById(1L);
    }

    @Test
    @DisplayName("Should return empty when tokenizer not found")
    void testGetTokenizerNotFound() {
        // Arrange
        when(tokenizerRepository.findById(999L))
                .thenReturn(Optional.empty());

        // Act
        Optional<Tokenizer> result = tokenizerRepository.findById(999L);

        // Assert
        assertThat(result).isEmpty();
    }

    @Test
    @DisplayName("Should delete tokenizer and verify call")
    void testDeleteTokenizer() {
        // Act
        tokenizerRepository.deleteById(1L);

        // Assert
        verify(tokenizerRepository, times(1)).deleteById(1L);
        verifyNoMoreInteractions(tokenizerRepository);
    }

    @Test
    @DisplayName("Should update tokenizer with new merge count")
    void testUpdateTokenizer() {
        // Arrange
        testTokenizer.setMergeCount(5000);
        when(tokenizerRepository.save(testTokenizer))
                .thenReturn(testTokenizer);

        // Act
        Tokenizer updated = tokenizerRepository.save(testTokenizer);

        // Assert
        assertThat(updated.getMergeCount()).isEqualTo(5000);
        verify(tokenizerRepository).save(testTokenizer);
    }

    @Test
    @DisplayName("Should train tokenizer with corpus")
    void testTrainWithCorpus() {
        // Arrange
        when(tokenizerService.trainFromCorpus(testTokenizer, testCorpus))
                .thenReturn(1000);

        // Act
        int mergesApplied = tokenizerService.trainFromCorpus(testTokenizer, testCorpus);

        // Assert
        assertThat(mergesApplied).isPositive().isEqualTo(1000);
        verify(tokenizerService).trainFromCorpus(testTokenizer, testCorpus);
    }

    @Test
    @DisplayName("Should capture and verify arguments with ArgumentCaptor")
    void testArgumentCaptor() {
        // Arrange
        ArgumentCaptor<Tokenizer> captor = ArgumentCaptor.forClass(Tokenizer.class);
        when(tokenizerRepository.save(any(Tokenizer.class)))
                .thenReturn(testTokenizer);

        // Act
        tokenizerRepository.save(testTokenizer);

        // Assert
        verify(tokenizerRepository).save(captor.capture());
        Tokenizer capturedTokenizer = captor.getValue();
        assertThat(capturedTokenizer.getName()).isEqualTo("test-tokenizer");
        assertThat(capturedTokenizer.getId()).isEqualTo(1L);
    }

    @Test
    @DisplayName("Should verify interaction order with InOrder")
    void testVerifyInteractionOrder() {
        // Act
        tokenizerRepository.findById(1L);
        tokenizerRepository.save(testTokenizer);
        tokenizerRepository.deleteById(1L);

        // Assert
        org.mockito.InOrder inOrder = inOrder(tokenizerRepository);
        inOrder.verify(tokenizerRepository).findById(1L);
        inOrder.verify(tokenizerRepository).save(testTokenizer);
        inOrder.verify(tokenizerRepository).deleteById(1L);
    }

    @Test
    @DisplayName("Complex assertion chain with AssertJ")
    void testComplexAssertions() {
        // Arrange
        List<String> tokens = Arrays.asList("hello", "world", "tokenizer");

        // Assert
        assertThat(tokens)
                .isNotEmpty()
                .hasSize(3)
                .contains("hello", "world")
                .doesNotContain("unknown")
                .allMatch(t -> t.length() > 0)
                .noneMatch(String::isEmpty);

        assertThat("hello")
                .isNotNull()
                .isNotBlank()
                .hasSize(5)
                .startsWith("hel")
                .endsWith("lo")
                .containsIgnoringCase("HELLO");
    }

    @Test
    @DisplayName("Soft assertions to collect multiple failures")
    void testSoftAssertions() {
        // Arrange
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.setId(1L);
        tokenizer.setName("test");
        tokenizer.setMergeCount(50000);

        // Assert (soft assertions don't fail immediately)
        org.assertj.core.api.SoftAssertions.assertSoftly(soft -> {
            soft.assertThat(tokenizer.getId()).isEqualTo(1L);
            soft.assertThat(tokenizer.getName()).isEqualTo("test");
            soft.assertThat(tokenizer.getMergeCount()).isGreaterThan(10000);
            soft.assertThat(tokenizer.getMergeCount()).isLessThan(100000);
        });
    }
}

// Minimal mock classes for testing
class Tokenizer {
    private Long id;
    private String name;
    private String pattern;
    private Integer mergeCount;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getPattern() { return pattern; }
    public void setPattern(String pattern) { this.pattern = pattern; }

    public Integer getMergeCount() { return mergeCount; }
    public void setMergeCount(Integer mergeCount) { this.mergeCount = mergeCount; }
}

interface TokenizerRepository {
    Tokenizer save(Tokenizer tokenizer);
    Optional<Tokenizer> findById(Long id);
    void deleteById(Long id);
}

interface TokenizerService {
    List<Integer> encode(Tokenizer tokenizer, String text);
    int trainFromCorpus(Tokenizer tokenizer, List<String> corpus);
}

class TokenizerTrainer {
    private TokenizerRepository repository;
    private TokenizerService service;
}

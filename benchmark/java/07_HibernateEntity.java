package com.example.wisetok.model;

import javax.persistence.*;
import javax.validation.constraints.*;
import java.time.LocalDateTime;
import java.util.*;

/**
 * JPA/Hibernate entities demonstrating:
 * @Entity, @Table, @OneToMany with mappedBy, cascade types, fetch strategies,
 * and custom @Query repository methods.
 */
@Entity
@Table(name = "tokenizers", indexes = {
    @Index(name = "idx_name", columnList = "name"),
    @Index(name = "idx_created", columnList = "created_at")
})
public class Tokenizer {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 255)
    @NotBlank(message = "Tokenizer name is required")
    @Size(min = 3, max = 255)
    private String name;

    @Column(nullable = false, columnDefinition = "TEXT")
    @NotBlank
    private String pattern;

    @Column(name = "merge_count", nullable = false)
    @Min(value = 100, message = "Minimum 100 merges required")
    @Max(value = 500000)
    private Integer mergeCount;

    @Column(name = "vocabulary_size", nullable = false)
    @PositiveOrZero
    private Integer vocabularySize;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Version
    @Column(name = "version")
    private Long version;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private TokenizerStatus status = TokenizerStatus.DRAFT;

    // One-to-Many: Tokenizer -> TrainingRun
    @OneToMany(mappedBy = "tokenizer", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    private Set<TrainingRun> trainingRuns = new HashSet<>();

    // One-to-Many: Tokenizer -> MergeOperation
    @OneToMany(mappedBy = "tokenizer", cascade = {CascadeType.PERSIST, CascadeType.MERGE}, fetch = FetchType.LAZY)
    private List<MergeOperation> mergeOperations = new ArrayList<>();

    // Many-to-One: Tokenizer -> TokenizerConfig
    @ManyToOne(fetch = FetchType.EAGER, optional = false)
    @JoinColumn(name = "config_id", nullable = false)
    private TokenizerConfig config;

    public Tokenizer() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getPattern() { return pattern; }
    public void setPattern(String pattern) { this.pattern = pattern; }

    public Integer getMergeCount() { return mergeCount; }
    public void setMergeCount(Integer mergeCount) { this.mergeCount = mergeCount; }

    public Integer getVocabularySize() { return vocabularySize; }
    public void setVocabularySize(Integer vocabularySize) { this.vocabularySize = vocabularySize; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public TokenizerStatus getStatus() { return status; }
    public void setStatus(TokenizerStatus status) { this.status = status; }

    public Set<TrainingRun> getTrainingRuns() { return trainingRuns; }
    public void setTrainingRuns(Set<TrainingRun> trainingRuns) { this.trainingRuns = trainingRuns; }

    public List<MergeOperation> getMergeOperations() { return mergeOperations; }
    public void setMergeOperations(List<MergeOperation> mergeOperations) { this.mergeOperations = mergeOperations; }

    public TokenizerConfig getConfig() { return config; }
    public void setConfig(TokenizerConfig config) { this.config = config; }
}

@Entity
@Table(name = "training_runs")
public class TrainingRun {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    @NotNull
    private Integer epoch;

    @Column(nullable = false)
    @Positive
    private Long corpusSize;

    @Column(name = "tokens_processed")
    @PositiveOrZero
    private Long tokensProcessed;

    @Column(name = "started_at", nullable = false)
    private LocalDateTime startedAt;

    @Column(name = "completed_at")
    private LocalDateTime completedAt;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private TrainingStatus trainingStatus = TrainingStatus.IN_PROGRESS;

    // Many-to-One: TrainingRun -> Tokenizer
    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "tokenizer_id", nullable = false)
    private Tokenizer tokenizer;

    // One-to-Many: TrainingRun -> TrainingMetric
    @OneToMany(mappedBy = "trainingRun", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<TrainingMetric> metrics = new ArrayList<>();

    public TrainingRun() {
        this.startedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Integer getEpoch() { return epoch; }
    public void setEpoch(Integer epoch) { this.epoch = epoch; }

    public Long getCorpusSize() { return corpusSize; }
    public void setCorpusSize(Long corpusSize) { this.corpusSize = corpusSize; }

    public Tokenizer getTokenizer() { return tokenizer; }
    public void setTokenizer(Tokenizer tokenizer) { this.tokenizer = tokenizer; }

    public List<TrainingMetric> getMetrics() { return metrics; }
    public void setMetrics(List<TrainingMetric> metrics) { this.metrics = metrics; }
}

@Entity
@Table(name = "merge_operations")
public class MergeOperation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    @NotNull
    private Integer mergeIndex;

    @Column(nullable = false, length = 128)
    @NotBlank
    private String leftToken;

    @Column(nullable = false, length = 128)
    @NotBlank
    private String rightToken;

    @Column(nullable = false)
    @Positive
    private Long pairFrequency;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "tokenizer_id", nullable = false)
    private Tokenizer tokenizer;

    public MergeOperation() {
        this.createdAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public Integer getMergeIndex() { return mergeIndex; }
    public void setMergeIndex(Integer mergeIndex) { this.mergeIndex = mergeIndex; }

    public String getLeftToken() { return leftToken; }
    public void setLeftToken(String leftToken) { this.leftToken = leftToken; }

    public String getRightToken() { return rightToken; }
    public void setRightToken(String rightToken) { this.rightToken = rightToken; }

    public Long getPairFrequency() { return pairFrequency; }
    public void setPairFrequency(Long pairFrequency) { this.pairFrequency = pairFrequency; }

    public Tokenizer getTokenizer() { return tokenizer; }
    public void setTokenizer(Tokenizer tokenizer) { this.tokenizer = tokenizer; }
}

@Entity
@Table(name = "training_metrics")
public class TrainingMetric {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 50)
    @NotBlank
    private String metricName;

    @Column(nullable = false)
    @NotNull
    private Double metricValue;

    @Column(name = "recorded_at", nullable = false)
    private LocalDateTime recordedAt;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "training_run_id", nullable = false)
    private TrainingRun trainingRun;

    public TrainingMetric() {
        this.recordedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public String getMetricName() { return metricName; }
    public void setMetricName(String metricName) { this.metricName = metricName; }

    public Double getMetricValue() { return metricValue; }
    public void setMetricValue(Double metricValue) { this.metricValue = metricValue; }

    public TrainingRun getTrainingRun() { return trainingRun; }
    public void setTrainingRun(TrainingRun trainingRun) { this.trainingRun = trainingRun; }
}

@Entity
@Table(name = "tokenizer_configs")
public class TokenizerConfig {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 100)
    @NotBlank
    private String configName;

    @Column(columnDefinition = "TEXT")
    private String description;

    @OneToMany(mappedBy = "config", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<Tokenizer> tokenizers = new ArrayList<>();

    public Long getId() { return id; }
    public String getConfigName() { return configName; }
    public void setConfigName(String configName) { this.configName = configName; }

    public List<Tokenizer> getTokenizers() { return tokenizers; }
    public void setTokenizers(List<Tokenizer> tokenizers) { this.tokenizers = tokenizers; }
}

enum TokenizerStatus {
    DRAFT, TRAINING, COMPLETE, FAILED, ARCHIVED
}

enum TrainingStatus {
    IN_PROGRESS, COMPLETE, FAILED, CANCELLED
}

// Custom repository with @Query methods
interface TokenizerRepository extends org.springframework.data.jpa.repository.JpaRepository<Tokenizer, Long> {

    @org.springframework.data.jpa.repository.Query(
        "SELECT t FROM Tokenizer t WHERE t.status = :status ORDER BY t.createdAt DESC"
    )
    List<Tokenizer> findByStatus(TokenizerStatus status);

    @org.springframework.data.jpa.repository.Query(
        "SELECT t FROM Tokenizer t WHERE LOWER(t.name) LIKE LOWER(CONCAT('%', :query, '%'))"
    )
    List<Tokenizer> searchByName(String query);

    @org.springframework.data.jpa.repository.Query(
        "SELECT t FROM Tokenizer t JOIN FETCH t.trainingRuns WHERE t.id = :id"
    )
    Optional<Tokenizer> findByIdWithRuns(Long id);

    @org.springframework.data.jpa.repository.Query(value =
        "SELECT t.*, COUNT(m) as merge_count FROM tokenizers t " +
        "LEFT JOIN merge_operations m ON t.id = m.tokenizer_id " +
        "GROUP BY t.id HAVING COUNT(m) > :minMerges",
        nativeQuery = true
    )
    List<Tokenizer> findTokenizersWithMinMerges(Integer minMerges);
}

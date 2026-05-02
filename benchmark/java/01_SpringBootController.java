package com.example.wisetok.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;
import javax.validation.Valid;
import java.util.*;

/**
 * REST controller for tokenizer training and inference endpoints.
 * Demonstrates Spring Boot annotations, dependency injection, and exception handling.
 */
@RestController
@RequestMapping("/api/v1/tokenizer")
@CrossOrigin(origins = "*", maxAge = 3600)
public class TokenizerController {

    @Autowired
    private TokenizerService tokenizerService;

    @Autowired
    private TokenizerRepository tokenizerRepository;

    /**
     * GET /api/v1/tokenizer/{id}
     * Retrieve a tokenizer by ID.
     */
    @GetMapping("/{id}")
    public ResponseEntity<TokenizerDTO> getTokenizer(@PathVariable Long id) {
        Optional<Tokenizer> tokenizer = tokenizerRepository.findById(id);
        if (tokenizer.isPresent()) {
            return ResponseEntity.ok(TokenizerDTO.from(tokenizer.get()));
        }
        return ResponseEntity.notFound().build();
    }

    /**
     * POST /api/v1/tokenizer
     * Create a new tokenizer with validation.
     */
    @PostMapping
    public ResponseEntity<?> createTokenizer(
            @Valid @RequestBody TokenizerRequest request,
            BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return ResponseEntity
                    .badRequest()
                    .body(new ErrorResponse("Validation failed", bindingResult.getFieldErrors()));
        }
        try {
            Tokenizer tokenizer = tokenizerService.createTokenizer(request);
            return ResponseEntity.status(HttpStatus.CREATED).body(TokenizerDTO.from(tokenizer));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(new ErrorResponse(e.getMessage()));
        }
    }

    /**
     * PUT /api/v1/tokenizer/{id}
     * Update an existing tokenizer.
     */
    @PutMapping("/{id}")
    public ResponseEntity<?> updateTokenizer(
            @PathVariable Long id,
            @Valid @RequestBody TokenizerRequest request) {
        try {
            Tokenizer updated = tokenizerService.updateTokenizer(id, request);
            return ResponseEntity.ok(TokenizerDTO.from(updated));
        } catch (TokenizerNotFoundException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse("Update failed: " + e.getMessage()));
        }
    }

    /**
     * DELETE /api/v1/tokenizer/{id}
     * Delete a tokenizer by ID.
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTokenizer(@PathVariable Long id) {
        if (tokenizerRepository.existsById(id)) {
            tokenizerRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }

    /**
     * GET /api/v1/tokenizer/search
     * Search tokenizers by name (optional filtering).
     */
    @GetMapping("/search")
    public ResponseEntity<List<TokenizerDTO>> searchTokenizers(
            @RequestParam(value = "name", required = false) String name) {
        List<Tokenizer> results;
        if (name != null && !name.isEmpty()) {
            results = tokenizerRepository.findByNameContainingIgnoreCase(name);
        } else {
            results = tokenizerRepository.findAll();
        }
        List<TokenizerDTO> dtos = new ArrayList<>();
        for (Tokenizer t : results) {
            dtos.add(TokenizerDTO.from(t));
        }
        return ResponseEntity.ok(dtos);
    }

    /**
     * POST /api/v1/tokenizer/{id}/encode
     * Encode text using a specific tokenizer.
     */
    @PostMapping("/{id}/encode")
    public ResponseEntity<?> encodeText(
            @PathVariable Long id,
            @RequestBody TextEncodeRequest request) {
        try {
            Tokenizer tokenizer = tokenizerRepository.findById(id)
                    .orElseThrow(TokenizerNotFoundException::new);
            List<Integer> tokens = tokenizerService.encode(tokenizer, request.getText());
            return ResponseEntity.ok(new TextEncodeResponse(tokens));
        } catch (TokenizerNotFoundException e) {
            return ResponseEntity.notFound().build();
        }
    }

    @ExceptionHandler(TokenizerNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(TokenizerNotFoundException ex) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ErrorResponse("Tokenizer not found"));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGlobalException(Exception ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse("Internal server error: " + ex.getMessage()));
    }
}

interface TokenizerRepository extends JpaRepository<Tokenizer, Long> {
    List<Tokenizer> findByNameContainingIgnoreCase(String name);
}

class Tokenizer {
    private Long id;
    private String name;
    private String pattern;
    private Integer mergeCount;

    public Tokenizer() {}
    public Tokenizer(String name, String pattern) {
        this.name = name;
        this.pattern = pattern;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getPattern() { return pattern; }
    public void setPattern(String pattern) { this.pattern = pattern; }
    public Integer getMergeCount() { return mergeCount; }
    public void setMergeCount(Integer mergeCount) { this.mergeCount = mergeCount; }
}

class TokenizerDTO {
    public Long id;
    public String name;
    public Integer mergeCount;

    static TokenizerDTO from(Tokenizer t) {
        TokenizerDTO dto = new TokenizerDTO();
        dto.id = t.getId();
        dto.name = t.getName();
        dto.mergeCount = t.getMergeCount();
        return dto;
    }
}

class TokenizerRequest {
    public String name;
    public String pattern;
    public Integer merges = 50000;
}

class TextEncodeRequest {
    public String text;

    public String getText() { return text; }
}

class TextEncodeResponse {
    public List<Integer> tokens;

    public TextEncodeResponse(List<Integer> tokens) {
        this.tokens = tokens;
    }
}

class ErrorResponse {
    public String message;
    public Object details;

    public ErrorResponse(String message) {
        this.message = message;
    }

    public ErrorResponse(String message, Object details) {
        this.message = message;
        this.details = details;
    }
}

class TokenizerService {
    @Autowired
    private TokenizerRepository tokenizerRepository;

    public Tokenizer createTokenizer(TokenizerRequest request) {
        Tokenizer t = new Tokenizer(request.name, request.pattern);
        t.setMergeCount(request.merges);
        return tokenizerRepository.save(t);
    }

    public Tokenizer updateTokenizer(Long id, TokenizerRequest request) {
        Tokenizer t = tokenizerRepository.findById(id)
                .orElseThrow(TokenizerNotFoundException::new);
        t.setName(request.name);
        t.setPattern(request.pattern);
        t.setMergeCount(request.merges);
        return tokenizerRepository.save(t);
    }

    public List<Integer> encode(Tokenizer tokenizer, String text) {
        List<Integer> result = new ArrayList<>();
        for (char c : text.toCharArray()) {
            result.add((int) c);
        }
        return result;
    }
}

class TokenizerNotFoundException extends RuntimeException {
    public TokenizerNotFoundException() {
        super("Tokenizer not found");
    }
}

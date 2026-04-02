package ai.pipestream.module.semanticmanager.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Configuration options for the semantic manager, parsed from the ProcessDataRequest's jsonConfig.
 * Specifies which index to use for VectorSet resolution and optional overrides.
 *
 * Supports two usage modes:
 *
 * 1. **Explicit directives** — full control over chunker × embedder cartesian products.
 * 2. **Convenience fields** — set source_field, chunk_size, embedding_model, etc. for simple
 *    single-source configs. These are converted to a single implicit directive internally.
 *    Convenience fields are ignored when explicit directives are present.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record SemanticManagerOptions(
        @JsonProperty("index_name") String indexName,
        @JsonProperty("vector_set_ids") List<String> vectorSetIds,
        @JsonProperty("max_concurrent_chunkers") Integer maxConcurrentChunkers,
        @JsonProperty("max_concurrent_embedders") Integer maxConcurrentEmbedders,
        @JsonProperty("directives") List<DirectiveConfig> directives,

        // Convenience fields for simple single-source configs
        @JsonProperty("source_field") String sourceField,
        @JsonProperty("chunk_size") Integer chunkSize,
        @JsonProperty("chunk_overlap") Integer chunkOverlap,
        @JsonProperty("chunk_algorithm") String chunkAlgorithm,
        @JsonProperty("embedding_model") String embeddingModel,
        @JsonProperty("skip_chunking") Boolean skipChunking,
        @JsonProperty("result_set_name_template") String resultSetNameTemplate,
        @JsonProperty("semantic_config_id") String semanticConfigId
) {
    public static final String DEFAULT_INDEX_NAME = "default-index";
    public static final int DEFAULT_MAX_CONCURRENT_CHUNKERS = 4;
    public static final int DEFAULT_MAX_CONCURRENT_EMBEDDERS = 8;
    public static final String DEFAULT_SOURCE_FIELD = "body";
    public static final int DEFAULT_CHUNK_SIZE = 500;
    public static final int DEFAULT_CHUNK_OVERLAP = 50;
    public static final String DEFAULT_CHUNK_ALGORITHM = "TOKEN";
    public static final String DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2";

    /** Original 5-arg constructor for backward compatibility. */
    public SemanticManagerOptions(String indexName, List<String> vectorSetIds,
                                  Integer maxConcurrentChunkers, Integer maxConcurrentEmbedders,
                                  List<DirectiveConfig> directives) {
        this(indexName, vectorSetIds, maxConcurrentChunkers, maxConcurrentEmbedders, directives,
                null, null, null, null, null, null, null, null);
    }

    /** Default no-arg constructor. */
    public SemanticManagerOptions() {
        this(DEFAULT_INDEX_NAME, null, DEFAULT_MAX_CONCURRENT_CHUNKERS, DEFAULT_MAX_CONCURRENT_EMBEDDERS,
                null, null, null, null, null, null, null, null, null);
    }

    public boolean hasDirectives() {
        return directives != null && !directives.isEmpty();
    }

    /**
     * Returns true if any convenience field is explicitly set, meaning the user wants
     * an implicit directive built from these fields.
     */
    public boolean hasConvenienceFields() {
        return sourceField != null || chunkSize != null || chunkOverlap != null
                || chunkAlgorithm != null || embeddingModel != null
                || skipChunking != null || resultSetNameTemplate != null;
    }

    /**
     * Converts convenience fields into a single DirectiveConfig.
     * If skipChunking is true, the directive has no chunker configs (field-level embedding).
     * Otherwise, a single chunker config is created from chunk_size, chunk_overlap, and chunk_algorithm.
     */
    public DirectiveConfig toImplicitDirective() {
        String effectiveSourceField = sourceField != null ? sourceField : DEFAULT_SOURCE_FIELD;
        String effectiveModel = embeddingModel != null ? embeddingModel : DEFAULT_EMBEDDING_MODEL;
        boolean effectiveSkipChunking = skipChunking != null && skipChunking;

        // Embedder config — just the model ID, no extra config needed
        DirectiveConfig.NamedConfig embedderConfig = new DirectiveConfig.NamedConfig(effectiveModel, null);

        List<DirectiveConfig.NamedConfig> chunkerConfigs;
        if (effectiveSkipChunking) {
            chunkerConfigs = null; // no chunking
        } else {
            int effectiveChunkSize = chunkSize != null ? chunkSize : DEFAULT_CHUNK_SIZE;
            int effectiveChunkOverlap = chunkOverlap != null ? chunkOverlap : DEFAULT_CHUNK_OVERLAP;
            String effectiveAlgorithm = chunkAlgorithm != null ? chunkAlgorithm : DEFAULT_CHUNK_ALGORITHM;

            Map<String, Object> chunkerParams = Map.of(
                    "chunk_size", effectiveChunkSize,
                    "chunk_overlap", effectiveChunkOverlap,
                    "algorithm", effectiveAlgorithm
            );
            chunkerConfigs = List.of(new DirectiveConfig.NamedConfig("default", chunkerParams));
        }

        String celSelector = "document.search_metadata." + effectiveSourceField;

        return new DirectiveConfig(
                effectiveSourceField,
                celSelector,
                chunkerConfigs,
                List.of(embedderConfig),
                resultSetNameTemplate
        );
    }

    public String effectiveIndexName() {
        return indexName != null && !indexName.isEmpty() ? indexName : DEFAULT_INDEX_NAME;
    }

    public int effectiveMaxConcurrentChunkers() {
        return maxConcurrentChunkers != null ? maxConcurrentChunkers : DEFAULT_MAX_CONCURRENT_CHUNKERS;
    }

    public int effectiveMaxConcurrentEmbedders() {
        return maxConcurrentEmbedders != null ? maxConcurrentEmbedders : DEFAULT_MAX_CONCURRENT_EMBEDDERS;
    }

    public boolean hasSemanticConfigId() {
        return semanticConfigId != null && !semanticConfigId.isBlank();
    }

    public static String getJsonV7Schema() {
        return """
                {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "type": "object",
                  "title": "Semantic Manager Options",
                  "properties": {
                    "index_name": {
                      "type": "string",
                      "description": "OpenSearch index name for VectorSet resolution",
                      "default": "default-index"
                    },
                    "vector_set_ids": {
                      "type": "array",
                      "items": { "type": "string" },
                      "description": "Optional list of specific VectorSet IDs to process (all if empty)"
                    },
                    "max_concurrent_chunkers": {
                      "type": "integer",
                      "description": "Maximum concurrent chunker streams",
                      "default": 4,
                      "minimum": 1
                    },
                    "max_concurrent_embedders": {
                      "type": "integer",
                      "description": "Maximum concurrent embedder streams",
                      "default": 8,
                      "minimum": 1
                    },
                    "directives": {
                      "type": "array",
                      "description": "Explicit directives for chunker x embedder cartesian products. Takes priority over convenience fields.",
                      "items": { "type": "object" }
                    },
                    "source_field": {
                      "type": "string",
                      "description": "Convenience: source field to embed (e.g. 'body', 'title'). Used when no explicit directives are set.",
                      "default": "body"
                    },
                    "chunk_size": {
                      "type": "integer",
                      "description": "Convenience: chunk size in tokens. Ignored when skip_chunking=true.",
                      "default": 500,
                      "minimum": 1
                    },
                    "chunk_overlap": {
                      "type": "integer",
                      "description": "Convenience: chunk overlap in tokens. Ignored when skip_chunking=true.",
                      "default": 50,
                      "minimum": 0
                    },
                    "chunk_algorithm": {
                      "type": "string",
                      "description": "Convenience: chunking algorithm. Ignored when skip_chunking=true.",
                      "default": "TOKEN",
                      "enum": ["TOKEN", "SENTENCE", "PARAGRAPH"]
                    },
                    "embedding_model": {
                      "type": "string",
                      "description": "Convenience: embedding model ID (e.g. 'all-MiniLM-L6-v2').",
                      "default": "all-MiniLM-L6-v2"
                    },
                    "skip_chunking": {
                      "type": "boolean",
                      "description": "Convenience: if true, embed the raw field without chunking (field-level embedding).",
                      "default": false
                    },
                    "result_set_name_template": {
                      "type": "string",
                      "description": "Convenience: template for result set naming. Placeholders: {source_label}, {chunker_id}, {embedder_id}."
                    },
                    "semantic_config_id": {
                      "type": "string",
                      "description": "Optional: semantic config ID to stamp on all SemanticProcessingResults produced by this node. Enables downstream grouping by config."
                    }
                  }
                }
                """;
    }
}

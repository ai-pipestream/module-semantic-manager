package ai.pipestream.module.semanticmanager.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Fast unit tests for SemanticManagerOptions JSON parsing, defaults,
 * convenience fields, implicit directive generation, and priority resolution.
 * No Quarkus context needed.
 */
class SemanticManagerOptionsParsingTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    // =========================================================================
    // JSON parsing: basic
    // =========================================================================

    @Test
    void parseMinimalConfig_usesDefaults() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("{}", SemanticManagerOptions.class);

        assertThat(opts.indexName()).as("indexName from empty config").isNull();
        assertThat(opts.directives()).as("directives from empty config").isNull();
        assertThat(opts.hasDirectives()).as("hasDirectives").isFalse();
        assertThat(opts.hasConvenienceFields()).as("hasConvenienceFields").isFalse();
        assertThat(opts.effectiveIndexName()).as("effectiveIndexName falls back to default")
                .isEqualTo(SemanticManagerOptions.DEFAULT_INDEX_NAME);
    }

    @Test
    void parseWithIndexName() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                { "index_name": "my-corpus" }
                """, SemanticManagerOptions.class);

        assertThat(opts.effectiveIndexName()).as("explicit index name").isEqualTo("my-corpus");
        assertThat(opts.hasDirectives()).as("no directives").isFalse();
        assertThat(opts.hasConvenienceFields()).as("no convenience fields").isFalse();
    }

    @Test
    void parseUnknownFields_ignored() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                { "index_name": "x", "future_feature": "ignored" }
                """, SemanticManagerOptions.class);
        assertThat(opts.indexName()).as("known field survives unknown fields").isEqualTo("x");
    }

    // =========================================================================
    // Explicit directives parsing
    // =========================================================================

    @Test
    void parseExplicitDirectives_singleDirective() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                {
                  "directives": [{
                    "source_label": "body",
                    "cel_selector": "document.search_metadata.body",
                    "chunker_configs": [{ "config_id": "token_500", "config": { "chunk_size": 500 } }],
                    "embedder_configs": [{ "config_id": "all-MiniLM-L6-v2" }]
                  }]
                }
                """, SemanticManagerOptions.class);

        assertThat(opts.hasDirectives()).as("has directives").isTrue();
        assertThat(opts.directives()).as("directive count").hasSize(1);

        DirectiveConfig dc = opts.directives().get(0);
        assertThat(dc.sourceLabel()).as("source_label").isEqualTo("body");
        assertThat(dc.celSelector()).as("cel_selector").isEqualTo("document.search_metadata.body");
        assertThat(dc.chunkerConfigs()).as("chunker_configs").hasSize(1);
        assertThat(dc.chunkerConfigs().get(0).configId()).as("chunker config_id").isEqualTo("token_500");
        assertThat(dc.embedderConfigs()).as("embedder_configs").hasSize(1);
        assertThat(dc.embedderConfigs().get(0).configId()).as("embedder config_id").isEqualTo("all-MiniLM-L6-v2");
    }

    @Test
    void parseExplicitDirectives_cartesianProduct() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                {
                  "directives": [{
                    "source_label": "body",
                    "cel_selector": "document.search_metadata.body",
                    "chunker_configs": [{ "config_id": "c1" }],
                    "embedder_configs": [{ "config_id": "e1" }, { "config_id": "e2" }]
                  }, {
                    "source_label": "title",
                    "cel_selector": "document.search_metadata.title",
                    "embedder_configs": [{ "config_id": "e1" }]
                  }]
                }
                """, SemanticManagerOptions.class);

        assertThat(opts.directives()).as("two directives").hasSize(2);
        assertThat(opts.directives().get(0).embedderConfigs()).as("first directive: 2 embedders").hasSize(2);
        assertThat(opts.directives().get(1).chunkerConfigs())
                .as("second directive: no chunker_configs → field-level").isNull();
    }

    @Test
    void parseDirective_emptyChunkerConfigs_fieldLevel() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                {
                  "directives": [{
                    "source_label": "body",
                    "cel_selector": "document.search_metadata.body",
                    "chunker_configs": [],
                    "embedder_configs": [{ "config_id": "all-MiniLM-L6-v2" }]
                  }]
                }
                """, SemanticManagerOptions.class);

        assertThat(opts.directives().get(0).chunkerConfigs())
                .as("empty chunker_configs list → field-level path").isEmpty();
    }

    // =========================================================================
    // Convenience fields parsing
    // =========================================================================

    @Test
    void parseConvenienceFields_skipChunking() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                {
                  "source_field": "body",
                  "embedding_model": "all-MiniLM-L6-v2",
                  "skip_chunking": true
                }
                """, SemanticManagerOptions.class);

        assertThat(opts.hasDirectives()).as("no explicit directives").isFalse();
        assertThat(opts.hasConvenienceFields()).as("convenience fields detected").isTrue();
        assertThat(opts.sourceField()).as("source_field").isEqualTo("body");
        assertThat(opts.embeddingModel()).as("embedding_model").isEqualTo("all-MiniLM-L6-v2");
        assertThat(opts.skipChunking()).as("skip_chunking").isTrue();
    }

    @Test
    void parseConvenienceFields_withChunking() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                {
                  "source_field": "body",
                  "chunk_size": 300,
                  "chunk_overlap": 30,
                  "chunk_algorithm": "SENTENCE",
                  "embedding_model": "all-MiniLM-L6-v2"
                }
                """, SemanticManagerOptions.class);

        assertThat(opts.hasConvenienceFields()).as("convenience fields present").isTrue();
        assertThat(opts.chunkSize()).as("chunk_size").isEqualTo(300);
        assertThat(opts.chunkOverlap()).as("chunk_overlap").isEqualTo(30);
        assertThat(opts.chunkAlgorithm()).as("chunk_algorithm").isEqualTo("SENTENCE");
        assertThat(opts.skipChunking()).as("skip_chunking not set → null").isNull();
    }

    @Test
    void parseConvenienceFields_onlySourceField_stillTriggersConvenienceMode() throws Exception {
        SemanticManagerOptions opts = mapper.readValue("""
                { "source_field": "title" }
                """, SemanticManagerOptions.class);
        assertThat(opts.hasConvenienceFields())
                .as("setting just source_field should trigger convenience mode").isTrue();
    }

    // =========================================================================
    // Implicit directive generation from convenience fields
    // =========================================================================

    @Test
    void toImplicitDirective_skipChunking_noChunkerConfigs() {
        SemanticManagerOptions opts = new SemanticManagerOptions(
                null, null, null, null, null,
                "body", null, null, null, "all-MiniLM-L6-v2", true, null, null);

        DirectiveConfig dc = opts.toImplicitDirective();

        assertThat(dc.sourceLabel()).as("source_label").isEqualTo("body");
        assertThat(dc.celSelector()).as("cel_selector auto-derived")
                .isEqualTo("document.search_metadata.body");
        assertThat(dc.chunkerConfigs()).as("skip_chunking=true → no chunker configs").isNull();
        assertThat(dc.embedderConfigs()).as("single embedder config").hasSize(1);
        assertThat(dc.embedderConfigs().get(0).configId()).as("embedder model")
                .isEqualTo("all-MiniLM-L6-v2");
    }

    @Test
    void toImplicitDirective_withChunking_hasChunkerConfig() {
        SemanticManagerOptions opts = new SemanticManagerOptions(
                null, null, null, null, null,
                "body", 300, 30, "SENTENCE", "all-MiniLM-L6-v2", false, null, null);

        DirectiveConfig dc = opts.toImplicitDirective();

        assertThat(dc.chunkerConfigs()).as("chunker configs present").hasSize(1);
        assertThat(dc.chunkerConfigs().get(0).configId()).as("default chunker config_id").isEqualTo("default");

        Map<String, Object> config = dc.chunkerConfigs().get(0).config();
        assertThat(config).as("chunker params")
                .containsEntry("chunk_size", 300)
                .containsEntry("chunk_overlap", 30)
                .containsEntry("algorithm", "SENTENCE");
    }

    @Test
    void toImplicitDirective_allNull_usesConventionDefaults() {
        SemanticManagerOptions opts = new SemanticManagerOptions(
                null, null, null, null, null,
                null, null, null, null, null, null, null, null);

        DirectiveConfig dc = opts.toImplicitDirective();

        assertThat(dc.sourceLabel()).as("default source_label").isEqualTo("body");
        assertThat(dc.embedderConfigs().get(0).configId()).as("default embedding model")
                .isEqualTo("all-MiniLM-L6-v2");
        assertThat(dc.chunkerConfigs()).as("skip_chunking=null → false → has chunker").hasSize(1);

        Map<String, Object> config = dc.chunkerConfigs().get(0).config();
        assertThat(config).as("default chunker params")
                .containsEntry("chunk_size", 500)
                .containsEntry("chunk_overlap", 50)
                .containsEntry("algorithm", "TOKEN");
    }

    @Test
    void toImplicitDirective_customTemplate_passedThrough() {
        SemanticManagerOptions opts = new SemanticManagerOptions(
                null, null, null, null, null,
                "body", null, null, null, "model1", true,
                "{source_label}_custom_{embedder_id}", null);

        DirectiveConfig dc = opts.toImplicitDirective();
        assertThat(dc.fieldNameTemplate()).as("custom template preserved")
                .isEqualTo("{source_label}_custom_{embedder_id}");
    }

    // =========================================================================
    // Priority resolution
    // =========================================================================

    @Test
    void priority_directivesPresentAndConvenienceFields_bothDetected() {
        var directive = new DirectiveConfig("body", "document.search_metadata.body",
                List.of(new DirectiveConfig.NamedConfig("c1", null)),
                List.of(new DirectiveConfig.NamedConfig("e1", null)),
                null);

        SemanticManagerOptions opts = new SemanticManagerOptions(
                "idx", null, 4, 8, List.of(directive),
                "title", 300, 30, "SENTENCE", "other-model", true, null, null);

        assertThat(opts.hasDirectives()).as("explicit directives present").isTrue();
        assertThat(opts.hasConvenienceFields()).as("convenience fields also present").isTrue();
        // Orchestrator checks hasDirectives() first → directives win
    }

    @Test
    void priority_noDirectivesNoConvenience_fallsToVectorSetService() {
        SemanticManagerOptions opts = new SemanticManagerOptions("my-index", null, 4, 8, null);

        assertThat(opts.hasDirectives()).as("no directives").isFalse();
        assertThat(opts.hasConvenienceFields()).as("no convenience fields").isFalse();
        assertThat(opts.effectiveIndexName()).as("index name for VectorSetService fallback")
                .isEqualTo("my-index");
    }

    // =========================================================================
    // Backward compatibility
    // =========================================================================

    @Test
    void fiveArgConstructor_noConvenienceFields() {
        SemanticManagerOptions opts = new SemanticManagerOptions("idx", null, 4, 8, null);

        assertThat(opts.hasDirectives()).as("5-arg: no directives").isFalse();
        assertThat(opts.hasConvenienceFields()).as("5-arg: no convenience fields").isFalse();
        assertThat(opts.sourceField()).as("5-arg: sourceField is null").isNull();
        assertThat(opts.skipChunking()).as("5-arg: skipChunking is null").isNull();
    }

    @Test
    void noArgConstructor_allDefaults() {
        SemanticManagerOptions opts = new SemanticManagerOptions();

        assertThat(opts.effectiveIndexName()).as("default index name")
                .isEqualTo(SemanticManagerOptions.DEFAULT_INDEX_NAME);
        assertThat(opts.effectiveMaxConcurrentChunkers()).as("default max chunkers")
                .isEqualTo(SemanticManagerOptions.DEFAULT_MAX_CONCURRENT_CHUNKERS);
        assertThat(opts.effectiveMaxConcurrentEmbedders()).as("default max embedders")
                .isEqualTo(SemanticManagerOptions.DEFAULT_MAX_CONCURRENT_EMBEDDERS);
    }

    // =========================================================================
    // JSON Schema
    // =========================================================================

    @Test
    void jsonSchema_isValidJson_andDocumentsAllFields() throws Exception {
        var tree = mapper.readTree(SemanticManagerOptions.getJsonV7Schema());

        assertThat(tree.get("type").asText()).as("schema type").isEqualTo("object");

        var props = tree.get("properties");
        // Core fields
        assertThat(props.has("index_name")).as("schema has index_name").isTrue();
        assertThat(props.has("directives")).as("schema has directives").isTrue();
        // Convenience fields
        assertThat(props.has("source_field")).as("schema has source_field").isTrue();
        assertThat(props.has("chunk_size")).as("schema has chunk_size").isTrue();
        assertThat(props.has("chunk_overlap")).as("schema has chunk_overlap").isTrue();
        assertThat(props.has("chunk_algorithm")).as("schema has chunk_algorithm").isTrue();
        assertThat(props.has("embedding_model")).as("schema has embedding_model").isTrue();
        assertThat(props.has("skip_chunking")).as("schema has skip_chunking").isTrue();
        assertThat(props.has("result_set_name_template")).as("schema has result_set_name_template").isTrue();
    }
}

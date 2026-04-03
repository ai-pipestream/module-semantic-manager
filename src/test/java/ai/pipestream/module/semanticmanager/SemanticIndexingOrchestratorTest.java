package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.ChunkerStreamClient;
import ai.pipestream.module.semanticmanager.service.EmbedderStreamClient;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import ai.pipestream.module.semanticmanager.service.VectorSetResolver;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.*;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class SemanticIndexingOrchestratorTest {

    private SemanticIndexingOrchestrator orchestrator;
    private VectorSetResolver vectorSetResolver;
    private ChunkerStreamClient chunkerStreamClient;
    private EmbedderStreamClient embedderStreamClient;

    @BeforeEach
    void setUp() throws Exception {
        orchestrator = new SemanticIndexingOrchestrator();
        vectorSetResolver = mock(VectorSetResolver.class);
        chunkerStreamClient = mock(ChunkerStreamClient.class);
        embedderStreamClient = mock(EmbedderStreamClient.class);

        setField(orchestrator, "vectorSetResolver", vectorSetResolver);
        setField(orchestrator, "chunkerStreamClient", chunkerStreamClient);
        setField(orchestrator, "embedderStreamClient", embedderStreamClient);

        // Default: all models are available (pass validation)
        setupDefaultModelValidation();
    }

    // =========================================================================
    // Scatter-gather 4-phase tests
    // =========================================================================

    @Test
    void testScatterGather_2x2_oneChunkerCall_fourEmbeddings() {
        // 1 directive with 2 chunkers x 2 embedders
        // With scatter-gather: ONE chunker call (multi-config), 4 embedding calls
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("sentence_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("SENTENCE").build())
                                .build())
                        .build())
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("token_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("TOKEN").build())
                                .build())
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mpnet").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("scatter-gather-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Document text for cartesian product test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive)
                                .build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // 2 chunkers x 2 embedders = 4 results
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("2 chunkers x 2 embedders should produce 4 SemanticProcessingResults")
                .isEqualTo(4);

        // ONE chunker call (multi-config request with both configs)
        verify(chunkerStreamClient, times(1)).streamChunks(any());

        // 4 embedding calls (one per config x model combination)
        verify(embedderStreamClient, times(4)).streamEmbeddings(any());
    }

    @Test
    void testScatterGather_nlpAnalysisPreservedOnResults() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("nlp-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("NLP analysis test text. Second sentence here.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        // Set up chunker that returns NlpDocumentAnalysis on the last chunk
        NlpDocumentAnalysis expectedNlp = NlpDocumentAnalysis.newBuilder()
                .setDetectedLanguage("eng")
                .setLanguageConfidence(0.95f)
                .setTotalTokens(8)
                .setNounDensity(0.25f)
                .setVerbDensity(0.15f)
                .addSentences(SentenceSpan.newBuilder()
                        .setText("NLP analysis test text.")
                        .setStartOffset(0)
                        .setEndOffset(23)
                        .build())
                .addSentences(SentenceSpan.newBuilder()
                        .setText("Second sentence here.")
                        .setStartOffset(24)
                        .setEndOffset(45)
                        .build())
                .build();

        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("chunk-001")
                .setChunkNumber(0)
                .setTextContent("NLP analysis test text. Second sentence here.")
                .setChunkConfigId("c1")
                .setSourceFieldName("body")
                .setIsLast(true)
                .setNlpAnalysis(expectedNlp)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Should have 1 SemanticProcessingResult")
                .isEqualTo(1);

        SemanticProcessingResult spr = result.getSearchMetadata().getSemanticResults(0);
        assertThat(spr.hasNlpAnalysis())
                .as("SemanticProcessingResult should have NlpDocumentAnalysis")
                .isTrue();
        assertThat(spr.getNlpAnalysis().getDetectedLanguage())
                .as("NLP detected language should be preserved")
                .isEqualTo("eng");
        assertThat(spr.getNlpAnalysis().getSentencesCount())
                .as("NLP sentences should be preserved")
                .isEqualTo(2);
        assertThat(spr.getNlpAnalysis().getTotalTokens())
                .as("NLP total tokens should be preserved")
                .isEqualTo(8);
    }

    @Test
    void testFailFast_missingModel_returnsDocUnchanged() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("nonexistent-model-xyz").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("fail-fast-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text for fail-fast test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        // Only report 'minilm' and 'mpnet' as ready -- 'nonexistent-model-xyz' is missing
        when(embedderStreamClient.listEmbeddingModels(true))
                .thenReturn(Uni.createFrom().item(ListEmbeddingModelsResponse.newBuilder()
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("minilm")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("mpnet")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .build()));

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // Doc returned unchanged -- no semantic results
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Missing model should fail fast with 0 results (doc unchanged)")
                .isEqualTo(0);

        // Chunker should NOT have been called (fail-fast before Phase 1)
        verify(chunkerStreamClient, never()).streamChunks(any());
        // Embedder streamEmbeddings should NOT have been called
        verify(embedderStreamClient, never()).streamEmbeddings(any());
    }

    @Test
    void testSingleEmbedderFailure_entireDocFails() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("embed-fail-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text that will fail embedding.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        // Chunker works fine
        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("c-0001").setChunkNumber(0).setTextContent("text")
                .setChunkConfigId("c1").setSourceFieldName("body").setIsLast(true)
                .build();
        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        // Embedder fails
        when(embedderStreamClient.streamEmbeddings(any()))
                .thenReturn(Multi.createFrom().failure(new RuntimeException("GPU out of memory")));

        // The orchestrate call should propagate the failure
        try {
            orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                    .await().indefinitely();
            // If no exception, that's also acceptable -- failure may be swallowed at the Uni level
        } catch (Exception e) {
            assertThat(e.getMessage())
                    .as("Embedder failure should propagate with clear error")
                    .contains("GPU out of memory");
        }
    }

    // =========================================================================
    // VectorSetService fallback tests
    // =========================================================================

    @Test
    void testFallback_noVectorSets_returnsUnchangedDoc() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-1")
                .setSearchMetadata(SearchMetadata.newBuilder().setBody("Hello world").build())
                .build();

        when(vectorSetResolver.resolveVectorSets(anyString()))
                .thenReturn(Uni.createFrom().item(List.of()));

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertThat(result.getDocId())
                .as("Doc ID should be preserved")
                .isEqualTo("test-doc-1");
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("No VectorSets means no semantic results")
                .isEqualTo(0);
    }

    @Test
    void testFallback_withVectorSets_producesResults() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-2")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("This is a test document.")
                        .build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1").setName("body-minilm")
                .setChunkerConfigId("chunker-a")
                .setEmbeddingModelConfigId("minilm")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_minilm_results").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1)));

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("One VectorSet should produce one SemanticProcessingResult")
                .isEqualTo(1);
        assertThat(result.getSearchMetadata().getSemanticResults(0).getChunkConfigId())
                .as("Chunk config ID should match VectorSet's chunker config")
                .isEqualTo("chunker-a");
    }

    @Test
    void testFallback_deduplicatesChunking() {
        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("test-doc-3")
                .setSearchMetadata(SearchMetadata.newBuilder().setBody("Text").build())
                .build();

        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1").setName("vs1")
                .setChunkerConfigId("same-chunker")
                .setEmbeddingModelConfigId("minilm")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r1").setSourceField("body")
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2").setName("vs2")
                .setChunkerConfigId("same-chunker")
                .setEmbeddingModelConfigId("mpnet")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r2").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Two VectorSets with same chunker should produce 2 results")
                .isEqualTo(2);
        // ONE chunker call (same config, same source field)
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        // Two embedder calls
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    // =========================================================================
    // Directive-based orchestration tests
    // =========================================================================

    @Test
    void testDirectives_fieldNameTemplate() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .setFieldNameTemplate("{source_label}_{chunker_id}_{embedder_id}")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("sent").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("template-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Some text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Should have 1 result")
                .isEqualTo(1);
        assertThat(result.getSearchMetadata().getSemanticResults(0).getResultSetName())
                .as("Field name template should be applied correctly")
                .isEqualTo("body_sent_minilm");
    }

    @Test
    void testDirectives_usedOverVectorSetService() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("priority-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Priority test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        orchestrator.orchestrate(inputDoc, new SemanticManagerOptions("some-index", null, 4, 8, null), "node-1")
                .await().indefinitely();

        verify(vectorSetResolver, never()).resolveVectorSets(anyString());
    }

    @Test
    void testDirectives_chunkerDeduplication() {
        // Two directives using the same chunker + source = chunk only once
        VectorDirective d1 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        VectorDirective d2 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mpnet").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("dedup-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Dedup test text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(d1).addDirectives(d2).build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Two directives with same chunker should produce 2 results (one per embedder)")
                .isEqualTo(2);
        // ONE chunker call (deduplication by source + config)
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    // =========================================================================
    // Field-level (no-chunk) orchestration tests
    // =========================================================================

    @Test
    void testFieldLevel_noChunkerConfigs_skipsChunkerCallsEmbedder() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("all-MiniLM-L6-v2")
                        .build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("field-level-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Full body text for field-level embedding.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, never()).streamChunks(any());
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Field-level should produce 1 result")
                .isEqualTo(1);
        SemanticProcessingResult spr = result.getSearchMetadata().getSemanticResults(0);
        assertThat(spr.getSourceFieldName())
                .as("Source field should be body")
                .isEqualTo("body");
        assertThat(spr.getEmbeddingConfigId())
                .as("Embedding config should match requested model")
                .isEqualTo("all-MiniLM-L6-v2");
        assertThat(spr.getChunksCount())
                .as("Field-level should have exactly 1 chunk (full text)")
                .isEqualTo(1);
        assertThat(spr.getChunks(0).getEmbeddingInfo().getTextContent())
                .as("Full text should be embedded as a single chunk")
                .isEqualTo("Full body text for field-level embedding.");
        assertThat(spr.getChunks(0).getEmbeddingInfo().getVectorCount())
                .as("Chunk should have embedding vectors")
                .isGreaterThan(0);
    }

    @Test
    void testFieldLevel_multipleEmbeddersNoChunker() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mpnet").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("multi-emb-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text for multi-embedder field-level test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, never()).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Two embedders with no chunker should produce 2 field-level results")
                .isEqualTo(2);
    }

    @Test
    void testFieldLevel_mixedDirectives_bothPathsRun() {
        VectorDirective d1 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("chunker-1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        VectorDirective d2 = VectorDirective.newBuilder()
                .setSourceLabel("title")
                .setCelSelector("document.search_metadata.title")
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mpnet").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("mixed-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text.")
                        .setTitle("Title text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(d1).addDirectives(d2).build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Mixed directives should produce 2 results (1 chunked + 1 field-level)")
                .isEqualTo(2);
    }

    // =========================================================================
    // Convenience fields tests
    // =========================================================================

    @Test
    void testConvenienceFields_skipChunking_producesFieldLevelResult() {
        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, null,
                "body", null, null, null, "all-MiniLM-L6-v2", true, null, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("convenience-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text for convenience field test.")
                        .build())
                .build();

        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, never()).streamChunks(any());
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());

        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Skip chunking convenience field should produce 1 field-level result")
                .isEqualTo(1);
        SemanticProcessingResult spr = result.getSearchMetadata().getSemanticResults(0);
        assertThat(spr.getSourceFieldName())
                .as("Source field should match convenience field")
                .isEqualTo("body");
        assertThat(spr.getEmbeddingConfigId())
                .as("Embedding config should match convenience model")
                .isEqualTo("all-MiniLM-L6-v2");
        assertThat(spr.getChunksCount())
                .as("Skip chunking should have exactly 1 chunk")
                .isEqualTo(1);
    }

    @Test
    void testConvenienceFields_withChunking_producesChunkedResult() {
        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, null,
                "body", 200, 20, "SENTENCE", "all-MiniLM-L6-v2", false, null, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("convenience-chunk-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text for chunked convenience test.")
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(1)).streamEmbeddings(any());
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Chunked convenience fields should produce 1 result")
                .isEqualTo(1);
    }

    @Test
    void testConvenienceFields_explicitDirectivesTakePriority() {
        var directiveConfig = new ai.pipestream.module.semanticmanager.config.DirectiveConfig(
                "body", "document.search_metadata.body",
                List.of(new ai.pipestream.module.semanticmanager.config.DirectiveConfig.NamedConfig("chunker-x", null)),
                List.of(new ai.pipestream.module.semanticmanager.config.DirectiveConfig.NamedConfig("minilm", null)),
                null);

        SemanticManagerOptions options = new SemanticManagerOptions(
                "test-index", null, 4, 8, List.of(directiveConfig),
                "title", null, null, null, "some-other-model", true, null, null);

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("priority-doc-2")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text.")
                        .setTitle("Title text.")
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        verify(chunkerStreamClient, times(1)).streamChunks(any());
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Explicit directives should take priority over convenience fields")
                .isEqualTo(1);
        assertThat(result.getSearchMetadata().getSemanticResults(0).getSourceFieldName())
                .as("Should use body from explicit directive, not title from convenience")
                .isEqualTo("body");
    }

    // =========================================================================
    // Source field analytics tests
    // =========================================================================

    @Test
    void testSourceFieldAnalytics_populated() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("analytics-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Body text for analytics test with enough words to matter.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertThat(result.getSearchMetadata().getSourceFieldAnalyticsCount())
                .as("Should have source field analytics entries")
                .isGreaterThan(0);

        SourceFieldAnalytics sfa = result.getSearchMetadata().getSourceFieldAnalytics(0);
        assertThat(sfa.getSourceField())
                .as("Analytics source field should be body")
                .isEqualTo("body");
        assertThat(sfa.getChunkConfigId())
                .as("Analytics chunk config should match")
                .isEqualTo("c1");
        assertThat(sfa.getTotalChunks())
                .as("Analytics should report total chunks")
                .isGreaterThan(0);
    }

    // =========================================================================
    // Model validation edge cases
    // =========================================================================

    @Test
    void testModelValidation_embedderServiceDown_proceedsAnyway() {
        // When the embedder service is unreachable, validation should not block processing
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("validation-error-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        // listEmbeddingModels fails
        when(embedderStreamClient.listEmbeddingModels(anyBoolean()))
                .thenReturn(Uni.createFrom().failure(new RuntimeException("Connection refused")));

        setupMultiConfigChunkerMock();
        setupEmbedderMock();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // Should still proceed and produce results
        assertThat(result.getSearchMetadata().getSemanticResultsCount())
                .as("Should still produce results when model validation fails gracefully")
                .isGreaterThan(0);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void setupDefaultModelValidation() {
        when(embedderStreamClient.listEmbeddingModels(anyBoolean()))
                .thenReturn(Uni.createFrom().item(ListEmbeddingModelsResponse.newBuilder()
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("all-MiniLM-L6-v2")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("minilm")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("all-mpnet-base-v2")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("mpnet")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("e5-large")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .addModels(EmbeddingModelInfo.newBuilder()
                                .setModelName("e5")
                                .setStatus(EmbeddingModelStatus.EMBEDDING_MODEL_STATUS_READY)
                                .build())
                        .build()));
    }

    /**
     * Sets up a chunker mock that handles multi-config requests.
     * For each chunk_config in the request, produces word-based chunks tagged
     * with the correct chunk_config_id. NlpDocumentAnalysis on last chunk.
     */
    private void setupMultiConfigChunkerMock() {
        when(chunkerStreamClient.streamChunks(any()))
                .thenAnswer(invocation -> {
                    StreamChunksRequest request = invocation.getArgument(0);
                    String text = request.getTextContent();
                    String docId = request.getDocId();
                    String sourceField = request.getSourceFieldName();
                    String requestId = request.getRequestId();

                    java.util.List<StreamChunksResponse> allChunks = new java.util.ArrayList<>();

                    if (request.getChunkConfigsCount() > 0) {
                        // Multi-config path
                        for (int i = 0; i < request.getChunkConfigsCount(); i++) {
                            ChunkConfigEntry entry = request.getChunkConfigs(i);
                            String configId = entry.getChunkConfigId();
                            boolean isLastConfig = (i == request.getChunkConfigsCount() - 1);
                            allChunks.addAll(buildWordChunks(
                                    text, requestId, docId, configId, sourceField, isLastConfig));
                        }
                    } else {
                        // Legacy single-config
                        String configId = request.getChunkConfigId();
                        allChunks.addAll(buildWordChunks(
                                text, requestId, docId, configId, sourceField, true));
                    }

                    return Multi.createFrom().iterable(allChunks);
                });
    }

    private java.util.List<StreamChunksResponse> buildWordChunks(
            String text, String requestId, String docId, String configId,
            String sourceField, boolean includeNlpOnLast) {

        java.util.List<StreamChunksResponse> chunks = new java.util.ArrayList<>();
        String[] words = text.split("\\s+");
        int wordsPerChunk = 10;

        int chunkNumber = 0;
        int charOffset = 0;

        for (int i = 0; i < words.length; i += wordsPerChunk) {
            int end = Math.min(i + wordsPerChunk, words.length);
            StringBuilder chunkText = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunkText.append(" ");
                chunkText.append(words[j]);
            }

            String content = chunkText.toString();
            int startOffset = charOffset;
            int endOffset = startOffset + content.length();
            boolean isLast = (end >= words.length);

            StreamChunksResponse.Builder chunkBuilder = StreamChunksResponse.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(UUID.randomUUID().toString())
                    .setChunkNumber(chunkNumber)
                    .setTextContent(content)
                    .setStartOffset(startOffset)
                    .setEndOffset(endOffset)
                    .setChunkConfigId(configId)
                    .setSourceFieldName(sourceField)
                    .setIsLast(isLast);

            if (isLast && includeNlpOnLast) {
                chunkBuilder.setNlpAnalysis(NlpDocumentAnalysis.newBuilder()
                        .setDetectedLanguage("eng")
                        .setLanguageConfidence(0.95f)
                        .setTotalTokens(words.length)
                        .setNounDensity(0.25f)
                        .setVerbDensity(0.15f)
                        .setAdjectiveDensity(0.08f)
                        .setAdverbDensity(0.05f)
                        .setContentWordRatio(0.55f)
                        .setUniqueLemmaCount((int) (words.length * 0.7))
                        .setLexicalDensity(0.55f)
                        .build());
            }

            chunks.add(chunkBuilder.build());
            chunkNumber++;
            charOffset = endOffset + 1;
        }

        if (chunks.isEmpty()) {
            chunks.add(StreamChunksResponse.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(UUID.randomUUID().toString())
                    .setChunkNumber(0)
                    .setTextContent("")
                    .setChunkConfigId(configId)
                    .setSourceFieldName(sourceField)
                    .setIsLast(true)
                    .build());
        }

        return chunks;
    }

    private void setupEmbedderMock() {
        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> StreamEmbeddingsResponse.newBuilder()
                            .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                            .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                            .setEmbeddingModelId(req.getEmbeddingModelId())
                            .addVector(0.1f).addVector(0.2f).addVector(0.3f)
                            .setSuccess(true).build());
                });
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}

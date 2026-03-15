package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.ChunkerStreamClient;
import ai.pipestream.module.semanticmanager.service.EmbedderStreamClient;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import ai.pipestream.module.semanticmanager.service.VectorSetResolver;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
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

        assertEquals("test-doc-1", result.getDocId());
        assertEquals(0, result.getSearchMetadata().getSemanticResultsCount());
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
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_minilm_results").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1)));

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        assertEquals("chunker-a", result.getSearchMetadata().getSemanticResults(0).getChunkConfigId());
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
                .setEmbeddingModelConfigId("model-a")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r1").setSourceField("body")
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2").setName("vs2")
                .setChunkerConfigId("same-chunker")
                .setEmbeddingModelConfigId("model-b")
                .setIndexName("test-index").setFieldName("f").setResultSetName("r2").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions("test-index", null, 4, 8, null);

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    // =========================================================================
    // Directive-based orchestration tests
    // =========================================================================

    @Test
    void testDirectives_cartesianProduct() {
        // 1 directive with 2 chunkers × 2 embedders = 4 results
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("sentence_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("sentence").build())
                                .build())
                        .build())
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("token_v1")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("token").build())
                                .build())
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("minilm")
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                        .setConfigId("mpnet")
                        .build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("directive-doc-1")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Document text for cartesian product test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive)
                                .build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        SemanticManagerOptions options = new SemanticManagerOptions();

        PipeDoc result = orchestrator.orchestrate(inputDoc, options, "node-1")
                .await().indefinitely();

        // 2 chunkers × 2 embedders = 4 results
        assertEquals(4, result.getSearchMetadata().getSemanticResultsCount());
        // But chunker should only be called TWICE (one per chunker config)
        verify(chunkerStreamClient, times(2)).streamChunks(any());
        // Embedder called 4 times (cartesian product)
        verify(embedderStreamClient, times(4)).streamEmbeddings(any());
    }

    @Test
    void testDirectives_fieldNameTemplate() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .setFieldNameTemplate("{source_label}_{chunker_id}_{embedder_id}")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("sent").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mini").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("template-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Some text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        assertEquals(1, result.getSearchMetadata().getSemanticResultsCount());
        assertEquals("body_sent_mini",
                result.getSearchMetadata().getSemanticResults(0).getResultSetName());
    }

    @Test
    void testDirectives_usedOverVectorSetService() {
        // When directives are present, VectorSetService should NOT be called
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("e1").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("priority-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Priority test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        orchestrator.orchestrate(inputDoc, new SemanticManagerOptions("some-index", null, 4, 8, null), "node-1")
                .await().indefinitely();

        // VectorSetResolver should never be called
        verify(vectorSetResolver, never()).resolveVectorSets(anyString());
    }

    @Test
    void testDirectives_chunkerDeduplication() {
        // Two directives using the same chunker + source = chunk only once
        VectorDirective d1 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb_a").build())
                .build();

        VectorDirective d2 = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("shared_chunker").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("emb_b").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("dedup-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Dedup test text.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(d1).addDirectives(d2).build())
                        .build())
                .build();

        setupMockChunkerAndEmbedder();

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // 2 results (one per embedder)
        assertEquals(2, result.getSearchMetadata().getSemanticResultsCount());
        // But chunker called only ONCE (deduplication)
        verify(chunkerStreamClient, times(1)).streamChunks(any());
        verify(embedderStreamClient, times(2)).streamEmbeddings(any());
    }

    @Test
    void testDirectives_partialEmbedderFailure() {
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("c1").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("ok_emb").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("fail_emb").build())
                .build();

        PipeDoc inputDoc = PipeDoc.newBuilder()
                .setDocId("partial-doc")
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Partial failure test.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive).build())
                        .build())
                .build();

        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("c-0001").setChunkNumber(0).setTextContent("text")
                .setChunkConfigId("c1").setSourceFieldName("body").setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

        when(embedderStreamClient.streamEmbeddings(any()))
                .thenAnswer(invocation -> {
                    @SuppressWarnings("unchecked")
                    Multi<StreamEmbeddingsRequest> reqs = (Multi<StreamEmbeddingsRequest>) invocation.getArgument(0);
                    return reqs.map(req -> {
                        if ("fail_emb".equals(req.getEmbeddingModelId())) {
                            throw new RuntimeException("Embedder down");
                        }
                        return StreamEmbeddingsResponse.newBuilder()
                                .setRequestId(req.getRequestId()).setDocId(req.getDocId())
                                .setChunkId(req.getChunkId()).setChunkConfigId(req.getChunkConfigId())
                                .setEmbeddingModelId(req.getEmbeddingModelId())
                                .addVector(0.5f).setSuccess(true).build();
                    });
                });

        PipeDoc result = orchestrator.orchestrate(inputDoc, new SemanticManagerOptions(), "node-1")
                .await().indefinitely();

        // At least 1 result (partial failure tolerance)
        assertTrue(result.getSearchMetadata().getSemanticResultsCount() >= 1);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void setupMockChunkerAndEmbedder() {
        StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                .setChunkId("chunk-0001").setChunkNumber(0).setTextContent("chunk text")
                .setChunkConfigId("default").setSourceFieldName("body").setIsLast(true)
                .build();

        when(chunkerStreamClient.streamChunks(any()))
                .thenReturn(Multi.createFrom().items(chunk));

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

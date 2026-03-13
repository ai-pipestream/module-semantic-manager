package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.module.v1.GetServiceRegistrationRequest;
import ai.pipestream.data.module.v1.PipeStepProcessorService;
import ai.pipestream.data.module.v1.ProcessDataRequest;
import ai.pipestream.data.module.v1.ProcessDataResponse;
import ai.pipestream.data.module.v1.ServiceMetadata;
import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.service.VectorSetResolver;
import ai.pipestream.opensearch.v1.VectorSet;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import io.quarkus.grpc.GrpcClient;
import io.quarkus.test.InjectMock;
import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Integration test for the SemanticManager module.
 *
 * Uses real in-process mock gRPC services for chunker and embedder:
 * - MockChunkerService: splits text into 10-word chunks
 * - MockEmbedderService: returns random 384-dim vectors
 *
 * VectorSetResolver is mocked via @InjectMock since opensearch-manager isn't available.
 *
 * Tests the full pipeline: ProcessData gRPC call → orchestrator → chunker stream → embedder stream → enriched PipeDoc.
 */
@QuarkusTest
class SemanticManagerServiceTest {

    private static final Logger log = LoggerFactory.getLogger(SemanticManagerServiceTest.class);

    @GrpcClient
    PipeStepProcessorService pipeStepProcessorService;

    @InjectMock
    VectorSetResolver vectorSetResolver;

    @BeforeEach
    void setupMocks() {
        // Default: return empty VectorSets (directives take priority anyway)
        when(vectorSetResolver.resolveVectorSets(anyString()))
                .thenReturn(Uni.createFrom().item(List.of()));
    }

    @Test
    void testGetServiceRegistration() {
        var registration = pipeStepProcessorService.getServiceRegistration(
                        GetServiceRegistrationRequest.newBuilder().build())
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        assertEquals("semantic-manager", registration.getModuleName());
        assertTrue(registration.hasJsonConfigSchema());
        assertTrue(registration.getJsonConfigSchema().contains("index_name"));
        assertTrue(registration.getHealthCheckPassed());
    }

    @Test
    void testProcessData_noDocument() {
        ProcessDataRequest request = ProcessDataRequest.newBuilder()
                .setMetadata(ServiceMetadata.newBuilder()
                        .setPipelineName("test-pipeline")
                        .setPipeStepName("semantic-manager-step")
                        .build())
                .setConfig(ProcessConfiguration.newBuilder().build())
                .build();

        var response = pipeStepProcessorService.processData(request)
                .subscribe().withSubscriber(UniAssertSubscriber.create())
                .awaitItem()
                .getItem();

        assertTrue(response.getSuccess());
        assertTrue(response.getProcessorLogsList().stream()
                .anyMatch(l -> l.contains("no document")));
    }

    @Test
    void testProcessData_withVectorSets_singleModel() {
        VectorSet vs = VectorSet.newBuilder()
                .setId("vs-1")
                .setName("body-minilm")
                .setChunkerConfigId("default-chunker")
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index")
                .setFieldName("embeddings")
                .setResultSetName("body_minilm")
                .setSourceField("body")
                .setVectorDimensions(384)
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs)));

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("The quick brown fox jumps over the lazy dog. " +
                                "This is a test document with enough text to produce multiple chunks. " +
                                "We want to verify that the semantic manager correctly orchestrates " +
                                "the chunking and embedding pipeline end to end.")
                        .build())
                .build();

        ProcessDataRequest request = buildRequest(testDoc, "test-index");

        ProcessDataResponse response = pipeStepProcessorService.processData(request)
                .await().atMost(Duration.ofSeconds(30));

        assertTrue(response.getSuccess(), "Processing should succeed. Logs: " + response.getProcessorLogsList());
        assertTrue(response.hasOutputDoc(), "Should return enriched document");
        assertEquals(testDoc.getDocId(), response.getOutputDoc().getDocId());

        PipeDoc outputDoc = response.getOutputDoc();
        int resultCount = outputDoc.getSearchMetadata().getSemanticResultsCount();
        log.info("Single model test: {} SemanticProcessingResults", resultCount);
        assertEquals(1, resultCount, "Should have exactly 1 SemanticProcessingResult");

        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);
        assertEquals("default-chunker", result.getChunkConfigId());
        assertEquals("all-MiniLM-L6-v2", result.getEmbeddingConfigId());
        assertEquals("body_minilm", result.getResultSetName());
        assertTrue(result.getChunksCount() > 0, "Should have at least 1 chunk");

        // Verify each chunk has embeddings
        for (SemanticChunk chunk : result.getChunksList()) {
            assertTrue(chunk.getEmbeddingInfo().getVectorCount() > 0,
                    "Chunk " + chunk.getChunkId() + " should have embedding vectors");
            log.info("  chunk[{}]: text='{}...', vectorDim={}",
                    chunk.getChunkNumber(),
                    chunk.getEmbeddingInfo().getTextContent().substring(0,
                            Math.min(30, chunk.getEmbeddingInfo().getTextContent().length())),
                    chunk.getEmbeddingInfo().getVectorCount());
        }
    }

    @Test
    void testProcessData_withVectorSets_multipleModels() {
        // 2 VectorSets with SAME chunker but different embedders
        VectorSet vs1 = VectorSet.newBuilder()
                .setId("vs-1").setName("body-minilm")
                .setChunkerConfigId("default-chunker")
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_minilm").setSourceField("body")
                .build();

        VectorSet vs2 = VectorSet.newBuilder()
                .setId("vs-2").setName("body-mpnet")
                .setChunkerConfigId("default-chunker")
                .setEmbeddingModelConfigId("all-mpnet-base-v2")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_mpnet").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs1, vs2)));

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("Artificial intelligence and machine learning are transforming how we process " +
                                "and understand large volumes of text data. Semantic search enables finding " +
                                "relevant information based on meaning rather than just keyword matching. " +
                                "Vector embeddings capture the semantic essence of text passages.")
                        .build())
                .build();

        ProcessDataRequest request = buildRequest(testDoc, "test-index");

        ProcessDataResponse response = pipeStepProcessorService.processData(request)
                .await().atMost(Duration.ofSeconds(30));

        assertTrue(response.getSuccess());
        PipeDoc outputDoc = response.getOutputDoc();
        assertEquals(2, outputDoc.getSearchMetadata().getSemanticResultsCount(),
                "Should have 2 results (one per embedder model)");

        // Both should share the same chunker config and chunk count
        SemanticProcessingResult r1 = outputDoc.getSearchMetadata().getSemanticResults(0);
        SemanticProcessingResult r2 = outputDoc.getSearchMetadata().getSemanticResults(1);
        assertEquals(r1.getChunksCount(), r2.getChunksCount(),
                "Same chunker should produce same number of chunks for both models");

        log.info("Multi-model test: {} chunks, {} results",
                r1.getChunksCount(), outputDoc.getSearchMetadata().getSemanticResultsCount());
    }

    @Test
    void testProcessData_withDirectives_cartesianProduct() {
        // 1 directive: 2 chunkers × 2 embedders = 4 SemanticProcessingResults
        VectorDirective directive = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("sentence-splitter")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("sentence").build())
                                .build())
                        .build())
                .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                        .setConfigId("token-splitter")
                        .setConfig(Struct.newBuilder()
                                .putFields("algorithm", Value.newBuilder().setStringValue("token").build())
                                .build())
                        .build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("mpnet").build())
                .setFieldNameTemplate("{source_label}_{chunker_id}_{embedder_id}")
                .build();

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("This document tests the cartesian product feature. " +
                                "With two chunkers and two embedders, we expect four semantic results. " +
                                "Each result represents a unique combination of chunking strategy " +
                                "and embedding model, giving downstream search maximum flexibility.")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(directive)
                                .build())
                        .build())
                .build();

        ProcessDataRequest request = buildRequest(testDoc, null);

        ProcessDataResponse response = pipeStepProcessorService.processData(request)
                .await().atMost(Duration.ofSeconds(30));

        assertTrue(response.getSuccess(), "Logs: " + response.getProcessorLogsList());
        PipeDoc outputDoc = response.getOutputDoc();

        assertEquals(4, outputDoc.getSearchMetadata().getSemanticResultsCount(),
                "2 chunkers × 2 embedders = 4 results");

        // VectorSetResolver should NOT have been called (directives take priority)
        verify(vectorSetResolver, never()).resolveVectorSets(anyString());

        // Verify field name templates were applied
        List<String> resultSetNames = outputDoc.getSearchMetadata().getSemanticResultsList().stream()
                .map(SemanticProcessingResult::getResultSetName)
                .toList();
        log.info("Cartesian product result set names: {}", resultSetNames);

        assertTrue(resultSetNames.contains("body_sentence-splitter_minilm"));
        assertTrue(resultSetNames.contains("body_sentence-splitter_mpnet"));
        assertTrue(resultSetNames.contains("body_token-splitter_minilm"));
        assertTrue(resultSetNames.contains("body_token-splitter_mpnet"));

        // Each result should have chunks with vectors
        for (SemanticProcessingResult result : outputDoc.getSearchMetadata().getSemanticResultsList()) {
            assertTrue(result.getChunksCount() > 0);
            for (SemanticChunk chunk : result.getChunksList()) {
                assertTrue(chunk.getEmbeddingInfo().getVectorCount() > 0,
                        "Every chunk should have an embedding vector");
            }
        }
    }

    @Test
    void testProcessData_withDirectives_multipleSourceFields() {
        // Two directives: one for body, one for title
        VectorDirective bodyDirective = VectorDirective.newBuilder()
                .setSourceLabel("body")
                .setCelSelector("document.search_metadata.body")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("chunker-a").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        VectorDirective titleDirective = VectorDirective.newBuilder()
                .setSourceLabel("title")
                .setCelSelector("document.search_metadata.title")
                .addChunkerConfigs(NamedChunkerConfig.newBuilder().setConfigId("chunker-a").build())
                .addEmbedderConfigs(NamedEmbedderConfig.newBuilder().setConfigId("minilm").build())
                .build();

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody("This is the body text that will be chunked and embedded separately from the title.")
                        .setTitle("An Important Document Title")
                        .setVectorSetDirectives(VectorSetDirectives.newBuilder()
                                .addDirectives(bodyDirective)
                                .addDirectives(titleDirective)
                                .build())
                        .build())
                .build();

        ProcessDataRequest request = buildRequest(testDoc, null);

        ProcessDataResponse response = pipeStepProcessorService.processData(request)
                .await().atMost(Duration.ofSeconds(30));

        assertTrue(response.getSuccess());
        PipeDoc outputDoc = response.getOutputDoc();
        assertEquals(2, outputDoc.getSearchMetadata().getSemanticResultsCount(),
                "Should have 2 results (one per source field)");

        // Verify both source fields are represented
        List<String> sourceFields = outputDoc.getSearchMetadata().getSemanticResultsList().stream()
                .map(SemanticProcessingResult::getSourceFieldName)
                .toList();
        assertTrue(sourceFields.contains("body"));
        assertTrue(sourceFields.contains("title"));
    }

    @Test
    void testProcessData_largerDocument_throughput() {
        // Test with a larger document to verify no bottlenecks
        StringBuilder largeBody = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            largeBody.append("Sentence number ").append(i)
                    .append(": the quick brown fox jumps over the lazy dog. ");
        }

        VectorSet vs = VectorSet.newBuilder()
                .setId("vs-1").setName("body-minilm")
                .setChunkerConfigId("large-doc-chunker")
                .setEmbeddingModelConfigId("all-MiniLM-L6-v2")
                .setIndexName("test-index").setFieldName("embeddings")
                .setResultSetName("body_minilm").setSourceField("body")
                .build();

        when(vectorSetResolver.resolveVectorSets("test-index"))
                .thenReturn(Uni.createFrom().item(List.of(vs)));

        PipeDoc testDoc = PipeDoc.newBuilder()
                .setDocId(UUID.randomUUID().toString())
                .setSearchMetadata(SearchMetadata.newBuilder()
                        .setBody(largeBody.toString())
                        .build())
                .build();

        ProcessDataRequest request = buildRequest(testDoc, "test-index");

        long startMs = System.currentTimeMillis();
        ProcessDataResponse response = pipeStepProcessorService.processData(request)
                .await().atMost(Duration.ofSeconds(60));
        long elapsedMs = System.currentTimeMillis() - startMs;

        assertTrue(response.getSuccess());
        PipeDoc outputDoc = response.getOutputDoc();
        SemanticProcessingResult result = outputDoc.getSearchMetadata().getSemanticResults(0);

        log.info("Large document test: {} chars, {} chunks, {} vectors per chunk, took {}ms",
                largeBody.length(), result.getChunksCount(),
                result.getChunks(0).getEmbeddingInfo().getVectorCount(), elapsedMs);

        assertTrue(result.getChunksCount() > 5, "Large doc should produce many chunks");
        assertTrue(elapsedMs < 30000, "Should complete within 30 seconds");
    }

    private ProcessDataRequest buildRequest(PipeDoc doc, String indexName) {
        ServiceMetadata metadata = ServiceMetadata.newBuilder()
                .setPipelineName("test-pipeline")
                .setPipeStepName("semantic-manager-step")
                .setStreamId(UUID.randomUUID().toString())
                .setCurrentHopNumber(1)
                .putContextParams("tenant", "test-tenant")
                .build();

        Struct.Builder configFields = Struct.newBuilder();
        if (indexName != null) {
            configFields.putFields("index_name",
                    Value.newBuilder().setStringValue(indexName).build());
        }

        ProcessConfiguration config = ProcessConfiguration.newBuilder()
                .setJsonConfig(configFields.build())
                .build();

        return ProcessDataRequest.newBuilder()
                .setDocument(doc)
                .setMetadata(metadata)
                .setConfig(config)
                .build();
    }
}

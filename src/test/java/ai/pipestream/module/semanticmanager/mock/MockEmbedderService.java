package ai.pipestream.module.semanticmanager.mock;

import ai.pipestream.semantic.v1.SemanticEmbedderService;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Multi;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * In-process mock embedder that returns random vectors.
 * Simulates a real embedding service without needing ML models.
 * Default dimension is 384 (MiniLM-sized), configurable by embedding model ID.
 */
@Singleton
@GrpcService
public class MockEmbedderService implements SemanticEmbedderService {

    private static final Logger log = LoggerFactory.getLogger(MockEmbedderService.class);
    private static final int DEFAULT_DIMENSIONS = 384;

    private final Random random = new Random(42); // deterministic seed for reproducibility

    @Override
    public Multi<StreamEmbeddingsResponse> streamEmbeddings(Multi<StreamEmbeddingsRequest> requests) {
        return requests.map(req -> {
            int dimensions = getDimensions(req.getEmbeddingModelId());

            StreamEmbeddingsResponse.Builder builder = StreamEmbeddingsResponse.newBuilder()
                    .setRequestId(req.getRequestId())
                    .setDocId(req.getDocId())
                    .setChunkId(req.getChunkId())
                    .setChunkConfigId(req.getChunkConfigId())
                    .setEmbeddingModelId(req.getEmbeddingModelId())
                    .setSuccess(true);

            // Generate random vector
            for (int i = 0; i < dimensions; i++) {
                builder.addVector((float) (random.nextGaussian() * 0.1));
            }

            log.debug("MockEmbedder: embedded chunk={} with model={}, dims={}",
                    req.getChunkId(), req.getEmbeddingModelId(), dimensions);

            return builder.build();
        });
    }

    private int getDimensions(String modelId) {
        if (modelId == null) return DEFAULT_DIMENSIONS;
        return switch (modelId.toLowerCase()) {
            case "all-minilm-l6-v2", "minilm" -> 384;
            case "all-mpnet-base-v2", "mpnet" -> 768;
            case "e5-large", "e5" -> 1024;
            default -> DEFAULT_DIMENSIONS;
        };
    }
}

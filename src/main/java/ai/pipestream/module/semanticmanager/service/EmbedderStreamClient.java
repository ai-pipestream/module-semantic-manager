package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.quarkus.dynamicgrpc.GrpcClientFactory;
import ai.pipestream.semantic.v1.MutinySemanticEmbedderServiceGrpc;
import ai.pipestream.semantic.v1.StreamEmbeddingsRequest;
import ai.pipestream.semantic.v1.StreamEmbeddingsResponse;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client wrapper for the SemanticEmbedderService bidirectional streaming RPC.
 * Uses DynamicGrpcClientFactory for service discovery via Stork/Consul.
 */
@ApplicationScoped
public class EmbedderStreamClient {

    private static final Logger log = LoggerFactory.getLogger(EmbedderStreamClient.class);
    private static final String SERVICE_NAME = "embedder-service";

    @Inject
    GrpcClientFactory grpcClientFactory;

    /**
     * Opens a bidirectional streaming embedding call. Sends chunks as they arrive,
     * receives embedded vectors back. The embedder batches internally for GPU efficiency.
     */
    public Multi<StreamEmbeddingsResponse> streamEmbeddings(Multi<StreamEmbeddingsRequest> requests) {
        log.info("Opening StreamEmbeddings bidirectional stream");

        return grpcClientFactory.getClient(SERVICE_NAME, MutinySemanticEmbedderServiceGrpc::newMutinyStub)
                .onItem().transformToMulti(stub -> stub.streamEmbeddings(requests));
    }
}

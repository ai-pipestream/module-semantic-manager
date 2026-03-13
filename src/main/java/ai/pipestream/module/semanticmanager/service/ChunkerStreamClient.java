package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.quarkus.dynamicgrpc.GrpcClientFactory;
import ai.pipestream.semantic.v1.MutinySemanticChunkerServiceGrpc;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Client wrapper for the SemanticChunkerService streaming RPC.
 * Uses DynamicGrpcClientFactory for service discovery via Stork/Consul.
 */
@ApplicationScoped
public class ChunkerStreamClient {

    private static final Logger log = LoggerFactory.getLogger(ChunkerStreamClient.class);
    private static final String SERVICE_NAME = "chunker-service";

    @Inject
    GrpcClientFactory grpcClientFactory;

    /**
     * Opens a server-streaming chunking call. Returns a Multi of chunk responses
     * that can be forwarded to embedder streams as they arrive.
     */
    public Multi<StreamChunksResponse> streamChunks(StreamChunksRequest request) {
        log.info("Opening StreamChunks: requestId={}, docId={}, configId={}, sourceField={}",
                request.getRequestId(), request.getDocId(),
                request.getChunkConfigId(), request.getSourceFieldName());

        return grpcClientFactory.getClient(SERVICE_NAME, MutinySemanticChunkerServiceGrpc::newMutinyStub)
                .onItem().transformToMulti(stub -> stub.streamChunks(request));
    }
}

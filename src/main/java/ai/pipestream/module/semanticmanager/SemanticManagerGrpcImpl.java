package ai.pipestream.module.semanticmanager;

import ai.pipestream.data.module.v1.*;
import ai.pipestream.data.v1.LogEntry;
import ai.pipestream.data.v1.LogEntrySource;
import ai.pipestream.data.v1.LogLevel;
import ai.pipestream.data.v1.ModuleLogOrigin;
import ai.pipestream.data.v1.PipeDoc;
import ai.pipestream.data.v1.ProcessConfiguration;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.module.semanticmanager.service.SemanticIndexingOrchestrator;
import ai.pipestream.server.meta.BuildInfoProvider;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import java.util.List;
import com.google.protobuf.util.JsonFormat;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Semantic Manager gRPC service implementation. Implements PipeStepProcessorService
 * so the pipeline engine treats it as a normal module step.
 *
 * Receives a PipeDoc, resolves VectorSets (the semantic "recipe"), fans out to
 * chunker and embedder services via streaming RPCs, and returns an enriched PipeDoc
 * with all SemanticProcessingResults populated.
 */
@Singleton
@GrpcService
public class SemanticManagerGrpcImpl implements PipeStepProcessorService {

    private static final Logger log = LoggerFactory.getLogger(SemanticManagerGrpcImpl.class);

    @Inject
    ObjectMapper objectMapper;

    @Inject
    SemanticIndexingOrchestrator orchestrator;

    @Inject
    BuildInfoProvider buildInfoProvider;

    @Override
    public Uni<ProcessDataResponse> processData(ProcessDataRequest request) {
        if (request == null) {
            log.error("Received null request");
            return Uni.createFrom().item(createErrorResponse("Request cannot be null", null));
        }

        return Uni.createFrom().item(request)
                .chain(this::doProcess)
                .onFailure().recoverWithItem(throwable -> {
                    String errorMessage = "Error in SemanticManager: " + throwable.getMessage();
                    log.error(errorMessage, throwable);
                    return createErrorResponse(errorMessage, throwable);
                });
    }

    private Uni<ProcessDataResponse> doProcess(ProcessDataRequest request) {
        if (!request.hasDocument()) {
            return Uni.createFrom().item(
                    ProcessDataResponse.newBuilder()
                            .setOutcome(ai.pipestream.data.module.v1.ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS)
                            .addLogEntries(moduleLog("Semantic manager: no document to process.", LogLevel.LOG_LEVEL_INFO))
                            .build());
        }

        PipeDoc inputDoc = request.getDocument();
        ServiceMetadata metadata = request.getMetadata();
        String nodeId = metadata.getPipeStepName();
        String streamId = metadata.getStreamId();

        log.info("SemanticManager processing doc: {}, step: {}, stream: {}",
                inputDoc.getDocId(), nodeId, streamId);

        // Parse configuration
        SemanticManagerOptions options = parseOptions(request.getConfig());
        List<LogEntry> auditLogs = new java.util.ArrayList<>();

        String configSummary = String.format(
                "Config: indexName=%s, hasDirectives=%s, directivesCount=%d, hasConvenienceFields=%s",
                options.effectiveIndexName(),
                options.hasDirectives(),
                options.directives() != null ? options.directives().size() : 0,
                options.hasConvenienceFields());
        log.info(configSummary);
        auditLogs.add(moduleLog(configSummary, LogLevel.LOG_LEVEL_INFO));

        if (options.hasDirectives()) {
            for (int i = 0; i < options.directives().size(); i++) {
                var d = options.directives().get(i);
                String directiveInfo = String.format(
                        "Directive %d: source='%s', selector='%s', chunkers=%d, embedders=%d",
                        i, d.sourceLabel(), d.celSelector(),
                        d.chunkerConfigs() != null ? d.chunkerConfigs().size() : 0,
                        d.embedderConfigs() != null ? d.embedderConfigs().size() : 0);
                log.info(directiveInfo);
                auditLogs.add(moduleLog(directiveInfo, LogLevel.LOG_LEVEL_INFO));
            }
        } else if (options.hasConvenienceFields()) {
            String convenienceInfo = String.format(
                    "Convenience fields: source_field=%s, skip_chunking=%s, embedding_model=%s, chunk_size=%s",
                    options.sourceField(), options.skipChunking(), options.embeddingModel(), options.chunkSize());
            log.info(convenienceInfo);
            auditLogs.add(moduleLog(convenienceInfo, LogLevel.LOG_LEVEL_INFO));
        }

        // Run orchestration
        long startTime = System.currentTimeMillis();
        return orchestrator.orchestrate(inputDoc, options, nodeId)
                .map(enrichedDoc -> {
                    long duration = System.currentTimeMillis() - startTime;
                    int semanticResultCount = enrichedDoc.hasSearchMetadata()
                            ? enrichedDoc.getSearchMetadata().getSemanticResultsCount()
                            : 0;

                    String resultMsg = String.format(
                            "Semantic manager produced %d SemanticProcessingResults for doc: %s in %dms",
                            semanticResultCount, inputDoc.getDocId(), duration);
                    log.info(resultMsg);
                    auditLogs.add(moduleLog(resultMsg, LogLevel.LOG_LEVEL_INFO));

                    if (semanticResultCount == 0 && options.hasDirectives()) {
                        auditLogs.add(moduleLog("Warning: directives were provided but no results produced. " +
                                "Check that chunker and embedder services are running and registered " +
                                "with the correct Consul service names.", LogLevel.LOG_LEVEL_WARN));
                    }

                    // Determine expected vs actual result count for partial detection
                    int expectedCount = countExpectedResults(options);
                    var outcomeEnum = ai.pipestream.data.module.v1.ProcessingOutcome.PROCESSING_OUTCOME_SUCCESS;
                    ProcessDataResponse.Builder respBuilder = ProcessDataResponse.newBuilder()
                            .setOutputDoc(enrichedDoc)
                            .addAllLogEntries(auditLogs);

                    if (semanticResultCount < expectedCount && semanticResultCount > 0) {
                        outcomeEnum = ai.pipestream.data.module.v1.ProcessingOutcome.PROCESSING_OUTCOME_PARTIAL;
                        String partialMsg = String.format("Partial: %d of %d expected embedding strategies produced results",
                                semanticResultCount, expectedCount);
                        respBuilder.addPartialFailureDetails(partialMsg);
                        auditLogs.add(moduleLog(partialMsg, LogLevel.LOG_LEVEL_WARN));
                    } else if (semanticResultCount == 0 && expectedCount > 0) {
                        outcomeEnum = ai.pipestream.data.module.v1.ProcessingOutcome.PROCESSING_OUTCOME_FAILURE;
                    }

                    return respBuilder.setOutcome(outcomeEnum).build();
                });
    }

    private SemanticManagerOptions parseOptions(ProcessConfiguration config) {
        try {
            if (config != null && config.hasJsonConfig() && config.getJsonConfig().getFieldsCount() > 0) {
                Struct jsonConfig = config.getJsonConfig();
                log.info("Config Struct keys: {}", jsonConfig.getFieldsMap().keySet());
                String jsonStr = JsonFormat.printer().print(jsonConfig);
                log.debug("Config JSON for parsing: {}", jsonStr);
                SemanticManagerOptions parsed = objectMapper.readValue(jsonStr, SemanticManagerOptions.class);
                log.info("Parsed SemanticManagerOptions: indexName={}, hasDirectives={}, directivesCount={}",
                        parsed.effectiveIndexName(),
                        parsed.hasDirectives(),
                        parsed.directives() != null ? parsed.directives().size() : 0);
                return parsed;
            }
        } catch (Exception e) {
            log.warn("Failed to parse SemanticManagerOptions, using defaults: {}", e.getMessage(), e);
        }
        return new SemanticManagerOptions();
    }

    @Override
    public Uni<GetServiceRegistrationResponse> getServiceRegistration(GetServiceRegistrationRequest request) {
        log.info("Semantic manager service registration requested");

        Capabilities capabilities = Capabilities.newBuilder()
                .addTypes(CapabilityType.CAPABILITY_TYPE_UNSPECIFIED)
                .build();

        GetServiceRegistrationResponse.Builder responseBuilder = GetServiceRegistrationResponse.newBuilder()
                .setModuleName("semantic-manager")
                .setVersion(buildInfoProvider.getVersion())
                .setDisplayName("Semantic Manager")
                .setDescription("Semantic indexing coordinator - multiplexes chunking and embedding across VectorSets")
                .setJsonConfigSchema(SemanticManagerOptions.getJsonV7Schema())
                .setCapabilities(capabilities)
                .putAllMetadata(buildInfoProvider.registrationMetadata())
                .setHealthCheckPassed(true)
                .setHealthCheckMessage("Semantic manager module is ready");

        return Uni.createFrom().item(responseBuilder.build());
    }

    private static int countExpectedResults(SemanticManagerOptions options) {
        if (options.hasDirectives()) {
            int count = 0;
            for (var directive : options.directives()) {
                int chunkers = directive.chunkerConfigs() != null ? directive.chunkerConfigs().size() : 1;
                int embedders = directive.embedderConfigs() != null ? directive.embedderConfigs().size() : 1;
                count += chunkers * embedders;
            }
            return count;
        }
        // Convenience fields = 1 result
        return options.hasConvenienceFields() ? 1 : 0;
    }

    private static LogEntry moduleLog(String message, LogLevel level) {
        return LogEntry.newBuilder()
            .setSource(LogEntrySource.LOG_ENTRY_SOURCE_MODULE)
            .setLevel(level)
            .setMessage(message)
            .setTimestampEpochMs(System.currentTimeMillis())
            .setModule(ModuleLogOrigin.newBuilder().setModuleName("semantic-manager").build())
            .build();
    }

    private ProcessDataResponse createErrorResponse(String errorMessage, Throwable e) {
        ProcessDataResponse.Builder responseBuilder = ProcessDataResponse.newBuilder();
        responseBuilder.setOutcome(ai.pipestream.data.module.v1.ProcessingOutcome.PROCESSING_OUTCOME_FAILURE);
        responseBuilder.addLogEntries(moduleLog(errorMessage, LogLevel.LOG_LEVEL_ERROR));

        Struct.Builder errorDetailsBuilder = Struct.newBuilder();
        errorDetailsBuilder.putFields("error_message",
                Value.newBuilder().setStringValue(errorMessage).build());
        if (e != null) {
            errorDetailsBuilder.putFields("error_type",
                    Value.newBuilder().setStringValue(e.getClass().getName()).build());
            if (e.getCause() != null) {
                errorDetailsBuilder.putFields("error_cause",
                        Value.newBuilder().setStringValue(e.getCause().getMessage()).build());
            }
        }
        responseBuilder.setErrorDetails(errorDetailsBuilder.build());
        return responseBuilder.build();
    }

}

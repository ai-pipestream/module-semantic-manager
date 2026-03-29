package ai.pipestream.module.semanticmanager.service;

import ai.pipestream.data.v1.*;
import ai.pipestream.module.semanticmanager.config.DirectiveConfig;
import ai.pipestream.module.semanticmanager.config.SemanticManagerOptions;
import ai.pipestream.opensearch.v1.VectorSet;
import ai.pipestream.semantic.v1.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Struct;
import com.google.protobuf.util.JsonFormat;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Core orchestrator for semantic indexing using a 4-phase scatter-gather pattern.
 * <p>
 * Supports two recipe sources:
 * <ol>
 *   <li><b>VectorSetDirectives on the doc</b> (primary) — populated by the mapper step.</li>
 *   <li><b>VectorSetService</b> (fallback) — queries opensearch-manager for VectorSets.</li>
 * </ol>
 * <p>
 * The 4 phases are:
 * <ul>
 *   <li><b>Phase 0 — Validate Models:</b> Fail-fast if any requested embedding model is unavailable.</li>
 *   <li><b>Phase 1 — Chunk:</b> ONE chunker call per source text with ALL chunk configs.</li>
 *   <li><b>Phase 2 — Embed:</b> Parallel fan-out of all (chunk_config x embedder) combinations.</li>
 *   <li><b>Phase 3 — Assemble:</b> Build enriched PipeDoc with SemanticProcessingResults.</li>
 * </ul>
 */
@ApplicationScoped
public class SemanticIndexingOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(SemanticIndexingOrchestrator.class);

    private static final String DEFAULT_FIELD_NAME_TEMPLATE = "{source_label}_{chunker_id}_{embedder_id}";

    @Inject
    VectorSetResolver vectorSetResolver;

    @Inject
    ChunkerStreamClient chunkerStreamClient;

    @Inject
    EmbedderStreamClient embedderStreamClient;

    /**
     * Orchestrates semantic indexing for a document. Checks for directives on the doc first,
     * falls back to VectorSetService resolution.
     */
    public Uni<PipeDoc> orchestrate(PipeDoc inputDoc, SemanticManagerOptions options, String nodeId) {
        String docId = inputDoc.getDocId();

        // Priority 1: use directives from the doc if present
        if (hasDirectives(inputDoc)) {
            log.info("Using VectorSetDirectives from doc: {} ({} directives)",
                    docId, inputDoc.getSearchMetadata().getVectorSetDirectives().getDirectivesCount());
            return orchestrateFromDirectives(inputDoc, nodeId);
        }

        // Priority 2: use directives from module config (node custom_config)
        if (options.hasDirectives()) {
            log.info("Using directives from module config for doc: {} ({} directives)",
                    docId, options.directives().size());
            return orchestrateFromConfigDirectives(inputDoc, options, nodeId);
        }

        // Priority 2.5: if no explicit directives but convenience fields are set, build an implicit directive
        if (options.hasConvenienceFields()) {
            log.info("Building implicit directive from convenience fields for doc: {} (source_field={}, skip_chunking={}, model={})",
                    docId, options.sourceField(), options.skipChunking(), options.embeddingModel());
            DirectiveConfig implicit = options.toImplicitDirective();
            SemanticManagerOptions implicitOptions = new SemanticManagerOptions(
                    options.indexName(), options.vectorSetIds(),
                    options.maxConcurrentChunkers(), options.maxConcurrentEmbedders(),
                    java.util.List.of(implicit));
            return orchestrateFromConfigDirectives(inputDoc, implicitOptions, nodeId);
        }

        // Priority 3: resolve from VectorSetService
        log.info("No directives on doc or config: {}, falling back to VectorSetService for index: {}",
                docId, options.effectiveIndexName());
        return orchestrateFromVectorSetService(inputDoc, options, nodeId);
    }

    private boolean hasDirectives(PipeDoc doc) {
        return doc.hasSearchMetadata()
                && doc.getSearchMetadata().hasVectorSetDirectives()
                && doc.getSearchMetadata().getVectorSetDirectives().getDirectivesCount() > 0;
    }

    // =========================================================================
    // Directive-based orchestration (primary path) — 4-phase scatter-gather
    // =========================================================================

    private Uni<PipeDoc> orchestrateFromDirectives(PipeDoc inputDoc, String nodeId) {
        String docId = inputDoc.getDocId();
        VectorSetDirectives directives = inputDoc.getSearchMetadata().getVectorSetDirectives();

        // Parse directives into work items
        // Group by source text for chunking (one call per source text)
        // Track which (chunk_config x embedder) combos produce results
        Map<String, SourceTextWork> sourceTextWorkMap = new LinkedHashMap<>();
        List<FieldLevelTarget> fieldLevelTargets = new ArrayList<>();

        for (VectorDirective directive : directives.getDirectivesList()) {
            String sourceLabel = directive.getSourceLabel();
            String sourceText = extractTextByCelSelector(inputDoc, directive.getCelSelector(), sourceLabel);

            if (sourceText == null || sourceText.isEmpty()) {
                log.warn("CEL selector '{}' returned no text for source_label '{}' in doc: {}",
                        directive.getCelSelector(), sourceLabel, docId);
                continue;
            }

            String template = directive.hasFieldNameTemplate()
                    ? directive.getFieldNameTemplate() : DEFAULT_FIELD_NAME_TEMPLATE;

            if (directive.getChunkerConfigsCount() == 0) {
                // Field-level: no chunking, embed raw text directly
                for (NamedEmbedderConfig embedderCfg : directive.getEmbedderConfigsList()) {
                    fieldLevelTargets.add(new FieldLevelTarget(
                            sourceText, sourceLabel,
                            embedderCfg.getConfigId(), embedderCfg.getConfig(), template));
                }
            } else {
                // Standard chunking path — group by source text
                SourceTextWork work = sourceTextWorkMap.computeIfAbsent(sourceLabel,
                        k -> new SourceTextWork(sourceText, sourceLabel, new LinkedHashMap<>()));

                for (NamedChunkerConfig chunkerCfg : directive.getChunkerConfigsList()) {
                    String chunkConfigId = chunkerCfg.getConfigId();

                    // Add this chunk config to the source work if not already present
                    work.chunkConfigs().putIfAbsent(chunkConfigId,
                            new ChunkConfigWork(chunkConfigId, chunkerCfg.getConfig(), new ArrayList<>()));

                    ChunkConfigWork ccWork = work.chunkConfigs().get(chunkConfigId);

                    // Each embedder in this directive gets paired with this chunk config
                    for (NamedEmbedderConfig embedderCfg : directive.getEmbedderConfigsList()) {
                        ccWork.embedderTargets().add(new EmbedderTarget(
                                embedderCfg.getConfigId(), embedderCfg.getConfig(), template));
                    }
                }
            }
        }

        if (sourceTextWorkMap.isEmpty() && fieldLevelTargets.isEmpty()) {
            log.warn("No valid directives produced work items for doc: {}", docId);
            return Uni.createFrom().item(inputDoc);
        }

        log.info("Directive orchestration: {} source texts with chunking, {} field-level targets for doc: {}",
                sourceTextWorkMap.size(), fieldLevelTargets.size(), docId);

        // ─── Phase 0: Validate embedding models ───
        Set<String> requiredModelIds = new LinkedHashSet<>();
        for (SourceTextWork work : sourceTextWorkMap.values()) {
            for (ChunkConfigWork ccWork : work.chunkConfigs().values()) {
                for (EmbedderTarget et : ccWork.embedderTargets()) {
                    requiredModelIds.add(et.embedderConfigId());
                }
            }
        }
        for (FieldLevelTarget flt : fieldLevelTargets) {
            requiredModelIds.add(flt.embedderConfigId());
        }

        return validateModels(requiredModelIds, docId)
                .chain(validationOk -> {
                    if (!validationOk) {
                        log.error("Model validation failed for doc: {}. Returning doc unchanged.", docId);
                        return Uni.createFrom().item(inputDoc);
                    }

                    // ─── Phase 1: Chunk (one call per source text) ───
                    List<Uni<SourceTextChunkResult>> chunkUnis = new ArrayList<>();
                    for (SourceTextWork work : sourceTextWorkMap.values()) {
                        chunkUnis.add(chunkSourceText(work, docId));
                    }

                    // Collect all field-level targets into a separate Uni
                    Uni<List<AssemblyOutput>> fieldLevelUni;
                    if (!fieldLevelTargets.isEmpty()) {
                        fieldLevelUni = processFieldLevelTargets(inputDoc, fieldLevelTargets, nodeId);
                    } else {
                        fieldLevelUni = Uni.createFrom().item(Collections.emptyList());
                    }

                    if (chunkUnis.isEmpty()) {
                        // Only field-level work
                        return fieldLevelUni.map(fieldResults ->
                                assembleDocument(inputDoc, Collections.emptyList(), fieldResults, nodeId));
                    }

                    // Combine all chunk results
                    return Uni.combine().all().unis(chunkUnis)
                            .with(rawResults -> {
                                List<SourceTextChunkResult> chunkResults = new ArrayList<>();
                                for (Object r : rawResults) {
                                    chunkResults.add((SourceTextChunkResult) r);
                                }
                                return chunkResults;
                            })
                            .chain(chunkResults -> {
                                // ─── Phase 2: Embed (parallel fan-out) ───
                                List<Uni<AssemblyOutput>> embedUnis = new ArrayList<>();

                                for (int i = 0; i < chunkResults.size(); i++) {
                                    SourceTextChunkResult cr = chunkResults.get(i);
                                    SourceTextWork work = new ArrayList<>(sourceTextWorkMap.values()).get(i);

                                    for (ChunkConfigWork ccWork : work.chunkConfigs().values()) {
                                        String chunkConfigId = ccWork.chunkConfigId();
                                        List<StreamChunksResponse> chunksForConfig =
                                                cr.chunksByConfigId().getOrDefault(chunkConfigId, Collections.emptyList());

                                        if (chunksForConfig.isEmpty()) {
                                            log.warn("No chunks for config {} on source {}", chunkConfigId, work.sourceLabel());
                                            continue;
                                        }

                                        for (EmbedderTarget et : ccWork.embedderTargets()) {
                                            String resultSetName = resolveResultSetName(
                                                    et.template(), work.sourceLabel(),
                                                    chunkConfigId, et.embedderConfigId());

                                            embedUnis.add(embedChunks(
                                                    docId, chunksForConfig, chunkConfigId,
                                                    et.embedderConfigId(), et.embedderConfig(),
                                                    work.sourceLabel(), resultSetName, nodeId,
                                                    cr.nlpAnalysis()));
                                        }
                                    }
                                }

                                if (embedUnis.isEmpty()) {
                                    return fieldLevelUni.map(fieldResults ->
                                            assembleDocument(inputDoc, Collections.emptyList(), fieldResults, nodeId));
                                }

                                // Combine embed results with field-level results
                                Uni<List<AssemblyOutput>> allEmbedUni = Uni.combine().all().unis(embedUnis)
                                        .with(rawEmbedResults -> {
                                            List<AssemblyOutput> results = new ArrayList<>();
                                            for (Object r : rawEmbedResults) {
                                                results.add((AssemblyOutput) r);
                                            }
                                            return results;
                                        });

                                return Uni.combine().all().unis(allEmbedUni, fieldLevelUni)
                                        .with((embedResults, fieldResults) ->
                                                assembleDocument(inputDoc, embedResults, fieldResults, nodeId));
                            });
                });
    }

    // ─── Phase 0: Model Validation ───

    /**
     * Validates that all required embedding models are available.
     * Returns true if all models are ready, false otherwise.
     * On error (e.g., embedder service unreachable), logs warning and returns true
     * to allow processing to proceed (the embed phase will fail with better errors).
     */
    private Uni<Boolean> validateModels(Set<String> requiredModelIds, String docId) {
        if (requiredModelIds.isEmpty()) {
            return Uni.createFrom().item(true);
        }

        log.info("Phase 0: Validating {} embedding models for doc {}: {}",
                requiredModelIds.size(), docId, requiredModelIds);

        return embedderStreamClient.listEmbeddingModels(true)
                .map(response -> {
                    // Build lookup set from ALL model identifiers — enum name, serving name, and HuggingFace ID.
                    // Directives may use any of these naming conventions.
                    Set<String> readyIdentifiers = new java.util.HashSet<>();
                    for (EmbeddingModelInfo info : response.getModelsList()) {
                        readyIdentifiers.add(info.getEnumName());
                        readyIdentifiers.add(info.getModelName());
                        readyIdentifiers.add(info.getHuggingFaceId());
                    }

                    List<String> missing = requiredModelIds.stream()
                            .filter(id -> !readyIdentifiers.contains(id))
                            .toList();

                    if (!missing.isEmpty()) {
                        log.error("Phase 0 FAILED: embedding models not ready for doc {}: {}. " +
                                "Available identifiers: {}. Document will be returned unchanged for DLQ.",
                                docId, missing, readyIdentifiers);
                        return false;
                    }

                    log.info("Phase 0: All {} embedding models validated for doc {}",
                            requiredModelIds.size(), docId);
                    return true;
                })
                .onFailure().recoverWithItem(error -> {
                    log.warn("Phase 0: Could not contact embedder for model validation (doc {}): {}. " +
                            "Proceeding — embed phase will fail with specific errors.",
                            docId, error.getMessage());
                    return true; // proceed, embed phase will handle failures
                });
    }

    // ─── Phase 1: Chunk ───

    /**
     * Chunks a single source text with ALL its chunk configs in ONE gRPC call.
     * Returns chunks grouped by chunk_config_id, plus the NlpDocumentAnalysis.
     */
    private Uni<SourceTextChunkResult> chunkSourceText(SourceTextWork work, String docId) {
        String requestId = UUID.randomUUID().toString();

        StreamChunksRequest.Builder reqBuilder = StreamChunksRequest.newBuilder()
                .setRequestId(requestId)
                .setDocId(docId)
                .setSourceFieldName(work.sourceLabel())
                .setTextContent(work.sourceText());

        // Add all chunk configs as typed ChunkConfigEntry entries
        for (ChunkConfigWork ccWork : work.chunkConfigs().values()) {
            ChunkConfigEntry.Builder entryBuilder = ChunkConfigEntry.newBuilder()
                    .setChunkConfigId(ccWork.chunkConfigId());

            // Convert the JSON-style Struct config to typed ChunkerConfig proto
            ai.pipestream.semantic.v1.ChunkerConfig typedConfig =
                    convertToTypedChunkerConfig(ccWork.chunkerConfig());
            if (typedConfig != null) {
                entryBuilder.setConfig(typedConfig);
            }

            reqBuilder.addChunkConfigs(entryBuilder.build());
        }

        // For backward compat: if there's only one config, also set the legacy fields
        if (work.chunkConfigs().size() == 1) {
            ChunkConfigWork onlyConfig = work.chunkConfigs().values().iterator().next();
            reqBuilder.setChunkConfigId(onlyConfig.chunkConfigId());
            if (onlyConfig.chunkerConfig() != null) {
                reqBuilder.setChunkerConfig(onlyConfig.chunkerConfig());
            }
        }

        log.info("Phase 1: Chunking source '{}' for doc {} with {} configs: {}",
                work.sourceLabel(), docId, work.chunkConfigs().size(), work.chunkConfigs().keySet());

        return chunkerStreamClient.streamChunks(reqBuilder.build())
                .collect().asList()
                .map(allChunks -> {
                    // Group chunks by chunk_config_id
                    Map<String, List<StreamChunksResponse>> grouped = new LinkedHashMap<>();
                    NlpDocumentAnalysis nlpAnalysis = null;

                    for (StreamChunksResponse chunk : allChunks) {
                        String configId = chunk.getChunkConfigId();
                        grouped.computeIfAbsent(configId, k -> new ArrayList<>()).add(chunk);

                        // Capture NLP analysis from the last chunk
                        if (chunk.getIsLast() && chunk.hasNlpAnalysis()) {
                            nlpAnalysis = chunk.getNlpAnalysis();
                        }
                    }

                    log.info("Phase 1: Chunked source '{}' for doc {}: {} total chunks across {} configs",
                            work.sourceLabel(), docId, allChunks.size(), grouped.size());

                    return new SourceTextChunkResult(grouped, nlpAnalysis);
                });
    }

    /**
     * Converts a Struct-based chunker config to the typed ChunkerConfig proto.
     * Handles JSON keys: algorithm, chunkSize, chunk_size, chunkOverlap, chunk_overlap,
     * cleanText, clean_text, preserveUrls, preserve_urls.
     */
    private ai.pipestream.semantic.v1.ChunkerConfig convertToTypedChunkerConfig(Struct structConfig) {
        if (structConfig == null || structConfig.getFieldsCount() == 0) {
            return null;
        }

        ai.pipestream.semantic.v1.ChunkerConfig.Builder builder =
                ai.pipestream.semantic.v1.ChunkerConfig.newBuilder();

        var fields = structConfig.getFieldsMap();

        // Algorithm
        String algorithm = getStringField(fields, "algorithm");
        if (algorithm != null) {
            switch (algorithm.toUpperCase()) {
                case "TOKEN" -> builder.setAlgorithm(ChunkAlgorithm.CHUNK_ALGORITHM_TOKEN);
                case "SENTENCE" -> builder.setAlgorithm(ChunkAlgorithm.CHUNK_ALGORITHM_SENTENCE);
                case "CHARACTER" -> builder.setAlgorithm(ChunkAlgorithm.CHUNK_ALGORITHM_CHARACTER);
            }
        }

        // Chunk size
        Double chunkSize = getNumberField(fields, "chunkSize", "chunk_size");
        if (chunkSize != null) {
            builder.setChunkSize(chunkSize.intValue());
        }

        // Chunk overlap
        Double chunkOverlap = getNumberField(fields, "chunkOverlap", "chunk_overlap");
        if (chunkOverlap != null) {
            builder.setChunkOverlap(chunkOverlap.intValue());
        }

        // Clean text
        Boolean cleanText = getBoolField(fields, "cleanText", "clean_text");
        if (cleanText != null) {
            builder.setCleanText(cleanText);
        }

        // Preserve URLs
        Boolean preserveUrls = getBoolField(fields, "preserveUrls", "preserve_urls");
        if (preserveUrls != null) {
            builder.setPreserveUrls(preserveUrls);
        }

        return builder.build();
    }

    private String getStringField(Map<String, com.google.protobuf.Value> fields, String... keys) {
        for (String key : keys) {
            com.google.protobuf.Value v = fields.get(key);
            if (v != null && v.getKindCase() == com.google.protobuf.Value.KindCase.STRING_VALUE) {
                return v.getStringValue();
            }
        }
        return null;
    }

    private Double getNumberField(Map<String, com.google.protobuf.Value> fields, String... keys) {
        for (String key : keys) {
            com.google.protobuf.Value v = fields.get(key);
            if (v != null && v.getKindCase() == com.google.protobuf.Value.KindCase.NUMBER_VALUE) {
                return v.getNumberValue();
            }
        }
        return null;
    }

    private Boolean getBoolField(Map<String, com.google.protobuf.Value> fields, String... keys) {
        for (String key : keys) {
            com.google.protobuf.Value v = fields.get(key);
            if (v != null && v.getKindCase() == com.google.protobuf.Value.KindCase.BOOL_VALUE) {
                return v.getBoolValue();
            }
        }
        return null;
    }

    // ─── Phase 2: Embed ───

    /**
     * Embeds chunks for a single (chunk_config x embedder) combination.
     * Builds StreamEmbeddingsRequests from the chunk list and sends them as a Multi.
     */
    private Uni<AssemblyOutput> embedChunks(
            String docId,
            List<StreamChunksResponse> chunks,
            String chunkConfigId,
            String embedderConfigId,
            Struct embedderConfig,
            String sourceFieldName,
            String resultSetName,
            String nodeId,
            NlpDocumentAnalysis nlpAnalysis) {

        String requestId = UUID.randomUUID().toString();

        List<StreamEmbeddingsRequest> embeddingRequests = new ArrayList<>();
        for (StreamChunksResponse chunk : chunks) {
            StreamEmbeddingsRequest.Builder builder = StreamEmbeddingsRequest.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(chunk.getChunkId())
                    .setTextContent(chunk.getTextContent())
                    .setChunkConfigId(chunkConfigId)
                    .setEmbeddingModelId(embedderConfigId);
            if (embedderConfig != null) {
                builder.setEmbedderConfig(embedderConfig);
            }
            embeddingRequests.add(builder.build());
        }

        Multi<StreamEmbeddingsRequest> requestMulti = Multi.createFrom().iterable(embeddingRequests);

        return embedderStreamClient.streamEmbeddings(requestMulti)
                .collect().asList()
                .map(responses -> assembleResult(
                        chunks, responses, sourceFieldName, chunkConfigId,
                        embedderConfigId, resultSetName, nodeId, nlpAnalysis));
    }

    // ─── Phase 3: Assemble ───

    /**
     * Assembles the final PipeDoc from all embed results and field-level results.
     */
    private PipeDoc assembleDocument(
            PipeDoc inputDoc,
            List<AssemblyOutput> embedResults,
            List<AssemblyOutput> fieldResults,
            String nodeId) {

        String docId = inputDoc.getDocId();
        PipeDoc.Builder outputDocBuilder = inputDoc.toBuilder();
        SearchMetadata.Builder smBuilder = inputDoc.hasSearchMetadata()
                ? inputDoc.getSearchMetadata().toBuilder()
                : SearchMetadata.newBuilder();

        // Collect all outputs; deduplicate SourceFieldAnalytics by (source_field, chunk_config_id)
        Map<String, SourceFieldAnalytics.Builder> sfaMap = new LinkedHashMap<>();

        List<AssemblyOutput> allOutputs = new ArrayList<>();
        allOutputs.addAll(embedResults);
        allOutputs.addAll(fieldResults);

        for (AssemblyOutput output : allOutputs) {
            smBuilder.addSemanticResults(output.result());

            String sfaKey = output.result().getSourceFieldName()
                    + "::" + output.result().getChunkConfigId();

            sfaMap.computeIfAbsent(sfaKey, k -> {
                SourceFieldAnalytics.Builder sfaBuilder = SourceFieldAnalytics.newBuilder()
                        .setSourceField(output.result().getSourceFieldName())
                        .setChunkConfigId(output.result().getChunkConfigId())
                        .setTotalChunks(output.totalChunks());

                if (output.documentAnalytics() != null) {
                    sfaBuilder.setDocumentAnalytics(output.documentAnalytics());
                }

                // Compute chunk size stats
                int totalSize = 0;
                int minSize = Integer.MAX_VALUE;
                int maxSize = 0;
                for (SemanticChunk chunk : output.result().getChunksList()) {
                    int size = chunk.getEmbeddingInfo().getTextContent().length();
                    totalSize += size;
                    minSize = Math.min(minSize, size);
                    maxSize = Math.max(maxSize, size);
                }
                int count = output.result().getChunksCount();
                sfaBuilder.setAverageChunkSize(count > 0 ? (float) totalSize / count : 0)
                        .setMinChunkSize(minSize == Integer.MAX_VALUE ? 0 : minSize)
                        .setMaxChunkSize(maxSize);

                return sfaBuilder;
            });
        }

        for (SourceFieldAnalytics.Builder sfaBuilder : sfaMap.values()) {
            smBuilder.addSourceFieldAnalytics(sfaBuilder.build());
        }

        outputDocBuilder.setSearchMetadata(smBuilder.build());
        log.info("Phase 3: Semantic orchestration complete for doc: {}. Total semantic results: {}, source field analytics: {}",
                docId, smBuilder.getSemanticResultsCount(), sfaMap.size());
        return outputDocBuilder.build();
    }

    /**
     * Processes field-level targets -- embeds raw field text without chunking.
     */
    private Uni<List<AssemblyOutput>> processFieldLevelTargets(
            PipeDoc inputDoc, List<FieldLevelTarget> targets, String nodeId) {

        String docId = inputDoc.getDocId();
        String requestId = UUID.randomUUID().toString();

        List<Uni<AssemblyOutput>> embedderUnis = new ArrayList<>();

        for (FieldLevelTarget target : targets) {
            String resultSetName = resolveResultSetName(target.template(), target.sourceLabel(),
                    "field_level", target.embedderConfigId());

            StreamEmbeddingsRequest embReq = StreamEmbeddingsRequest.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(docId + "_" + target.sourceLabel() + "_full")
                    .setTextContent(target.sourceText())
                    .setChunkConfigId("field_level")
                    .setEmbeddingModelId(target.embedderConfigId())
                    .build();

            Multi<StreamEmbeddingsRequest> singleRequest = Multi.createFrom().item(embReq);

            embedderUnis.add(
                    embedderStreamClient.streamEmbeddings(singleRequest)
                            .collect().asList()
                            .map(responses -> {
                                String chunkId = docId + "_" + target.sourceLabel() + "_full";

                                StreamChunksResponse syntheticChunk = StreamChunksResponse.newBuilder()
                                        .setChunkId(chunkId)
                                        .setChunkNumber(0)
                                        .setTextContent(target.sourceText())
                                        .setChunkConfigId("field_level")
                                        .setSourceFieldName(target.sourceLabel())
                                        .setStartOffset(0)
                                        .setEndOffset(target.sourceText().length())
                                        .setIsLast(true)
                                        .build();

                                return assembleResult(
                                        List.of(syntheticChunk), responses,
                                        target.sourceLabel(), "field_level",
                                        target.embedderConfigId(), resultSetName, nodeId,
                                        null);
                            })
                            .onFailure().recoverWithItem(error -> {
                                log.error("Field-level embedder {} failed for doc {}: {}",
                                        target.embedderConfigId(), docId, error.getMessage());
                                return null;
                            })
            );
        }

        return Uni.combine().all().unis(embedderUnis)
                .with(results -> {
                    List<AssemblyOutput> nonNull = new ArrayList<>();
                    for (Object r : results) {
                        if (r != null) {
                            nonNull.add((AssemblyOutput) r);
                        }
                    }
                    return nonNull;
                });
    }

    // =========================================================================
    // Config-based orchestration (directives from module config)
    // =========================================================================

    private Uni<PipeDoc> orchestrateFromConfigDirectives(PipeDoc inputDoc,
                                                          SemanticManagerOptions options,
                                                          String nodeId) {
        VectorSetDirectives.Builder directivesBuilder = VectorSetDirectives.newBuilder();

        for (DirectiveConfig dc : options.directives()) {
            VectorDirective.Builder vd = VectorDirective.newBuilder()
                    .setSourceLabel(dc.sourceLabel() != null ? dc.sourceLabel() : "")
                    .setCelSelector(dc.celSelector() != null ? dc.celSelector() : "");

            if (dc.fieldNameTemplate() != null) {
                vd.setFieldNameTemplate(dc.fieldNameTemplate());
            }

            if (dc.chunkerConfigs() != null) {
                for (DirectiveConfig.NamedConfig cc : dc.chunkerConfigs()) {
                    NamedChunkerConfig.Builder ncb = NamedChunkerConfig.newBuilder()
                            .setConfigId(cc.configId() != null ? cc.configId() : "default");
                    if (cc.config() != null) {
                        try {
                            ObjectMapper mapper = new ObjectMapper();
                            String json = mapper.writeValueAsString(cc.config());
                            Struct.Builder sb = Struct.newBuilder();
                            JsonFormat.parser().ignoringUnknownFields().merge(json, sb);
                            ncb.setConfig(sb.build());
                        } catch (Exception e) {
                            log.warn("Failed to parse chunker config for '{}': {}", cc.configId(), e.getMessage());
                        }
                    }
                    vd.addChunkerConfigs(ncb.build());
                }
            }

            if (dc.embedderConfigs() != null) {
                for (DirectiveConfig.NamedConfig ec : dc.embedderConfigs()) {
                    NamedEmbedderConfig.Builder neb = NamedEmbedderConfig.newBuilder()
                            .setConfigId(ec.configId() != null ? ec.configId() : "default");
                    if (ec.config() != null) {
                        try {
                            ObjectMapper mapper = new ObjectMapper();
                            String json = mapper.writeValueAsString(ec.config());
                            Struct.Builder sb = Struct.newBuilder();
                            JsonFormat.parser().ignoringUnknownFields().merge(json, sb);
                            neb.setConfig(sb.build());
                        } catch (Exception e) {
                            log.warn("Failed to parse embedder config for '{}': {}", ec.configId(), e.getMessage());
                        }
                    }
                    vd.addEmbedderConfigs(neb.build());
                }
            }

            directivesBuilder.addDirectives(vd.build());
        }

        PipeDoc docWithDirectives = inputDoc.toBuilder()
                .setSearchMetadata(inputDoc.getSearchMetadata().toBuilder()
                        .setVectorSetDirectives(directivesBuilder.build())
                        .build())
                .build();

        return orchestrateFromDirectives(docWithDirectives, nodeId);
    }

    // =========================================================================
    // VectorSetService-based orchestration (fallback path)
    // =========================================================================

    private Uni<PipeDoc> orchestrateFromVectorSetService(PipeDoc inputDoc,
                                                          SemanticManagerOptions options,
                                                          String nodeId) {
        String docId = inputDoc.getDocId();

        return vectorSetResolver.resolveVectorSets(options.effectiveIndexName())
                .chain(vectorSets -> {
                    List<VectorSet> activeVectorSets = filterVectorSets(vectorSets, options.vectorSetIds());

                    if (activeVectorSets.isEmpty()) {
                        log.warn("No VectorSets found for index: {}. Returning doc unchanged.",
                                options.effectiveIndexName());
                        return Uni.createFrom().item(inputDoc);
                    }

                    log.info("Processing {} VectorSets for doc: {}", activeVectorSets.size(), docId);

                    // Convert VectorSets to directives and use the same 4-phase flow
                    VectorSetDirectives.Builder directivesBuilder = VectorSetDirectives.newBuilder();

                    // Group VectorSets by (source_field, chunker_config_id) for deduplication
                    Map<String, Map<String, List<VectorSet>>> grouped = new LinkedHashMap<>();
                    for (VectorSet vs : activeVectorSets) {
                        grouped.computeIfAbsent(vs.getSourceField(), k -> new LinkedHashMap<>())
                                .computeIfAbsent(vs.getChunkerConfigId(), k -> new ArrayList<>())
                                .add(vs);
                    }

                    // Build one directive per source field
                    for (Map.Entry<String, Map<String, List<VectorSet>>> sfEntry : grouped.entrySet()) {
                        String sourceField = sfEntry.getKey();

                        for (Map.Entry<String, List<VectorSet>> ccEntry : sfEntry.getValue().entrySet()) {
                            String chunkerConfigId = ccEntry.getKey();
                            List<VectorSet> vsGroup = ccEntry.getValue();

                            VectorDirective.Builder vd = VectorDirective.newBuilder()
                                    .setSourceLabel(sourceField)
                                    .setCelSelector("document.search_metadata." + sourceField)
                                    .addChunkerConfigs(NamedChunkerConfig.newBuilder()
                                            .setConfigId(chunkerConfigId)
                                            .build());

                            for (VectorSet vs : vsGroup) {
                                vd.addEmbedderConfigs(NamedEmbedderConfig.newBuilder()
                                        .setConfigId(vs.getEmbeddingModelConfigId())
                                        .build());
                            }

                            // Use the first VectorSet's result_set_name as template marker
                            // We'll override per-result later
                            directivesBuilder.addDirectives(vd.build());
                        }
                    }

                    PipeDoc docWithDirectives = inputDoc.toBuilder()
                            .setSearchMetadata(inputDoc.getSearchMetadata().toBuilder()
                                    .setVectorSetDirectives(directivesBuilder.build())
                                    .build())
                            .build();

                    return orchestrateFromDirectives(docWithDirectives, nodeId)
                            .map(enrichedDoc -> {
                                // Post-process: add VectorSet metadata to results
                                PipeDoc.Builder docBuilder = enrichedDoc.toBuilder();
                                SearchMetadata.Builder smb = enrichedDoc.getSearchMetadata().toBuilder();

                                // Clear and re-add results with VectorSet metadata
                                List<SemanticProcessingResult> origResults = new ArrayList<>(smb.getSemanticResultsList());
                                smb.clearSemanticResults();

                                for (SemanticProcessingResult spr : origResults) {
                                    // Find matching VectorSet
                                    Optional<VectorSet> matchingVs = activeVectorSets.stream()
                                            .filter(vs -> vs.getChunkerConfigId().equals(spr.getChunkConfigId())
                                                    && vs.getEmbeddingModelConfigId().equals(spr.getEmbeddingConfigId())
                                                    && vs.getSourceField().equals(spr.getSourceFieldName()))
                                            .findFirst();

                                    if (matchingVs.isPresent()) {
                                        VectorSet vs = matchingVs.get();
                                        SemanticProcessingResult enriched = spr.toBuilder()
                                                .setResultSetName(vs.getResultSetName())
                                                .putMetadata("vector_set_id", protoValue(vs.getId()))
                                                .putMetadata("vector_set_name", protoValue(vs.getName()))
                                                .build();
                                        smb.addSemanticResults(enriched);
                                    } else {
                                        smb.addSemanticResults(spr);
                                    }
                                }

                                docBuilder.setSearchMetadata(smb.build());
                                return docBuilder.build();
                            });
                });
    }

    // =========================================================================
    // Shared helpers
    // =========================================================================

    private String resolveResultSetName(String template, String sourceLabel,
                                         String chunkerConfigId, String embedderConfigId) {
        return template
                .replace("{source_label}", sourceLabel)
                .replace("{chunker_id}", chunkerConfigId)
                .replace("{embedder_id}", embedderConfigId);
    }

    private String extractTextByCelSelector(PipeDoc doc, String celSelector, String sourceLabel) {
        if (celSelector == null || celSelector.isEmpty()) {
            return extractSourceText(doc, sourceLabel);
        }

        String normalized = celSelector.toLowerCase().trim();
        if (normalized.startsWith("document.")) {
            normalized = normalized.substring("document.".length());
        }

        if (normalized.startsWith("search_metadata.")) {
            String field = normalized.substring("search_metadata.".length());
            return extractSourceText(doc, field);
        }

        return extractSourceText(doc, celSelector);
    }

    /**
     * Holds the assembled result plus document-level analytics.
     */
    record AssemblyOutput(
            SemanticProcessingResult result,
            DocumentAnalytics documentAnalytics,
            int totalChunks
    ) {}

    private AssemblyOutput assembleResult(
            List<StreamChunksResponse> chunks,
            List<StreamEmbeddingsResponse> embeddingResponses,
            String sourceFieldName,
            String chunkConfigId,
            String embeddingConfigId,
            String resultSetName,
            String nodeId,
            NlpDocumentAnalysis nlpAnalysis) {

        Map<String, StreamEmbeddingsResponse> embeddingMap = new HashMap<>();
        for (StreamEmbeddingsResponse resp : embeddingResponses) {
            if (resp.getSuccess()) {
                embeddingMap.put(resp.getChunkId(), resp);
            } else {
                log.warn("Embedding failed for chunk {}: {}", resp.getChunkId(), resp.getErrorMessage());
            }
        }

        SemanticProcessingResult.Builder resultBuilder = SemanticProcessingResult.newBuilder()
                .setResultId(UUID.randomUUID().toString())
                .setSourceFieldName(sourceFieldName)
                .setChunkConfigId(chunkConfigId)
                .setEmbeddingConfigId(embeddingConfigId)
                .setResultSetName(resultSetName);

        if (nodeId != null) {
            resultBuilder.putMetadata("coordinator_node_id", protoValue(nodeId));
        }

        // Set NLP analysis on the result if available
        if (nlpAnalysis != null) {
            resultBuilder.setNlpAnalysis(nlpAnalysis);
        }

        DocumentAnalytics documentAnalytics = null;
        int totalChunksReported = chunks.size();

        for (StreamChunksResponse chunk : chunks) {
            ChunkEmbedding.Builder embeddingInfoBuilder = ChunkEmbedding.newBuilder()
                    .setTextContent(chunk.getTextContent())
                    .setChunkId(chunk.getChunkId())
                    .setOriginalCharStartOffset(chunk.getStartOffset())
                    .setOriginalCharEndOffset(chunk.getEndOffset())
                    .setChunkConfigId(chunkConfigId);

            StreamEmbeddingsResponse embResp = embeddingMap.get(chunk.getChunkId());
            if (embResp != null) {
                embeddingInfoBuilder.addAllVector(embResp.getVectorList());
            }

            SemanticChunk.Builder chunkBuilder = SemanticChunk.newBuilder()
                    .setChunkId(chunk.getChunkId())
                    .setChunkNumber(chunk.getChunkNumber())
                    .setEmbeddingInfo(embeddingInfoBuilder.build())
                    .putAllMetadata(chunk.getMetadataMap());

            if (chunk.hasChunkAnalytics()) {
                chunkBuilder.setChunkAnalytics(chunk.getChunkAnalytics());
            }

            resultBuilder.addChunks(chunkBuilder.build());

            if (chunk.getIsLast()) {
                if (chunk.hasDocumentAnalytics()) {
                    documentAnalytics = chunk.getDocumentAnalytics();
                }
                if (chunk.hasTotalChunks()) {
                    totalChunksReported = chunk.getTotalChunks();
                }
            }
        }

        log.info("Assembled SemanticProcessingResult: resultSet={}, chunks={}, embeddings={}",
                resultSetName, chunks.size(), embeddingMap.size());

        return new AssemblyOutput(resultBuilder.build(), documentAnalytics, totalChunksReported);
    }

    private String extractSourceText(PipeDoc doc, String sourceField) {
        if (!doc.hasSearchMetadata()) {
            return null;
        }
        SearchMetadata sm = doc.getSearchMetadata();
        return switch (sourceField.toLowerCase()) {
            case "body" -> sm.hasBody() ? sm.getBody() : null;
            case "title" -> sm.hasTitle() ? sm.getTitle() : null;
            default -> {
                log.warn("Unsupported source field: {}", sourceField);
                yield null;
            }
        };
    }

    private List<VectorSet> filterVectorSets(List<VectorSet> vectorSets, List<String> vectorSetIds) {
        if (vectorSetIds == null || vectorSetIds.isEmpty()) {
            return vectorSets;
        }
        Set<String> idSet = new HashSet<>(vectorSetIds);
        return vectorSets.stream()
                .filter(vs -> idSet.contains(vs.getId()))
                .collect(Collectors.toList());
    }

    private static com.google.protobuf.Value protoValue(String s) {
        return com.google.protobuf.Value.newBuilder().setStringValue(s).build();
    }

    // =========================================================================
    // Internal records
    // =========================================================================

    /** Work for a single source text — all chunk configs that apply to it. */
    record SourceTextWork(
            String sourceText,
            String sourceLabel,
            Map<String, ChunkConfigWork> chunkConfigs
    ) {}

    /** A single chunk config with its embedder targets. */
    record ChunkConfigWork(
            String chunkConfigId,
            Struct chunkerConfig,
            List<EmbedderTarget> embedderTargets
    ) {}

    /** Result of chunking a source text — chunks grouped by config_id. */
    record SourceTextChunkResult(
            Map<String, List<StreamChunksResponse>> chunksByConfigId,
            NlpDocumentAnalysis nlpAnalysis
    ) {}

    record EmbedderTarget(String embedderConfigId, Struct embedderConfig, String template) {}

    record FieldLevelTarget(
            String sourceText,
            String sourceLabel,
            String embedderConfigId,
            Struct embedderConfig,
            String template
    ) {}
}

package ai.pipestream.module.semanticmanager.mock;

import ai.pipestream.semantic.v1.SemanticChunkerService;
import ai.pipestream.semantic.v1.StreamChunksRequest;
import ai.pipestream.semantic.v1.StreamChunksResponse;
import io.quarkus.grpc.GrpcService;
import io.smallrye.mutiny.Multi;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * In-process mock chunker that splits text into simple word-based chunks.
 * Each chunk is ~10 words. No ML models needed.
 */
@Singleton
@GrpcService
public class MockChunkerService implements SemanticChunkerService {

    private static final Logger log = LoggerFactory.getLogger(MockChunkerService.class);
    private static final int WORDS_PER_CHUNK = 10;

    @Override
    public Multi<StreamChunksResponse> streamChunks(StreamChunksRequest request) {
        String text = request.getTextContent();
        String docId = request.getDocId();
        String configId = request.getChunkConfigId();
        String sourceField = request.getSourceFieldName();
        String requestId = request.getRequestId();

        log.info("MockChunker: chunking doc={}, configId={}, sourceField={}, textLen={}",
                docId, configId, sourceField, text.length());

        List<StreamChunksResponse> chunks = splitIntoChunks(text, requestId, docId, configId, sourceField);

        log.info("MockChunker: produced {} chunks for doc={}", chunks.size(), docId);
        return Multi.createFrom().iterable(chunks);
    }

    private List<StreamChunksResponse> splitIntoChunks(
            String text, String requestId, String docId, String configId, String sourceField) {

        List<StreamChunksResponse> chunks = new ArrayList<>();
        String[] words = text.split("\\s+");

        int chunkNumber = 0;
        int charOffset = 0;

        for (int i = 0; i < words.length; i += WORDS_PER_CHUNK) {
            int end = Math.min(i + WORDS_PER_CHUNK, words.length);
            StringBuilder chunkText = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunkText.append(" ");
                chunkText.append(words[j]);
            }

            String content = chunkText.toString();
            int startOffset = charOffset;
            int endOffset = startOffset + content.length();
            boolean isLast = (end >= words.length);

            StreamChunksResponse chunk = StreamChunksResponse.newBuilder()
                    .setRequestId(requestId)
                    .setDocId(docId)
                    .setChunkId(UUID.randomUUID().toString())
                    .setChunkNumber(chunkNumber)
                    .setTextContent(content)
                    .setStartOffset(startOffset)
                    .setEndOffset(endOffset)
                    .setChunkConfigId(configId)
                    .setSourceFieldName(sourceField)
                    .setIsLast(isLast)
                    .build();

            chunks.add(chunk);
            chunkNumber++;
            charOffset = endOffset + 1; // +1 for space between chunks
        }

        if (chunks.isEmpty()) {
            // Empty text → single empty chunk
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
}

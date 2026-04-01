package ai.pipestream.module.semanticmanager.service;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes centroid (averaged) vectors at various granularities:
 * paragraph, section, and document level.
 * <p>
 * Pure computation — no I/O, no CDI dependencies.
 * All centroids are L2 normalized after averaging.
 */
public final class CentroidComputer {

    private CentroidComputer() {}

    /**
     * Result of a centroid computation.
     */
    public record CentroidResult(
            float[] vector,
            String text,
            int sourceVectorCount,
            String sectionTitle,
            Integer sectionDepth
    ) {}

    /**
     * Averages vectors and L2-normalizes the result.
     * Returns empty array if input is empty.
     */
    public static float[] averageAndNormalize(List<float[]> vectors) {
        if (vectors.isEmpty()) {
            return new float[0];
        }

        int dim = vectors.get(0).length;
        float[] sum = new float[dim];
        for (float[] v : vectors) {
            for (int i = 0; i < dim; i++) {
                sum[i] += v[i];
            }
        }

        float count = vectors.size();
        for (int i = 0; i < dim; i++) {
            sum[i] /= count;
        }

        return l2Normalize(sum);
    }

    /**
     * L2-normalizes a vector in place and returns it.
     */
    public static float[] l2Normalize(float[] v) {
        float norm = 0f;
        for (float f : v) {
            norm += f * f;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
        }
        return v;
    }

    /**
     * Detects paragraph boundaries in the original text by finding double-newline gaps
     * between consecutive sentences.
     *
     * @param originalText    the source text
     * @param sentenceOffsets [i] = {startOffset, endOffset} for sentence i
     * @return list of paragraphs, each being a list of sentence indices
     */
    public static List<List<Integer>> detectParagraphBoundaries(String originalText, int[][] sentenceOffsets) {
        List<List<Integer>> paragraphs = new ArrayList<>();
        List<Integer> currentParagraph = new ArrayList<>();

        for (int i = 0; i < sentenceOffsets.length; i++) {
            currentParagraph.add(i);

            if (i < sentenceOffsets.length - 1) {
                int gapStart = Math.min(sentenceOffsets[i][1], originalText.length());
                int gapEnd = Math.min(sentenceOffsets[i + 1][0], originalText.length());
                if (gapStart < gapEnd) {
                    String gap = originalText.substring(gapStart, gapEnd);
                    if (gap.contains("\n\n") || gap.contains("\r\n\r\n")) {
                        paragraphs.add(currentParagraph);
                        currentParagraph = new ArrayList<>();
                    }
                }
            }
        }

        if (!currentParagraph.isEmpty()) {
            paragraphs.add(currentParagraph);
        }

        return paragraphs;
    }

    /**
     * Computes paragraph centroids by averaging sentence vectors within each paragraph.
     */
    public static List<CentroidResult> computeParagraphCentroids(
            List<float[]> sentenceVectors,
            List<String> sentenceTexts,
            String originalText,
            int[][] sentenceOffsets) {

        List<List<Integer>> paragraphs = detectParagraphBoundaries(originalText, sentenceOffsets);
        List<CentroidResult> centroids = new ArrayList<>();

        for (List<Integer> paragraphIndices : paragraphs) {
            List<float[]> vecs = new ArrayList<>();
            StringBuilder text = new StringBuilder();
            for (int idx : paragraphIndices) {
                if (idx < sentenceVectors.size()) {
                    vecs.add(sentenceVectors.get(idx));
                }
                if (idx < sentenceTexts.size()) {
                    if (!text.isEmpty()) text.append(" ");
                    text.append(sentenceTexts.get(idx));
                }
            }
            if (!vecs.isEmpty()) {
                centroids.add(new CentroidResult(
                        averageAndNormalize(vecs),
                        text.toString(),
                        vecs.size(),
                        null, null));
            }
        }
        return centroids;
    }

    /**
     * Computes section centroids by grouping sentences into sections based on
     * Section char_start_offset/char_end_offset from DocOutline.
     * Each section's centroid is the average of its constituent sentence vectors.
     *
     * @param sentenceVectors  embedding vectors for each sentence
     * @param sentenceTexts    text of each sentence
     * @param sentenceOffsets  [i] = {startOffset, endOffset} for sentence i
     * @param sections         sections from DocOutline, must have char_start_offset populated
     * @return one CentroidResult per section that contains at least one sentence
     */
    public static List<CentroidResult> computeSectionCentroids(
            List<float[]> sentenceVectors,
            List<String> sentenceTexts,
            int[][] sentenceOffsets,
            List<SectionInfo> sections) {

        if (sections.isEmpty() || sentenceVectors.isEmpty()) {
            return List.of();
        }

        // Sort sections by char_start_offset
        List<SectionInfo> sorted = sections.stream()
                .filter(s -> s.charStartOffset >= 0)
                .sorted(java.util.Comparator.comparingInt(s -> s.charStartOffset))
                .toList();

        if (sorted.isEmpty()) {
            return List.of();
        }

        List<CentroidResult> centroids = new ArrayList<>();

        for (int s = 0; s < sorted.size(); s++) {
            SectionInfo section = sorted.get(s);
            int sectionStart = section.charStartOffset;
            int sectionEnd = (s + 1 < sorted.size())
                    ? sorted.get(s + 1).charStartOffset
                    : (section.charEndOffset > sectionStart ? section.charEndOffset : Integer.MAX_VALUE);

            // Find sentences whose start offset falls within this section
            List<float[]> vecs = new ArrayList<>();
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < sentenceOffsets.length; i++) {
                int sentStart = sentenceOffsets[i][0];
                if (sentStart >= sectionStart && sentStart < sectionEnd) {
                    if (i < sentenceVectors.size()) vecs.add(sentenceVectors.get(i));
                    if (i < sentenceTexts.size()) {
                        if (!text.isEmpty()) text.append(" ");
                        text.append(sentenceTexts.get(i));
                    }
                }
            }

            if (!vecs.isEmpty()) {
                centroids.add(new CentroidResult(
                        averageAndNormalize(vecs),
                        text.toString(),
                        vecs.size(),
                        section.title,
                        section.depth));
            }
        }

        return centroids;
    }

    /**
     * Lightweight section info extracted from DocOutline Section proto.
     * Avoids proto dependency in the pure computation class.
     */
    public record SectionInfo(
            String title,
            int depth,
            int charStartOffset,
            int charEndOffset
    ) {}

    /**
     * Computes a single document centroid by averaging all sentence vectors.
     */
    public static CentroidResult computeDocumentCentroid(
            List<float[]> sentenceVectors,
            String fullText) {

        return new CentroidResult(
                averageAndNormalize(sentenceVectors),
                fullText,
                sentenceVectors.size(),
                null, null);
    }
}

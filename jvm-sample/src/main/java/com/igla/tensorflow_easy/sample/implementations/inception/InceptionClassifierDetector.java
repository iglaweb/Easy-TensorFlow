package com.igla.tensorflow_easy.sample.implementations.inception;

import com.igla.tensorflow_easy.classifier.models.ClassifyRecognition;
import com.igla.tensorflow_easy.classifier.models.ClassifyResult;
import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.DetectorBase;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class InceptionClassifierDetector<T> extends
        DetectorBase<T, ClassifyRecognition, ClassifyResult> {

    private final List<ClassifyRecognition> emptyRecognitions = Collections.emptyList();

    public InceptionClassifierDetector(Config<T> config) throws IOException {
        super(config);
    }

    @Override
    public CustomGraphProcessor<ClassifyResult> createGraphProcessor() {
        return new InceptionCustomGraphProcessor(this.inputModel);
    }

    @Override
    public List<ClassifyRecognition> processDetections(ClassifyResult detection, int width, int height) {

        float[] labelProbabilities = detection.getLabelProbabilities();
        int numDetections = detection.getNum_detections();
        if (numDetections == 0) return emptyRecognitions;

        int bestLabelIdx = maxIndex(labelProbabilities);
        String label = labels != null && bestLabelIdx < labels.size() ? labels.get(bestLabelIdx) : null;
        float probability = labelProbabilities[bestLabelIdx];
        final List<ClassifyRecognition> objectRecognitions = new ArrayList<>();
        ClassifyRecognition classifyRecognition = new ClassifyRecognition(0, label, probability);
        objectRecognitions.add(classifyRecognition);
        return objectRecognitions;
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    @Override
    public void close() {
        super.close();
    }
}

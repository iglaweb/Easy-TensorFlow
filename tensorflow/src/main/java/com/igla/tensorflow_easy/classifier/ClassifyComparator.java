package com.igla.tensorflow_easy.classifier;

import com.igla.tensorflow_easy.classifier.models.ClassifyRecognition;

import java.util.Comparator;

/**
 * Used to make sure the detections with highest confidence, is placed highest in queue.
 */
public class ClassifyComparator implements Comparator<ClassifyRecognition> {
    @Override
    public int compare(final ClassifyRecognition objectRecognitionA, final ClassifyRecognition objectRecognitionB) {
        return Float.compare(objectRecognitionB.getConfidence(), objectRecognitionA.getConfidence());
    }
}
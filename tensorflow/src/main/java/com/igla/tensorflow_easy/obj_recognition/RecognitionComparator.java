package com.igla.tensorflow_easy.obj_recognition;

import com.igla.tensorflow_easy.models.ObjectRecognition;

import java.util.Comparator;

/**
 * Used to make sure the detections with highest confidence, is placed highest in queue.
 */
public class RecognitionComparator implements Comparator<ObjectRecognition> {
    @Override
    public int compare(final ObjectRecognition objectRecognitionA, final ObjectRecognition objectRecognitionB) {
        return Float.compare(objectRecognitionB.getConfidence(), objectRecognitionA.getConfidence());
    }
}
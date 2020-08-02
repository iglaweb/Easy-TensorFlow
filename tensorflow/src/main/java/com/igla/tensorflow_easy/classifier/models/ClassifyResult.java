package com.igla.tensorflow_easy.classifier.models;

public class ClassifyResult {

    private int num_detections;
    private float[] labelProbabilities;

    public ClassifyResult(int num_detections, float[] labelProbabilities) {
        this.num_detections = num_detections;
        this.labelProbabilities = labelProbabilities;
    }

    public int getNum_detections() {
        return num_detections;
    }

    public float[] getLabelProbabilities() {
        return labelProbabilities;
    }
}

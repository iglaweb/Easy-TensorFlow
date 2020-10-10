package com.igla.tensorflow_easy.sample.implementations.faceboxes;

import com.igla.tensorflow_easy.core.DetectorBase;
import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.obj_recognition.ObjectDetector;
import com.igla.tensorflow_easy.models.Detection;
import com.igla.tensorflow_easy.models.ObjectRecognition;
import com.igla.tensorflow_easy.models.RectFloats;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import com.igla.tensorflow_easy.obj_recognition.RecognitionComparator;

import java.io.IOException;
import java.util.*;

public class FaceboxesObjectDetector<T> extends DetectorBase<T, ObjectRecognition, Detection>
        implements ObjectDetector<T> {

    private final List<ObjectRecognition> emptyRecognitions = Collections.emptyList();

    // Find the best detections.
    private final Queue<ObjectRecognition> priorityQueue;

    public FaceboxesObjectDetector(Config<T> config) throws IOException {
        super(config);
        Comparator<ObjectRecognition> recognitionComparator = new RecognitionComparator();
        this.priorityQueue =
                new PriorityQueue<>(1, recognitionComparator);
    }

    @Override
    public CustomGraphProcessor<Detection> createGraphProcessor() {
        return new FaceboxesCustomGraphProcessor(this.inputModel);
    }

    @Override
    public List<ObjectRecognition> processDetections(Detection detection, int width, int height) {
        // clear before filling
        priorityQueue.clear();

        float[][] detection_boxes = detection.getDetection_boxes();
        float[] detection_scores = detection.getDetection_scores();
        float[] detection_classes = detection.getDetection_classes();

        // Scale them back to the input size.
        for (int i = 0; i < detection_scores.length; ++i) {
            float score = detection_scores[i];
            if (score < thresholdValue) continue;

            float left = detection_boxes[i][1] * width;
            float top = detection_boxes[i][0] * height;
            float right = detection_boxes[i][3] * width;
            float bottom = detection_boxes[i][2] * height;
            final RectFloats rectDetection =
                    new RectFloats(
                            left,
                            top,
                            right - left,
                            bottom - top
                    );

            String label = labels == null || labels.isEmpty() ?
                    "" : labels.get(((int) detection_classes[i]) - 1);
            priorityQueue.add(
                    new ObjectRecognition(
                            i,
                            label,
                            score,
                            rectDetection
                    )
            );
        }

        if (priorityQueue.isEmpty()) return emptyRecognitions;
        final List<ObjectRecognition> objectRecognitions = new ArrayList<>();
        int maxObjects = Math.min(priorityQueue.size(), config.getMaxPriorityObjects());
        for (int i = 0; i < maxObjects; ++i) {
            objectRecognitions.add(priorityQueue.poll());
        }
        return objectRecognitions;
    }

    @Override
    public void close() {
        super.close();
    }
}

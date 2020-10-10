package com.igla.tensorflow_easy.sample.implementations.yolo;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.DetectorBase;
import com.igla.tensorflow_easy.models.Detection;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import com.igla.tensorflow_easy.sample.implementations.yolo.models.Recognition;

import java.io.IOException;
import java.util.List;

public class Yolov2ObjectDetector<T> extends
        DetectorBase<T, Recognition, Detection> {

    public Yolov2ObjectDetector(Config<T> config) throws IOException {
        super(config);
    }

    @Override
    public CustomGraphProcessor<Detection> createGraphProcessor() {
        return new YoloCustomGraphProcessor(this.inputModel);
    }

    @Override
    public List<Recognition> processDetections(Detection detection, int width, int height) {
        return YOLOClassifier.getInstance().classifyImage(
                detection.getDetection_scores(),
                labels
        );
    }

    @Override
    public void close() {
        super.close();
    }
}

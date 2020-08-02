package com.igla.tensorflow_easy.obj_recognition;

import com.igla.tensorflow_easy.models.ObjectRecognition;

import java.util.List;

public interface ObjectDetector<T> extends AutoCloseable {

    /**
     * Returns all detected objects on the input image, that the trained model can find and recognize.
     *
     * @param image The image to perform the objectdetection on.
     * @return All found detections in the input image.
     */
    List<ObjectRecognition> classifyImage(T image);
}

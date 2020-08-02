package com.igla.tensorflow_easy.core;

import java.util.List;

public interface BaseProcessImage<T, R> extends AutoCloseable {

    /**
     * Returns all detected objects on the input image, that the trained model can find and recognize.
     *
     * @param image The image to perform the objectdetection on.
     * @return All found detections in the input image.
     */
    List<R> classifyImage(T image);
}

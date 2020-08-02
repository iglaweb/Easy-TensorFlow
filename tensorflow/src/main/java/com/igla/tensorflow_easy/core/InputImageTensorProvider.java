package com.igla.tensorflow_easy.core;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

public interface InputImageTensorProvider<T> extends AutoCloseable {
    Tensor<? extends Number> getTensor(T image);

    int getImageHeight(T image);

    int getImageWidth(T image);
}

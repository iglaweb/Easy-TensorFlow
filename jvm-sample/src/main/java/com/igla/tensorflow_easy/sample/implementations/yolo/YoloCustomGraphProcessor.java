package com.igla.tensorflow_easy.sample.implementations.yolo;

import com.igla.tensorflow_easy.core.InputModel;
import com.igla.tensorflow_easy.models.Detection;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;

public class YoloCustomGraphProcessor extends CustomGraphProcessor<Detection> {

    YoloCustomGraphProcessor(@NotNull InputModel graph) {
        super(graph);
    }

    @Override
    public String[] resultOperationNames() {
        return new String[]{"output"};
    }

    @Override
    public Detection detections() {
        Tensor<?> detection_output = getTensor("output");
        if (detection_output == null) return new Detection(0, new float[0][4], new float[0], new float[0]);

        Tensor<Float> t = detection_output.expect(Float.class);

        float[] outputTensor = new float[YOLOClassifier.getInstance().getOutputSizeByShape(t)];
        FloatBuffer floatBuffer = FloatBuffer.wrap(outputTensor);
        detection_output.writeTo(floatBuffer);

        return new Detection(
                outputTensor.length,
                new float[0][0],
                outputTensor,
                new float[0]
        );
    }
}

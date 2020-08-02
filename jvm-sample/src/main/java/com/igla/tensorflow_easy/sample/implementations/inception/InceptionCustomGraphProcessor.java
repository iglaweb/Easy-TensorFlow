package com.igla.tensorflow_easy.sample.implementations.inception;

import com.igla.tensorflow_easy.classifier.models.ClassifyResult;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

import java.util.Arrays;

public class InceptionCustomGraphProcessor extends CustomGraphProcessor<ClassifyResult> {

    private int tempTensorObjectSize = 0;

    InceptionCustomGraphProcessor(@NotNull Graph graph) {
        super(graph);
    }

    @Override
    public String[] resultOperationNames() {
        return new String[]{"output"};
    }

    @Override
    public ClassifyResult detections() {
        Tensor<?> detection_output = getTensor("output");
        if (detection_output == null) {
            return new ClassifyResult(0, new float[0]);
        }

        final long[] rshape = detection_output.shape();
        if (detection_output.numDimensions() != 2 || rshape[0] != 1) {
            throw new RuntimeException(
                    String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
        }

        int maxObjects = (int) rshape[1];
        float[] output = detection_output.copyTo(new float[1][maxObjects])[0];

        this.tempTensorObjectSize = maxObjects;
        return new ClassifyResult(
                maxObjects,
                output
        );
    }
}

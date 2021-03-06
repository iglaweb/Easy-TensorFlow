package com.igla.tensorflow_easy.sample.implementations.faceboxes;

import com.igla.tensorflow_easy.core.InputModel;
import com.igla.tensorflow_easy.models.Detection;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.tensorflow.Tensor;

import java.util.Arrays;

public class FaceboxesCustomGraphProcessor extends CustomGraphProcessor<Detection> {

    @Nullable
    private float[][] temp_detection_scores = null;
    @Nullable
    private float[][][] temp_detection_boxes_float = null;
    private int tempTensorObjectSize = 0;

    FaceboxesCustomGraphProcessor(@NotNull InputModel inputModel) {
        super(inputModel);
    }

    @Override
    public String[] resultOperationNames() {
        return new String[]{"boxes", "scores", "num_boxes"};
    }

    @Override
    public Detection detections() {
        Tensor<?> detection_scores = getTensor("scores");
        Tensor<?> detection_boxes = getTensor("boxes");
        if (detection_scores == null || detection_boxes == null) {
            return new Detection(0, new float[0][4], new float[0], new float[0]);
        }

        final long[] rshape = detection_scores.shape();
        if (detection_scores.numDimensions() != 2 || rshape[0] != 1) {
            throw new RuntimeException(
                    String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
        }

        int maxObjects = (int) rshape[1];
        if (temp_detection_scores == null || tempTensorObjectSize != maxObjects) {
            temp_detection_scores = new float[1][maxObjects];
        }
        float[] detection_scores_floats = detection_scores.copyTo(temp_detection_scores)[0];

        if (temp_detection_boxes_float == null || tempTensorObjectSize != maxObjects) {
            temp_detection_boxes_float = new float[1][maxObjects][4];
        }
        float[][] detection_boxes_float = detection_boxes.copyTo(temp_detection_boxes_float)[0];

        this.tempTensorObjectSize = maxObjects;
        return new Detection(
                maxObjects,
                detection_boxes_float,
                detection_scores_floats,
                new float[0]
        );
    }
}

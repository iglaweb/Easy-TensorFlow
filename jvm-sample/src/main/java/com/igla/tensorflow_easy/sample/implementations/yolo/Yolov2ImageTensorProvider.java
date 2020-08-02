package com.igla.tensorflow_easy.sample.implementations.yolo;

import com.igla.tensorflow_easy.core.GraphBuilder;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.utils.TensorFlowUtils;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class Yolov2ImageTensorProvider implements InputImageTensorProvider<byte[]> {

    private static final int SIZE = 416;
    private static final float MEAN = 255f;

    private final Graph graph;
    private final Output graphOutput;

    public Yolov2ImageTensorProvider() {
        this.graph = new Graph();
        GraphBuilder b = new GraphBuilder(graph);
        final Output<String> input = b.placeholder("input", DataType.STRING);
        graphOutput =
                b.div(
                        b.resizeBilinear( // Resize using bilinear interpolation
                                b.expandDims( // Increase the output tensors dimension
                                        b.cast(b.decodeJpeg(input, 3), Float.class), // Cast the output to Float
                                        b.constant("make_batch", 0)),
                                b.constant("size", new int[]{SIZE, SIZE})),
                        b.constant("scale", MEAN));
    }

    @Override
    public Tensor<Float> getTensor(byte[] image) {
        return TensorFlowUtils.executeImageOnGraph(image, "input", graph, graphOutput);
    }

    @Override
    public int getImageHeight(byte[] image) {
        return 0;
    }

    @Override
    public int getImageWidth(byte[] image) {
        return 0;
    }

    @Override
    public void close() {
        if (graph != null) {
            graph.close();
        }
    }
}

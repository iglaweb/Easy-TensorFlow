package com.igla.tensorflow_easy.sample.implementations.inception;

import com.igla.tensorflow_easy.core.GraphBuilder;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.utils.TensorFlowUtils;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class InceptionImageTensorProvider implements InputImageTensorProvider<byte[]> {

    private static final int H = 224;
    private static final int W = 224;
    private static final float mean = 117f;
    private static final float scale = 1f;

    private final Graph graph;
    private final Output graphOutput;

    public InceptionImageTensorProvider() {
        this.graph = new Graph();
        GraphBuilder b = new GraphBuilder(graph);
        // Some constants specific to the pre-trained model at:
        // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
        //
        // - The model was trained with images scaled to 224x224 pixels.
        // - The colors, represented as R, G, B in 1-byte each were converted to
        //   float using (value - Mean)/Scale.
        final Output<String> input = b.placeholder("input", DataType.STRING);
        graphOutput =
                b.div(
                        b.sub(
                                b.resizeBilinear(
                                        b.expandDims(
                                                b.cast(b.decodeJpeg(input, 3), Float.class),
                                                b.constant("make_batch", 0)),
                                        b.constant("size", new int[]{H, W})),
                                b.constant("mean", mean)),
                        b.constant("scale", scale));
    }

    @Override
    public Tensor<Float> getTensor(byte[] image) {
        return TensorFlowUtils.executeImageOnGraph(image,"input", graph, graphOutput);
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

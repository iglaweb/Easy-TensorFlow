package com.igla.tensorflow_easy.utils;

import org.jetbrains.annotations.NotNull;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public final class TensorFlowUtils {

    /**
     * Pre-process input. It resize the image and normalize its pixels
     *
     * @param image Input image
     * @return Tensor&lt;Float&gt; with shape [1][416][416][3]
     */
    public static Tensor<Float> executeImageOnGraph(byte[] image, String inputOp, Graph graph, Output graphOutput) {
        try (Session s = new Session(graph)) {
            Tensor inputTensor = Tensor.create(image);
            // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
            List<Tensor<?>> tensorList = s.runner()
                    .feed(inputOp, inputTensor)
                    .fetch(graphOutput.op().name()).run();
            Tensor<Float> imageTensor = tensorList.get(0).expect(Float.class);
            for (int i = 1; i < tensorList.size(); i++) {
                tensorList.get(i).close();
            }
            inputTensor.close();
            return imageTensor;
        }
    }

    public static Graph importGraph(byte[] graphBytes) throws IllegalArgumentException {
        Graph graph = new Graph();
        graph.importGraphDef(graphBytes);
        return graph;
    }

    public static Graph readGraphFile(@NotNull File graphFile) throws IOException {
        System.out.println("Loading TensorFlow graph: " + graphFile.getAbsolutePath());
        long start = System.currentTimeMillis();
        try (InputStream graphInputStream = Files.newInputStream(graphFile.toPath())) {
            return setup(graphInputStream);
        } finally {
            long timeDiff = System.currentTimeMillis() - start;
            System.out.println("TensorFlow graph loaded in " + timeDiff + " ms");
        }
    }

    /**
     * Find object classification in images with a pre-trained graph model, and a ordered list of the possible labels.
     *
     * @param graphFile The graph model stream to load the pre-trained graph from.
     * @return
     * @throws IOException if an I/O error occurs.
     */
    private static Graph setup(InputStream graphFile) throws IOException {
        byte[] graphBytes = IoUtils.loadGraph(graphFile);
        return importGraph(graphBytes);
    }

    /**
     * Read label names from label file.
     *
     * @param labelFile the label file to read
     * @throws IOException
     */
    public static List<String> loadLabels(File labelFile) throws IOException {
        List<String> labels = new ArrayList<>(2);
        List<String> fileLines = Files.readAllLines(labelFile.toPath());
        for (String line : fileLines) {
            if (line.contains("name:")) {
                int i = line.indexOf("'");
                String substring = line.substring(i + 1, line.length() - 1);
                labels.add(substring);
            }
        }
        return labels;
    }
}

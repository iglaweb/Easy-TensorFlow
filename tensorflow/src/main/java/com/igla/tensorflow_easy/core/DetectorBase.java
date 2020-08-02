package com.igla.tensorflow_easy.core;

import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import com.igla.tensorflow_easy.utils.TensorFlowUtils;
import com.igla.tensorflow_easy.utils.Timber;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.util.List;

public abstract class DetectorBase<T, Result, PreResult> implements BaseProcessImage<T, Result> {

    protected Graph graph;

    @Nullable
    protected List<String> labels;

    @NotNull
    private final CustomGraphProcessor<PreResult> classifier;

    protected final float thresholdValue;

    @NotNull
    private InputImageTensorProvider<T> inputImageTensorProvider;

    protected final Config<T> config;

    public DetectorBase(Config<T> config) throws IOException {
        this.graph = readGraph(config);
        if (this.graph == null) {
            throw new IllegalArgumentException("No graph data provided");
        }

        Config.LabelsFile labelsFile = config.getLabelsFile();
        if (labelsFile != null) {
            if (labelsFile.getFile() != null) {
                this.labels = TensorFlowUtils.loadLabels(labelsFile.getFile());
            } else if (labelsFile.getLabels() != null) {
                this.labels = labelsFile.getLabels();
            }
        }

        this.config = config;
        this.thresholdValue = config.getThresholdValue();
        this.inputImageTensorProvider = config.getInputImageTensorProvider();
        this.classifier = createGraphProcessor();
    }

    private Graph readGraph(Config<T> config) throws IOException {
        Config.GraphFile graphFile = config.getGraphFile();
        if (graphFile.getFile() != null) {
            return TensorFlowUtils.readGraphFile(graphFile.getFile());
        } else if (graphFile.getGraphFile() != null) {
            return setup(graphFile.getGraphFile());
        }
        return null;
    }

    /**
     * Executes graph on the given preprocessed image
     *
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private PreResult executeGraph(final Tensor<?> image) {
        try {
            classifier.feed(config.getFeedInputTensorName(), image);
            classifier.run();
            return classifier.detections();
        } finally {
            classifier.closeOutputTensors();
        }
    }

    protected abstract CustomGraphProcessor<PreResult> createGraphProcessor();

    protected abstract List<Result> processDetections(PreResult detection, int width, int height);

    private Graph setup(byte[] graphBytes) {
        long start = System.currentTimeMillis();
        Timber.i("Loading TensorFlow graph...");
        Graph graph = TensorFlowUtils.importGraph(graphBytes);
        long timeDiff = System.currentTimeMillis() - start;
        System.out.println("TensorFlow graph loaded in " + timeDiff + " ms");
        return graph;
    }

    public void close() {
        try {
            this.classifier.close();
            this.graph.close();
            this.inputImageTensorProvider.close();
        } catch (Exception e) {
            Timber.e(e);
        }
    }

    @Override
    public List<Result> classifyImage(T image) {
        int width = inputImageTensorProvider.getImageWidth(image);
        int height = inputImageTensorProvider.getImageHeight(image);
        Tensor<? extends Number> imageTensor = inputImageTensorProvider.getTensor(image);

        PreResult detection = executeGraph(imageTensor);
        List<Result> objectRecognitions = processDetections(detection, width, height);
        imageTensor.close();
        return objectRecognitions;
    }
}

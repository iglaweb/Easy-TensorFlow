package com.igla.tensorflow_easy.core;

import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import com.igla.tensorflow_easy.utils.IoUtils;
import com.igla.tensorflow_easy.utils.TensorFlowUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.util.List;

public abstract class DetectorBase<T, Result, PreResult> implements BaseProcessImage<T, Result> {

    protected InputModel inputModel;

    @Nullable
    protected List<String> labels;

    @NotNull
    private final CustomGraphProcessor<PreResult> classifier;

    protected final float thresholdValue;

    @NotNull
    private final InputImageTensorProvider<T> inputImageTensorProvider;

    protected final Config<T> config;

    public DetectorBase(Config<T> config) throws IOException {
        Config.GraphFile graphFile = config.getGraphFile();

        this.inputModel = InputModel.createModelObj(graphFile);

        Config.LabelsFile labelsFile = config.getLabelsFile();
        this.labels = readLabels(labelsFile);

        this.config = config;
        this.thresholdValue = config.getThresholdValue();
        this.inputImageTensorProvider = config.getInputImageTensorProvider();
        this.classifier = createGraphProcessor();
    }

    private List<String> readLabels(Config.LabelsFile labelsFile) throws IOException {
        if (labelsFile != null) {
            if (labelsFile.getFile() != null) {
                return TensorFlowUtils.loadLabels(labelsFile.getFile());
            } else if (labelsFile.getLabels() != null) {
                return labelsFile.getLabels();
            }
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

    public void close() {
        IoUtils.closeQuietly(this.classifier);
        IoUtils.closeQuietly(this.inputModel);
        IoUtils.closeQuietly(this.inputImageTensorProvider);
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

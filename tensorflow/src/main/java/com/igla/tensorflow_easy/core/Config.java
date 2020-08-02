package com.igla.tensorflow_easy.core;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.File;
import java.util.List;

public class Config<T> {
    private final GraphFile graphFile;
    private final LabelsFile labelsFile;
    private final float thresholdValue;
    private final InputImageTensorProvider<T> inputImageTensorProvider;
    private final String feedInputTensorName;
    private final int maxPriorityObjects;

    public static class FileBase {
        private File file;

        FileBase() {
        }

        FileBase(File file) {
            this.file = file;
        }

        public static FileBase create(File file) {
            return new FileBase(file);
        }

        public File getFile() {
            return file;
        }
    }

    public static final class LabelsFile extends FileBase {
        private List<String> labels;

        private LabelsFile(List<String> labels) {
            this.labels = labels;
        }

        private LabelsFile(File file) {
            super(file);
        }

        public static LabelsFile create(File file) {
            return new LabelsFile(file);
        }

        public static LabelsFile create(List<String> file) {
            return new LabelsFile(file);
        }

        public List<String> getLabels() {
            return labels;
        }
    }

    public static final class GraphFile extends FileBase {
        private byte[] graphFile;

        private GraphFile(File file) {
            super(file);
        }

        private GraphFile(byte[] labels) {
            this.graphFile = labels;
        }

        public static GraphFile create(File file) {
            return new GraphFile(file);
        }

        public static GraphFile create(byte[] file) {
            return new GraphFile(file);
        }

        public byte[] getGraphFile() {
            return graphFile;
        }
    }

    public Config(
            GraphFile graphFile,
            @Nullable LabelsFile labelsFile,
            float thresholdValue,
            InputImageTensorProvider<T> inputImageTensorProvider,
            String feedInputTensorName,
            int maxPriorityObjects) {
        this.graphFile = graphFile;
        this.labelsFile = labelsFile;

        this.thresholdValue = thresholdValue;
        this.inputImageTensorProvider = inputImageTensorProvider;
        this.feedInputTensorName = feedInputTensorName;
        this.maxPriorityObjects = maxPriorityObjects;
    }

    public GraphFile getGraphFile() {
        return graphFile;
    }

    public LabelsFile getLabelsFile() {
        return labelsFile;
    }

    public float getThresholdValue() {
        return thresholdValue;
    }

    public InputImageTensorProvider<T> getInputImageTensorProvider() {
        return inputImageTensorProvider;
    }

    public String getFeedInputTensorName() {
        return feedInputTensorName;
    }

    public int getMaxPriorityObjects() {
        return maxPriorityObjects;
    }


    public static class ConfigBuilder<T> {

        private GraphFile graphFile;
        private LabelsFile labelsFile;
        private float thresholdValue = 0.4f;

        private String feedInputTensorName;
        private InputImageTensorProvider<T> inputImageTensorProvider;

        private int maxPriorityObjects;

        public ConfigBuilder(@NotNull GraphFile graphFile) {
            this.graphFile = graphFile;

        }

        public ConfigBuilder<T> setMaxPriorityObjects(int maxPriorityObjects) {
            this.maxPriorityObjects = maxPriorityObjects;
            return this;
        }

        public ConfigBuilder<T> setConfidence(float confidence) {
            this.thresholdValue = confidence;
            return this;
        }

        public ConfigBuilder<T> setLabels(LabelsFile labelsFile) {
            this.labelsFile = labelsFile;
            return this;
        }

        public ConfigBuilder<T> setInputTensor(
                String feedInputTensorName,
                InputImageTensorProvider<T> inputImageTensorProvider) {
            this.feedInputTensorName = feedInputTensorName;
            this.inputImageTensorProvider = inputImageTensorProvider;
            return this;
        }

        public Config<T> build() {
            return new Config<T>(
                    graphFile,
                    labelsFile,
                    thresholdValue,
                    inputImageTensorProvider,
                    feedInputTensorName,
                    maxPriorityObjects
            );
        }
    }
}
package com.igla.tensorflow_easy.obj_recognition;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class CustomGraphProcessor<T> implements AutoCloseable {

    public abstract String[] resultOperationNames();

    private Session session;
    private Session.Runner runner;
    private List<Tensor<?>> outputTensors = new ArrayList<>(4);
    private List<Tensor<?>> inputTensors = new ArrayList<>(1);
    private List<String> operationNames = new ArrayList<>(resultOperationNames().length);

    public CustomGraphProcessor(@NotNull Graph graph) {
        this.session = new Session(graph);
        this.operationNames.addAll(Arrays.asList(this.resultOperationNames()));
    }

    public CustomGraphProcessor(@NotNull Graph graph, byte[] configProto) {
        this.session = new Session(graph, configProto);
        this.operationNames.addAll(Arrays.asList(this.resultOperationNames()));
    }

    public void feed(String operationName, Tensor<?> tensor) {
        this.runner = session.runner();
        runner.feed(operationName, tensor);
        inputTensors.clear();
        inputTensors.add(tensor);
    }

    public void run() {
        for (String operationName : resultOperationNames()) {
            runner.fetch(operationName);
        }
        outputTensors = runner.run();
        for (Tensor<?> tensor : inputTensors) {
            tensor.close();
        }
    }

    public void closeOutputTensors() {
        for (Tensor<?> tensor : outputTensors) {
            tensor.close();
        }
        outputTensors.clear();
    }

    public abstract T detections();

    @Nullable
    public Tensor<?> getTensor(String name) {
        for (int i = 0; i < operationNames.size(); i++) {
            if (operationNames.get(i).equals(name)) {
                return outputTensors.get(i);
            }
        }
        return null;
    }

    @Override
    public void close() {
        closeOutputTensors();
        session.close();
    }
}

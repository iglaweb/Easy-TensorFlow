package com.igla.tensorflow_easy.core;

import com.igla.tensorflow_easy.utils.IoUtils;
import com.igla.tensorflow_easy.utils.TensorFlowUtils;
import com.igla.tensorflow_easy.utils.Timber;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

import java.io.IOException;

public class InputModel implements AutoCloseable {

    byte[] configProto;
    Graph graph;
    SavedModelBundle savedModelBundle;

    public InputModel(Graph graph, byte[] configProto) {
        this.graph = graph;
        this.configProto = configProto;
    }

    public InputModel(SavedModelBundle savedModelBundle) {
        this.savedModelBundle = savedModelBundle;
    }

    public Session requestSession() {
        if (graph != null) {
            return configProto != null ? new Session(graph, configProto) : new Session(graph);
        } else {
            return savedModelBundle.session();
        }
    }

    @NotNull
    public static InputModel createModelObj(Config.GraphFile graphFile) throws IOException {
        return createModelObj(graphFile, null);
    }

    public static InputModel createModelObj(Config.GraphFile graphFile, byte[] configProto) throws IOException {
        if (graphFile.isSavedBundle()) {
            SavedModelBundle savedModelBundle =
                    SavedModelBundle.load(graphFile.getFile().getAbsolutePath(), "serve");
            return new InputModel(savedModelBundle);
        } else {
            Graph graph = readGraph(graphFile);
            return new InputModel(graph, configProto);
        }
    }

    private static Graph readGraph(Config.GraphFile graphFile) throws IOException {
        if (graphFile.getFile() != null) {
            return TensorFlowUtils.readGraphFile(graphFile.getFile());
        } else if (graphFile.getGraphFile() != null) {
            return setup(graphFile.getGraphFile());
        }
        return null;
    }

    private static Graph setup(byte[] graphBytes) {
        long start = System.currentTimeMillis();
        Timber.i("Loading TensorFlow graph...");
        Graph graph = TensorFlowUtils.importGraph(graphBytes);
        long timeDiff = System.currentTimeMillis() - start;
        System.out.println("TensorFlow graph loaded in " + timeDiff + " ms");
        return graph;
    }

    @Override
    public void close() {
        IoUtils.closeQuietly(graph);
        IoUtils.closeQuietly(savedModelBundle);
    }
}

package com.igla.tensorflow_easy.sample.implementations.yolo;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.DetectorBase;
import com.igla.tensorflow_easy.models.Detection;
import com.igla.tensorflow_easy.obj_recognition.CustomGraphProcessor;
import com.igla.tensorflow_easy.sample.implementations.yolo.models.Recognition;

import java.io.IOException;
import java.util.List;

public class Yolov2ObjectDetector<T> extends
        DetectorBase<T, Recognition, Detection> {

    public Yolov2ObjectDetector(Config<T> config) throws IOException {
        super(config);
    }

    @Override
    public CustomGraphProcessor<Detection> createGraphProcessor() {
        return new YoloCustomGraphProcessor(this.graph, fullTraceRunOptions());
    }

    @Override
    public List<Recognition> processDetections(Detection detection, int width, int height) {
        return YOLOClassifier.getInstance().classifyImage(
                detection.getDetection_scores(),
                labels
        );
    }

    @Override
    public void close() {
        super.close();
    }

    /***
     * https://github.com/tensorflow/tensorflow/blob/9752b117ff63f204c4975cad52b5aab5c1f5e9a9/tensorflow/java/src/test/java/org/tensorflow/SessionTest.java
     * @return config options
     */
    private static byte[] fullTraceRunOptions() {
        // Ideally this would use the generated Java sources for protocol buffers
        // and end up with something like the snippet below. However, generating
        // the Java files for the .proto files in tensorflow/core:protos_all is
        // a bit cumbersome in bazel until the proto_library rule is setup.
        //
        // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
        // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
        // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
        //
        // For this test, for now, the use of specific bytes suffices.
        return new byte[]{0x08, 0x03};
    /*
    return org.tensorflow.framework.RunOptions.newBuilder()
        .setTraceLevel(RunOptions.TraceLevel.FULL_TRACE)
        .build()
        .toByteArray();
    */
    }
}

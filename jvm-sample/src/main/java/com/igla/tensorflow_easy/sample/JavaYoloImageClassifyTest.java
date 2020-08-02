package com.igla.tensorflow_easy.sample;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.sample.implementations.yolo.Yolov2ImageTensorProvider;
import com.igla.tensorflow_easy.sample.implementations.yolo.Yolov2ObjectDetector;
import com.igla.tensorflow_easy.sample.implementations.yolo.models.Recognition;
import com.igla.tensorflow_easy.sample.utils.JavaConsoleReportingTree;
import com.igla.tensorflow_easy.sample.utils.ResourceUtils;
import com.igla.tensorflow_easy.utils.IoUtils;
import com.igla.tensorflow_easy.utils.Timber;
import org.tensorflow.TensorFlow;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import static com.igla.tensorflow_easy.sample.utils.ResourceUtils.timing;

/**
 * Based on the use of https://github.com/szaza/tensorflow-example-java
 */
public class JavaYoloImageClassifyTest {

    public static void main(String[] args) throws IOException {
        System.out.println("TensorFlow version: " + TensorFlow.version());
        Timber.plant(new JavaConsoleReportingTree());

        File labelsFile = ResourceUtils.getFile("", "models/yolo-voc-labels.txt");
        File graphFile = ResourceUtils.getFile("", "models/yolo-voc.pb");

        List<String> labels =
                IoUtils.readAllLines(Paths.get(labelsFile.toURI()));

        File imageFile01 = ResourceUtils.getFile("", "img/cat.jpg");
        byte[] imageBytes = IoUtils.readAllBytes(Paths.get(imageFile01.toURI()));

        InputImageTensorProvider<byte[]> inputImageTensorProvider = new Yolov2ImageTensorProvider();
        Config<byte[]> configBuilder = new Config.ConfigBuilder<byte[]>(
                Config.GraphFile.create(graphFile))
                .setLabels(Config.LabelsFile.create(labels))
                .setConfidence(0.4f)
                .setMaxPriorityObjects(1)
                .setInputTensor("input", inputImageTensorProvider)
                .build();

        Yolov2ObjectDetector<byte[]> yolov2ObjectDetector = new Yolov2ObjectDetector<>(configBuilder);
        timing(() -> {
            List<Recognition> objectRecognitions = yolov2ObjectDetector.classifyImage(imageBytes);
            printToConsole(objectRecognitions);
        });

        File imageFile01_ = ResourceUtils.getFile("", "FFDB_samples/img_18.jpg");
        byte[] imageBytes_ = IoUtils.readAllBytes(Paths.get(imageFile01_.toURI()));
        timing(() -> {
            List<Recognition> objectRecognitions_ = yolov2ObjectDetector.classifyImage(imageBytes_);
            printToConsole(objectRecognitions_);
        });
    }

    /**
     * Prints out the recognize objects and its confidence
     *
     * @param recognitions list of recognitions
     */
    private static void printToConsole(final List<Recognition> recognitions) {
        for (Recognition recognition : recognitions) {
            Timber.i("Object: %s - confidence: %.2f%%", recognition.getTitle(), recognition.getConfidence());
        }
    }
}

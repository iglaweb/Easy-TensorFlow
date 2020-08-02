package com.igla.tensorflow_easy.sample;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.javax.InputImageTensorBufferedImageProvider;
import com.igla.tensorflow_easy.models.ObjectRecognition;
import com.igla.tensorflow_easy.obj_recognition.ObjectDetector;
import com.igla.tensorflow_easy.sample.implementations.ssd_mobilenet.SsdMobileNetObjectDetector;
import com.igla.tensorflow_easy.sample.utils.JavaConsoleReportingTree;
import com.igla.tensorflow_easy.sample.utils.ResourceUtils;
import com.igla.tensorflow_easy.utils.IoUtils;
import com.igla.tensorflow_easy.utils.Timber;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import static com.igla.tensorflow_easy.sample.utils.ResourceUtils.timing;

public class JavaClassificationImageTest {

    private final static String CLASSIFICATION_DIR = "models/classification_ssd_mobilenet";

    public static void main(String[] args) throws IOException {
        // setup logging
        Timber.plant(new JavaConsoleReportingTree());

        File imageFile01 = ResourceUtils.getFile("", "img/image_test_400_300.jpg");
        BufferedImage image = ImageIO.read(imageFile01);

        File modelFile = getClassificationModel();
        File labelFile = getClassificationLabels();

        List<String> labels = IoUtils.readAllLines(Paths.get(labelFile.toURI()));

        InputImageTensorProvider<BufferedImage> tensorBufferedImageProvider = new InputImageTensorBufferedImageProvider();

        Config<BufferedImage> configBuilder = new Config.ConfigBuilder<BufferedImage>(Config.GraphFile.create(modelFile))
                .setLabels(Config.LabelsFile.create(labels))
                .setConfidence(0.4f)
                .setMaxPriorityObjects(1)
                .setInputTensor("image_tensor", tensorBufferedImageProvider)
                .build();

        ObjectDetector<BufferedImage> objectDetector = new SsdMobileNetObjectDetector<>(configBuilder);

        timing(() -> {
            List<ObjectRecognition> objectRecognitions = objectDetector.classifyImage(image);
            Timber.i(objectRecognitions.toString());
        });
    }

    private static File getClassificationModel() {
        final String graphName = "frozen_inference_graph.pb";
        return ResourceUtils.getFile(CLASSIFICATION_DIR, graphName);
    }

    private static File getClassificationLabels() {
        final String labelsName = "coco_labels.txt";
        return ResourceUtils.getFile(CLASSIFICATION_DIR, labelsName);
    }
}

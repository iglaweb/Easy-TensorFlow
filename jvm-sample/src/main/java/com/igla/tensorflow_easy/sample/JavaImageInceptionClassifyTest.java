package com.igla.tensorflow_easy.sample;

import com.igla.tensorflow_easy.classifier.models.ClassifyRecognition;
import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.sample.implementations.inception.InceptionClassifierDetector;
import com.igla.tensorflow_easy.sample.implementations.inception.InceptionImageTensorProvider;
import com.igla.tensorflow_easy.sample.utils.JavaConsoleReportingTree;
import com.igla.tensorflow_easy.sample.utils.ResourceUtils;
import com.igla.tensorflow_easy.utils.IoUtils;
import com.igla.tensorflow_easy.utils.Timber;
import org.tensorflow.TensorFlow;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;

import static com.igla.tensorflow_easy.sample.utils.ResourceUtils.timing;

/**
 * Sample use of the TensorFlow Java API to label images using a pre-trained model.
 * Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)
 * "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
 */
public class JavaImageInceptionClassifyTest {

    public static void main(String[] args) throws IOException {
        System.out.println("TensorFlow version: " + TensorFlow.version());
        Timber.plant(new JavaConsoleReportingTree());

        File labelsFile = ResourceUtils.getFile("", "models/imagenet_comp_graph_label_strings.txt");
        File graphFile = ResourceUtils.getFile("", "models/tensorflow_inception_graph.pb");

        List<String> labels =
                IoUtils.readAllLines(Paths.get(labelsFile.toURI()));

        File imageFile01 = ResourceUtils.getFile("", "img/cat.jpg");
        byte[] imageBytes = IoUtils.readAllBytes(Paths.get(imageFile01.toURI()));

        InputImageTensorProvider<byte[]> inputImageTensorProvider = new InceptionImageTensorProvider();
        Config<byte[]> configBuilder = new Config.ConfigBuilder<byte[]>(
                Config.GraphFile.create(graphFile))
                .setLabels(Config.LabelsFile.create(labels))
                .setConfidence(0.4f)
                .setMaxPriorityObjects(1)
                .setInputTensor("input", inputImageTensorProvider)
                .build();
        InceptionClassifierDetector<byte[]> inceptionClassifierDetector = new InceptionClassifierDetector<>(configBuilder);

        timing(() -> {
            List<ClassifyRecognition> objectRecognitions = inceptionClassifierDetector.classifyImage(imageBytes);
            ClassifyRecognition classifyRecognition = objectRecognitions.get(0);
            System.out.printf(Locale.US, "BEST MATCH: %s (%.2f%% likely)%n",
                    classifyRecognition.getLabel(),
                    classifyRecognition.getConfidence() * 100f);
        });

        File imageFile01_ = ResourceUtils.getFile("", "img/image_test_400_300.jpg");
        byte[] imageBytes_ = IoUtils.readAllBytes(Paths.get(imageFile01_.toURI()));

        timing(() -> {
            List<ClassifyRecognition> objectRecognitions_ = inceptionClassifierDetector.classifyImage(imageBytes_);
            ClassifyRecognition classifyRecognition_ = objectRecognitions_.get(0);
            System.out.printf(Locale.US, "BEST MATCH: %s (%.2f%% likely)%n",
                    classifyRecognition_.getLabel(),
                    classifyRecognition_.getConfidence() * 100f);
        });
        inceptionClassifierDetector.close();
    }
}

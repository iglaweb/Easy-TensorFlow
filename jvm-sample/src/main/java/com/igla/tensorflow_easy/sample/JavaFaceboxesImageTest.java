package com.igla.tensorflow_easy.sample;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import com.igla.tensorflow_easy.obj_recognition.ObjectDetector;
import com.igla.tensorflow_easy.javax.InputImageTensorBufferedImageProvider;
import com.igla.tensorflow_easy.models.ObjectRecognition;
import com.igla.tensorflow_easy.sample.implementations.faceboxes.FaceboxesObjectDetector;
import com.igla.tensorflow_easy.sample.utils.JavaConsoleReportingTree;
import com.igla.tensorflow_easy.sample.utils.ResourceUtils;
import com.igla.tensorflow_easy.utils.Timber;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

public class JavaFaceboxesImageTest {

    public static void main(String[] args) throws Exception {
        // setup logging
        Timber.plant(new JavaConsoleReportingTree());

        File imageFile01 = ResourceUtils.getFile("", "img/image_test_400_300.jpg");
        BufferedImage image = ImageIO.read(imageFile01);

        File modelFile = ResourceUtils.getObjectDetectionModel();
        InputImageTensorProvider<BufferedImage> tensorBufferedImageProvider = new InputImageTensorBufferedImageProvider();

        Config<BufferedImage> configBuilder = new Config.ConfigBuilder<BufferedImage>(Config.GraphFile.create(modelFile))
                .setConfidence(0.4f)
                .setMaxPriorityObjects(1)
                .setInputTensor("image_tensor", tensorBufferedImageProvider)
                .build();

        ObjectDetector<BufferedImage> objectDetector = new FaceboxesObjectDetector<>(configBuilder);
        List<ObjectRecognition> objectRecognitions = objectDetector.classifyImage(image);

        tensorBufferedImageProvider.close();
        objectDetector.close();

        if (objectRecognitions.isEmpty()) {
            System.out.println("Empty recognitions");
            return;
        }

        for (ObjectRecognition recognition : objectRecognitions) {
            Timber.i(recognition.toString());
        }
    }
}

package com.igla.tensorflow_easy.sample;

import com.igla.tensorflow_easy.core.Config;
import com.igla.tensorflow_easy.obj_recognition.ObjectDetector;
import com.igla.tensorflow_easy.models.ObjectRecognition;
import com.igla.tensorflow_easy.opencv.InputImageTensorOpencvProvider;
import com.igla.tensorflow_easy.sample.implementations.faceboxes.FaceboxesObjectDetector;
import com.igla.tensorflow_easy.sample.utils.ImageSizeUtils;
import com.igla.tensorflow_easy.sample.utils.JavaConsoleReportingTree;
import com.igla.tensorflow_easy.sample.utils.ResourceUtils;
import com.igla.tensorflow_easy.utils.Timber;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class JavaVideoCaptureFaceboxesTest {

    private static void runTensorFlowCameraDetection(ObjectDetector<Mat> objectDetector) {
        VideoCapture videoCapture = new VideoCapture(0);
        if (videoCapture.isOpened()) {
            int frames = 0;
            while (true) {
                Mat matFrame = new Mat();
                if (videoCapture.read(matFrame)) {
                    frames++;
                    if (matFrame.empty()) { //to avoid assert exception in color.cpp
                        System.out.println("Grabbed Mat is empty! Get next frame...");
                        continue;
                    }
                    Mat resizedMat = ImageSizeUtils.createSizedMat(matFrame, 450);
                    long start = System.currentTimeMillis();
                    List<ObjectRecognition> recognitions = objectDetector.classifyImage(resizedMat);
                    long timeDiff = System.currentTimeMillis() - start;
                    System.out.println("Recognitions: " + recognitions.size() + ", time: " + timeDiff + " ms");
                    resizedMat.release();
                    matFrame.release();

                    if (frames % 10 == 0) {
                        System.out.println("Frames count read: " + frames);
                    }
                } else {
                    System.out.println("Frame is not obtained. Break!");
                    break;
                }
            }
            videoCapture.release(); //finalize
            System.out.println("Camera closed");

        } else {
            System.out.println("Camera NOT opened");
        }
    }

    public static void main(String[] args) throws IOException {
        Timber.plant(new JavaConsoleReportingTree());
        ResourceUtils.loadOpenCv();

        File faceboxesFile = ResourceUtils.getObjectDetectionModel();
        InputImageTensorOpencvProvider tensorOpencvProvider = new InputImageTensorOpencvProvider();

        Config<Mat> configBuilder = new Config.ConfigBuilder<Mat>(Config.GraphFile.create(faceboxesFile))
                .setConfidence(0.4f)
                .setMaxPriorityObjects(1)
                .setInputTensor("image_tensor", tensorOpencvProvider)
                .build();
        ObjectDetector<Mat> faceboxesEstimator = new FaceboxesObjectDetector<>(configBuilder);
        runTensorFlowCameraDetection(faceboxesEstimator);
        try {
            faceboxesEstimator.close();
        } catch (Exception e) {
            Timber.e(e);
        }
    }
}

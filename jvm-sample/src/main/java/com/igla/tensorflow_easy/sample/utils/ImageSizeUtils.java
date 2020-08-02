package com.igla.tensorflow_easy.sample.utils;

import org.jetbrains.annotations.NotNull;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public final class ImageSizeUtils {

    private ImageSizeUtils() {
        // no impl
    }

    /***
     * Return sized image. May return original
     * @param mat source
     * @return output
     */
    public static Mat createSizedMat(@NotNull Mat mat, int maxSize) {
        int width = mat.width();
        int height = mat.height();
        if (width > maxSize || height > maxSize) {
            Mat resizedMat = new Mat();
            sizeMat(mat, resizedMat, maxSize);
            return resizedMat;
        }
        return mat; // return input mat
    }

    private static void sizeMat(@NotNull Mat src, @NotNull Mat dst, int maxSize) {
        int width = src.width();
        int height = src.height();
        if (width > maxSize || height > maxSize) {
            final float ratio;
            if (width > height) {
                ratio = (float) width / maxSize;
                width = maxSize;
                height = (int) ((float) height / ratio);
            } else {
                ratio = (float) height / maxSize;
                height = maxSize;
                width = (int) ((float) width / ratio);
            }
            Imgproc.resize(
                    src,
                    dst,
                    new Size(width, height),
                    0, 0,
                    Imgproc.INTER_AREA
            );
        }
    }
}

package com.igla.tensorflow_easy.opencv;

import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import org.opencv.core.Mat;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.nio.ByteBuffer;

public class InputImageTensorOpencvProvider implements InputImageTensorProvider<Mat> {

    @Override
    public Tensor<UInt8> getTensor(Mat m) {
        byte[] data = new byte[m.rows() * m.cols() * m.channels()];
        m.get(0, 0, data);
        // mat seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);

        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[]{BATCH_SIZE, m.height(), m.width(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }

    @Override
    public int getImageHeight(Mat image) {
        return image.height();
    }

    @Override
    public int getImageWidth(Mat image) {
        return image.width();
    }

    /**
     * Converts image pixels from the type BGR to RGB
     *
     * @param data
     */
    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    @Override
    public void close() {
    }
}

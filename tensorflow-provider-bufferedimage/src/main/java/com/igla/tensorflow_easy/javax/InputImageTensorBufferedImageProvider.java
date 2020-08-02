package com.igla.tensorflow_easy.javax;

import com.igla.tensorflow_easy.core.InputImageTensorProvider;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.ByteBuffer;

public class InputImageTensorBufferedImageProvider implements InputImageTensorProvider<BufferedImage> {

    private static final long BATCH_SIZE = 1;
    private static final long CHANNELS = 3;

    @Override
    public Tensor<UInt8> getTensor(BufferedImage image) {
        byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
        bgr2rgb(data);

        long[] shape = new long[]{BATCH_SIZE, image.getHeight(), image.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }

    @Override
    public int getImageHeight(BufferedImage image) {
        return image.getHeight();
    }

    @Override
    public int getImageWidth(BufferedImage image) {
        return image.getWidth();
    }

    /**
     * Converts image pixels from the type BGR to RGB
     *
     * @param data byte array
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
        // no impl
    }
}

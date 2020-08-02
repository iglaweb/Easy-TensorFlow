package com.igla.tensorflow_easy.utils;

import org.jetbrains.annotations.Nullable;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class IoUtils {

    public static byte[] readAllBytes(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
        }
        return null;
    }

    public static List<String> readAllLines(Path path) {
        try {
            return Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
        }
        return null;
    }

    /**
     * Closes 'closeable', ignoring any checked exceptions. Does nothing if 'closeable' is null.
     */
    public static void closeQuietly(@Nullable final Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Loads the graph from the pre-trained model file.
     *
     * @param graphFile the trained graph/model file data to load from.
     * @return The graph model data.
     * @throws IOException if an I/O error occurs.
     */
    static byte[] loadGraph(InputStream graphFile) throws IOException {
        int baosInitSize = Math.max(graphFile.available(), 16384);
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(baosInitSize)) {
            int numBytesRead;
            byte[] buf = new byte[16384];
            while ((numBytesRead = graphFile.read(buf, 0, buf.length)) != -1) {
                baos.write(buf, 0, numBytesRead);
            }
            return baos.toByteArray();
        }
    }
}

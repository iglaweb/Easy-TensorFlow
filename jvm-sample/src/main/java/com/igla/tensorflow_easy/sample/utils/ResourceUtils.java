package com.igla.tensorflow_easy.sample.utils;

import com.igla.tensorflow_easy.sample.opencv.FileAbsolutePathLibraryLoader;
import com.igla.tensorflow_easy.sample.opencv.LibraryLoader;
import com.igla.tensorflow_easy.sample.opencv.OpenCVLoader;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public final class ResourceUtils {
    private ResourceUtils() {
    }

    private static <R> R timing(Supplier<R> operation) {
        long start = System.nanoTime();
        R result = operation.get();
        System.out.printf("Execution took %dms\n", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));
        return result;
    }

    public static void timing(Runnable operation) {
        timing(() -> {
            operation.run();
            return null;
        });
    }

    public static File getFile(String dir, String fileName) {
        Path path = Paths.get(dir, fileName);
        URL resource = Thread.currentThread().getContextClassLoader().getResource(path.toString());
        if (resource == null) {
            throw new RuntimeException("A needed resource file was not found. The missing file is: " + path.toString());
        }
        return new File(resource.getFile());
    }

    public static void loadOpenCv() {
        LibraryLoader fileNameLibraryLoader = new FileAbsolutePathLibraryLoader();
        File opencvNative = ResourceUtils.getFile("", "libopencv_java420.dylib");
        String opencvPath = opencvNative.getAbsolutePath();
        OpenCVLoader.initDebug(fileNameLibraryLoader, opencvPath);
    }
}

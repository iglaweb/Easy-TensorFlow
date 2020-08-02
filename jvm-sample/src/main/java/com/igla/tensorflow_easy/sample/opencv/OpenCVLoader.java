package com.igla.tensorflow_easy.sample.opencv;


/**
 * Helper class provides common initialization methods for OpenCV library.
 */
public class OpenCVLoader {
    /**
     * Loads and initializes OpenCV library from current application package. Roughly, it's an analog of system.loadLibrary("opencv_java").
     *
     * @return Returns true is initialization of OpenCV was successful.
     */
    public static boolean initDebug(LibraryLoader libraryLoader, String libraryName) {
        return StaticHelper.initOpenCV(false, libraryLoader, libraryName);
    }
}

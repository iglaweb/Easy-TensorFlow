package com.igla.tensorflow_easy.sample.opencv;

import com.igla.tensorflow_easy.utils.Timber;

public class FileAbsolutePathLibraryLoader implements LibraryLoader {

    @Override
    public void loadLibraryFile(String filename) {
        try {
            System.load(filename);
        } catch (UnsatisfiedLinkError e) {
            Timber.e(e);
        }
    }
}

package com.igla.tensorflow_easy.sample.opencv;

import java.util.logging.Level;
import java.util.logging.Logger;

public class FileAbsolutePathLibraryLoader implements LibraryLoader {

    private static final Logger logger = Logger.getLogger(FileAbsolutePathLibraryLoader.class.getName());

    @Override
    public void loadLibraryFile(String filename) {
        try {
            System.load(filename);
        } catch (UnsatisfiedLinkError e) {
            logger.log(Level.SEVERE, " ### " + filename + " library not found! ###");
        }
    }
}

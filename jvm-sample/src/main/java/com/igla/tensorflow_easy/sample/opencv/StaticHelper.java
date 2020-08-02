package com.igla.tensorflow_easy.sample.opencv;

import org.opencv.core.Core;

import java.util.StringTokenizer;
import java.util.logging.Level;
import java.util.logging.Logger;

class StaticHelper {

    private static final Logger logger = Logger.getLogger(StaticHelper.class.getName());

    public static boolean initOpenCV(boolean InitCuda, LibraryLoader libraryLoader, String opencvCoreLibNative) {
        boolean result;
        String libs = "";

        if (InitCuda) {
            loadLibrary(libraryLoader, "cudart");
            loadLibrary(libraryLoader, "nppc");
            loadLibrary(libraryLoader, "nppi");
            loadLibrary(libraryLoader, "npps");
            loadLibrary(libraryLoader, "cufft");
            loadLibrary(libraryLoader, "cublas");
        }

        logger.log(Level.INFO, "Trying to get library list");
        try {
            libraryLoader.loadLibraryFile("opencv_info");
            libs = getLibraryList();
        } catch (UnsatisfiedLinkError e) {
            logger.log(Level.SEVERE, "OpenCV error: Cannot load info library for OpenCV");
        }

        logger.log(Level.INFO, "Library list: \"" + libs + "\"");
        logger.log(Level.INFO, "First attempt to load libs");
        if (initOpenCVLibs(libraryLoader, libs, opencvCoreLibNative)) {
            logger.log(Level.SEVERE, "First attempt to load libs is OK");
            String eol = System.getProperty("line.separator");
            for (String str : Core.getBuildInformation().split(eol))
                logger.log(Level.INFO, str);

            result = true;
        } else {
            logger.log(Level.SEVERE, "First attempt to load libs fails");
            result = false;
        }

        return result;
    }

    private static boolean loadLibrary(LibraryLoader libraryLoader, String Name) {
        boolean result = true;

        logger.log(Level.INFO, "Trying to load library " + Name);
        try {
            libraryLoader.loadLibraryFile(Name);
            logger.log(Level.INFO, "Library " + Name + " loaded");
        } catch (UnsatisfiedLinkError e) {
            logger.log(Level.SEVERE, "Cannot load library \"" + Name + "\"");
            result = false;
        }

        return result;
    }

    private static boolean initOpenCVLibs(LibraryLoader libraryLoader, String Libs, String opencvCoreLibNative) {
        logger.log(Level.INFO, "Trying to init OpenCV libs");

        boolean result = true;

        if ((null != Libs) && (Libs.length() != 0)) {
            logger.log(Level.INFO, "Trying to load libs by dependency list");
            StringTokenizer splitter = new StringTokenizer(Libs, ";");
            while (splitter.hasMoreTokens()) {
                result &= loadLibrary(libraryLoader, splitter.nextToken());
            }
        } else {
            // If dependencies list is not defined or empty.
            result = loadLibrary(libraryLoader, opencvCoreLibNative);
        }

        return result;
    }

    private static native String getLibraryList();
}

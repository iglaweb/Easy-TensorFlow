package com.igla.tensorflow_easy.sample.utils;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import com.igla.tensorflow_easy.utils.Tree;

import java.util.logging.Level;
import java.util.logging.Logger;

public class JavaConsoleReportingTree extends Tree {

    private final Logger logger = Logger.getLogger(JavaConsoleReportingTree.class.getName());

    public JavaConsoleReportingTree() {
        setAddStackTrace(false);
    }

    @Override
    protected void log(int priority, @Nullable String tag, @NotNull String message, @Nullable Throwable throwable) {
        logger.log(Level.INFO, message);
    }
}

TensorFlow-Easy
======

TensorFlow-Easy will help you to run ML models in a simplified and uniformed way.


It is efficient by default:

 * Unified way to run models.
 * Memory management for TensorFlow models.


Example 1: Run Faceboxes [model][faceboxes-model]
---------

This program recognizes faces using Faceboxes [Full source][faceboxes-example].

```java
BufferedImage image = ImageIO.read("test.jpg");

File modelFile = ResourceUtils.getFile("", "models/faceboxes_model.pb");
InputImageTensorProvider<BufferedImage> tensorBufferedImageProvider = new InputImageTensorBufferedImageProvider();

Config<BufferedImage> configBuilder = new Config.ConfigBuilder<BufferedImage>(Config.GraphFile.create(modelFile))
        .setConfidence(0.4f)
        .setMaxPriorityObjects(1)
        .setInputTensor("image_tensor", tensorBufferedImageProvider)
        .build();

ObjectDetector<BufferedImage> objectDetector = new FaceboxesObjectDetector<>(configBuilder);
List<ObjectRecognition> objectRecognitions = objectDetector.classifyImage(image);
```


Example 2: Run Yolo [model][yolo-model]
----------------

This program posts data to a service. [Full source][yolo-example].

```java
File labelsFile = ResourceUtils.getFile("", "models/yolo-voc-labels.txt");
File graphFile = ResourceUtils.getFile("", "models/yolo-voc.pb");

List<String> labels =
        IoUtils.readAllLines(Paths.get(labelsFile.toURI()));

File imageFile01 = ResourceUtils.getFile("", "test.jpg");
byte[] imageBytes = IoUtils.readAllBytes(Paths.get(imageFile01.toURI()));

InputImageTensorProvider<byte[]> inputImageTensorProvider = new Yolov2ImageTensorProvider();
Config<byte[]> configBuilder = new Config.ConfigBuilder<byte[]>(
        Config.GraphFile.create(graphFile))
        .setLabels(Config.LabelsFile.create(labels))
        .setConfidence(0.4f)
        .setMaxPriorityObjects(1)
        .setInputTensor("input", inputImageTensorProvider)
        .build();

Yolov2ObjectDetector<byte[]> yolov2ObjectDetector = new Yolov2ObjectDetector<>(configBuilder);
timing(() -> {
    List<Recognition> objectRecognitions = yolov2ObjectDetector.classifyImage(imageBytes);
});
```


Requirements
------------

TensorFlow-Easy works on Java 8+.

TensorFlow-Easy depends on [TensorFlow-java][tensorflow-java] 2.3. 

To get it work you need to add tensorflow libraries specific for your [platform][tensorflow-lib-path] to java library path:

```java
// java instructions https://www.tensorflow.org/install/lang_java
def TENSORFLOW_LIB = 'jvm-sample/libs/libtensorflow_jni-cpu-darwin-x86_64-2.3.0'
tasks.withType(JavaExec) {
    systemProperty "java.library.path", TENSORFLOW_LIB
}
```

Releases
--------

Our [change log][changelog] has release history.

```kotlin
repositories { 
	maven {
	    url  "https://dl.bintray.com/iglaweb/maven"
	}
}
implementation 'com.igla.tensorflow_easy:tensorflow:0.1'
```


License
-------

```
Copyright 2020

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

 [changelog]: https://github.com/iglaweb/Easy-TensorFlow/blob/master/CHANGELOG.md
 [faceboxes-model]: https://github.com/TropComplique/FaceBoxes-tensorflow
 [faceboxes-example]: https://github.com/iglaweb/Easy-TensorFlow/blob/master/jvm-sample/src/main/java/com/igla/tensorflow_easy/sample/JavaFaceboxesImageTest.java
 [yolo-model]: https://github.com/szaza/android-yolo-v2
 [yolo-example]: https://github.com/iglaweb/Easy-TensorFlow/blob/master/jvm-sample/src/main/java/com/igla/tensorflow_easy/sample/JavaYoloImageClassifyTest.java
 [tensorflow-java]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java
 [tensorflow-lib-path]: https://github.com/iglaweb/Easy-TensorFlow/blob/master/jvm-sample/build.gradle
 [kotlin]: https://kotlinlang.org/

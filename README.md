# MediaPipe Tasks Hand Landmark Detection Android Demo

### Overview

This is an android application that uses the hand and pose landmarker of Mediapipe to detect landmarks, ans a tensorflow trained model to clasify the ASL sign given the landmarks. For the tensorflow model inference Chaquopy has been used, so in order to build the application it is needed to indicate the python executer path in Gradle.

### Use

The application classifies 25 signs and 'None' means that no sign has been detected. It must be performed horizontally and with slow movements as it runs slowly. The end of the signing sequence is indicated by a pause in the drawn landmarks.

/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.mediapipe.examples.handlandmarker.fragment.InputProcessing
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlinx.coroutines.runBlocking
import java.io.BufferedReader
import java.io.InputStreamReader

class ArmsLandmarkerHelper(
    var minArmDetectionConfidence: Float = DEFAULT_ARM_DETECTION_CONFIDENCE,
    var minArmTrackingConfidence: Float = DEFAULT_ARM_TRACKING_CONFIDENCE,
    var minArmPresenceConfidence: Float = DEFAULT_ARM_PRESENCE_CONFIDENCE,
    var maxNumHands: Int = DEFAULT_NUM_HANDS,   // 2
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    var currentPoseModel: Int = MODEL_POSE_LANDMARKER_FULL,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    //val handLandmarkerHelperListener: LandmarkerListener? = null
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val armLandmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example this needs to be a var so it can be reset on changes.
    // If the Hand Landmarker will not change, a lazy val would be preferable.
    private var handLandmarker: HandLandmarker? = null
    // For this example this needs to be a var so it can be reset on changes.
    // If the Pose Landmarker will not change, a lazy val would be preferable.
    private var poseLandmarker: PoseLandmarker? = null

    var resultsSequence= Array<Float>(2088) { 0.0F }    // order: ( pose (12*4), lh (21*3), rh (21*3) ) * 12 frames
    var frameCounterHand = 0    // counts the frame in the sequence of handListener, max is 12 frames per sequence
    var frameCounterPose = 0    // counts the frame in the sequence of poseListener, max is 12 frames per sequence
    val frameKeypoints = 174    // total keypoints per frame
    val poseKeypointsSize = 12*4    // total pose keypoints per frame
    val handKeypointsSize = 21*3    // one hand keypoints per frame
    var activeSequenceShift : Boolean = false
    lateinit var pythonModule :PyObject
    lateinit var actionsLabels : List<String>
    val framesPerSequence = 12
    val framesShiftInSequence = 6
    var framesCounter = 1

    init {
        setupArmLandmarker()
        setPython()
    }

    private fun setPython(){
        if (! Python.isStarted()) {
            Log.i(TAG, "python go")
            // Starting python interpreter
            Python.start(AndroidPlatform(context))

            // Getting an insterpreter instance
            val pythonInterpreter = Python.getInstance()

            // Calling the module main.py
            pythonModule = pythonInterpreter.getModule("main")

            // Setting up the TF Model
            pythonModule.callAttr("loadWeights")

            // Load sequence from file to plain List
            // 6 welcome, 9 please, 19 thanks
            //val sequence_list : Array<Float> = readFileToList("19.txt")

            //val pyobj:PyObject = py.getModule("main")
            //val result = pyobj.callAttr("loadWeights", sequence_list)
        }
    }

    // Reads values from a file where each row is a float
    // Returns a List with them
    fun readFileToList(filename: String): Array<Float> {
        val values = Array<Float>(2088){0.0F}
        var counter:Int = 0
        val assetManager = this.context?.assets
        assetManager?.open(filename)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line: String? = reader.readLine()
                while (line != null) {
                    val floatValue = line.toFloatOrNull()
                    if (floatValue != null) {
                        values[counter] = floatValue
                    }
                    line = reader.readLine()
                    counter++
                }
            }
        }
        Log.i("Input List File first value", values.size.toString())
        return values
    }

    fun loadLabels(filename: String): List<String> {
        var values : MutableList<String> = mutableListOf()
        val assetManager = this.context?.assets
        assetManager?.open(filename)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line: String? = reader.readLine()
                while (line != null) {
                    values.add(line)
                    line = reader.readLine()
                }
            }
        }
        return values.toList()
    }

    /*private fun testTFInterpreter (){
        var tfInterpreter = TFInterpreter(this.context)
        tfInterpreter.setInterpreter()
    }*/

    private fun testTensorflowModel(){
        // Test to check that the tf lite model works
        var tfProcess: InputProcessing = InputProcessing(this.context)
        tfProcess.setInterpreter()

    }

    fun clearArmLandmarker() {
        handLandmarker?.close()
        handLandmarker = null
        poseLandmarker?.close()
        poseLandmarker = null
    }

    // Return running status of ArmsLandmarkerHelper
    fun isClose(): Boolean {
        return handLandmarker == null && poseLandmarker == null
    }

    // Initialize the Arm landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupArmLandmarker() {
        testTensorflowModel()
        // Set general arm landmarker options
        val handBaseOptionBuilder = BaseOptions.builder()
        val poseBaseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                handBaseOptionBuilder.setDelegate(Delegate.CPU)
                poseBaseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                handBaseOptionBuilder.setDelegate(Delegate.GPU)
                poseBaseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        val poseModelName =
            when (currentPoseModel) {
                MODEL_POSE_LANDMARKER_FULL -> "pose_landmarker_full.task"   // reach 18 fps aprox
                MODEL_POSE_LANDMARKER_LITE -> "pose_landmarker_lite.task"   // reach 44 fps aprox
                MODEL_POSE_LANDMARKER_HEAVY -> "pose_landmarker_heavy.task" // reach 4 fps aprox
                else -> "pose_landmarker_full.task"
            }

        // Load the mediapipe model
        handBaseOptionBuilder.setModelAssetPath(MP_HAND_LANDMARKER_TASK)
        poseBaseOptionBuilder.setModelAssetPath(poseModelName)

        // Check if runningMode is consistent with handLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (armLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "armLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        // loading the labels of actions
        this.actionsLabels = loadLabels("actions.txt")

        try {
            // Building hand larmarker
            val handBaseOptions = handBaseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Hand Landmarker.
            val handOptionsBuilder =
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(handBaseOptions)
                    .setMinHandDetectionConfidence(minArmDetectionConfidence)
                    .setMinTrackingConfidence(minArmTrackingConfidence)
                    .setMinHandPresenceConfidence(minArmPresenceConfidence)
                    .setNumHands(maxNumHands)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                handOptionsBuilder
                    .setResultListener(this::returnHandLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val handOptions = handOptionsBuilder.build()
            handLandmarker =
                HandLandmarker.createFromOptions(context, handOptions)

            // Building pose landmarker
            val poseBaseOptions = poseBaseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Pose Landmarker.
            val poseOptionsBuilder =
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(poseBaseOptions)
                    .setMinPoseDetectionConfidence(minArmDetectionConfidence)
                    .setMinTrackingConfidence(minArmTrackingConfidence)
                    .setMinPosePresenceConfidence(minArmPresenceConfidence)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                poseOptionsBuilder
                    .setResultListener(this::returnPoseLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val poseOptions = poseOptionsBuilder.build()
            poseLandmarker =
                PoseLandmarker.createFromOptions(context, poseOptions)

            Log.i(TAG, "Set up and build models")
        } catch (e: IllegalStateException) {
            armLandmarkerHelperListener?.onError(
                "Arm Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            armLandmarkerHelperListener?.onError(
                "Arm Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to ArmlandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
    }

    // Run hand hand landmark using MediaPipe Hand Landmarker API
    // Send the image to returnLiveStream when detected
    // the results will be available via the OutputHandler.ResultListener
    // provided in the HandLandmarker.HandLandmarkerOptions.
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        handLandmarker?.detectAsync(mpImage, frameTime)
        poseLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // hand landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): ArmResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the hand and pose landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val handResultList = mutableListOf<HandLandmarkerResult>()
        val poseResultList = mutableListOf<PoseLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run hand landmarker using MediaPipe Hand Landmarker API
                    handLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            handResultList.add(detectionResult)
                        } ?: run {
                        didErrorOccurred = true
                        armLandmarkerHelperListener?.onError(
                            "HandResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                    // Run pose landmarker using MediaPipe Pose Landmarker API
                    poseLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            poseResultList.add(detectionResult)
                        } ?: {
                        didErrorOccurred = true
                        armLandmarkerHelperListener?.onError(
                            "PoseResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    armLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ArmResultBundle(handResultList, poseResultList, inferenceTimePerFrameMs, height, width)
        }
    }

    // Accepted a Bitmap and runs hand landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ArmResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run hand landmarker using MediaPipe Hand Landmarker API
        val hResult : HandLandmarkerResult? = handLandmarker?.detect(mpImage)
        val pResult : PoseLandmarkerResult? = poseLandmarker?.detect(mpImage)

        if (hResult != null && pResult != null){
            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ArmResultBundle(
                listOf(hResult),
                listOf(pResult),
                inferenceTimeMs,
                image.height,
                image.width
            )
        }


        // If handLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        armLandmarkerHelperListener?.onError(
            "Hand Landmarker failed to detect."
        )
        return null
    }

    // Return the landmark result to this HanLandmarkerHelper's caller
    // Este es el callback del modelo de HandLandmarker
    private fun returnHandLivestreamResult(
        result: HandLandmarkerResult,
        input: MPImage
    ) {
        Log.i(TAG, "Hand Stream")
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()


        // save results in the sequnce data for later action inference
        // 21 keypoints per hand, 21 * 3 values
        // result.landmarks().size = 1 for 1 hand and '= 2' for 2 hands
        // For some reason the first frame comes with empty values
        var handCounter = 0
        for (landmark_list in result.landmarks()) {
            // iterate per hand
            var landmarkCounter = 0
            if (result.handednesses().get(handCounter).get(0).categoryName() == "Right" ){
                // For the right hand, it has to go second in the list
                for (value in landmark_list){
                    var pos = (frameKeypoints*frameCounterHand) + (landmarkCounter * 3) + handKeypointsSize + poseKeypointsSize
                    resultsSequence[pos] = value.x()
                    resultsSequence[pos+1] = value.y()
                    resultsSequence[pos+2] = value.z()
                    landmarkCounter++
                    /*if (!activeSequenceShift)
                        Log.i(TAG,
                            "Mano derecha en $pos [$frameKeypoints $frameCounterHand $landmarkCounter $handKeypointsSize $poseKeypointsSize]"
                        )*/
                }
            }
            else{
                // For left hand, it has to go first in the list
                for (value in landmark_list){
                    var pos = (frameKeypoints*frameCounterHand) + (landmarkCounter * 3) + poseKeypointsSize
                    resultsSequence[pos] = value.x()
                    resultsSequence[pos+1] = value.y()
                    resultsSequence[pos+2] = value.z()
                    landmarkCounter++
                    /*if (!activeSequenceShift)
                        Log.i(TAG, "Mano izquierda en $pos")*/
                }
            }
            handCounter++
        }
        //Log.i(TAG, "Size of hand landmarks: " + result.handednesses().toString() )

        frameCounterHand++
        if (frameCounterHand >= framesPerSequence){
            activeSequenceShift = true
            // subtract the iterator position of the sequence buffer
            // the '-1' is because indexes start at 0
            frameCounterHand = framesPerSequence - framesShiftInSequence
        }

        if (activeSequenceShift){

            // For every 6 frames send data to 'actions model' in python for inference
            // framesCounter is among 1 and framesPerSequence (inclusive)
            if (framesCounter == 12){
                // Create a new coroutine scope
                runBlocking {
                    // Code to be executed in the separate thread
                    Log.i(TAG, "Python run in another thread in kotlin from Hand")
                    var result = pythonModule.callAttr("runModel", resultsSequence).toInt()
                    Log.i(TAG, "The action is: " + actionsLabels[result])
                }
                // each frame shift the sequence buffer
                shiftLeftSequence(framesShiftInSequence)
            }
        }

        // send data for drawing
        armLandmarkerHelperListener?.onHandResults(
            HandResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width
            )
        )
    }

    // Return the landmark result to this PoseLandmarkerHelper's caller
    private fun returnPoseLivestreamResult(
        result: PoseLandmarkerResult,
        input: MPImage
    ) {
        Log.i(TAG, "Pose Stream")
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()

        // Remove useless keypoints and keep only the needed ones
        val filteredPoseLandmarks = filterPoseLandmarks(result)

        // save results in the sequence data for later action inference
        // 12 keypoints per pose
        // result.landmarks().size = 1 for 1 hand and '= 2' for 2 hands
        if (filteredPoseLandmarks != null){
            for ((landmarkCounter, value) in filteredPoseLandmarks.withIndex()){
                var pos = (frameKeypoints*frameCounterPose) + (landmarkCounter * 4)
                resultsSequence[pos] = value.x()
                resultsSequence[pos+1] = value.y()
                resultsSequence[pos+2] = value.z()
                resultsSequence[pos+3] = 1F // this is the visibility, it has to be removed from the model TODO
            }
        }
        //Log.i(TAG, "Size of hand landmarks: " + result.handednesses().toString() )

        frameCounterPose++
        framesCounter = (framesCounter % framesPerSequence) + 1

        if (frameCounterPose >= framesPerSequence){
            activeSequenceShift = true
            // subtract the iterator position of the sequence buffer
            // the '-1' is because indexes start at 0
            frameCounterPose = framesPerSequence - framesShiftInSequence
        }

        if (activeSequenceShift){
            // For every 6 frames send data to 'actions model' in python for inference
            // framesCounter is among 1 and framesPerSequence (inclusive)
            if (framesCounter == 6){
                // Create a new coroutine scope
                runBlocking {
                    // Code to be executed in the separate thread
                    Log.i(TAG, "Python run in another thread in kotlin from Pose")
                    var result = pythonModule.callAttr("runModel", resultsSequence).toInt()
                    Log.i(TAG, "The action is: " + actionsLabels[result])
                }
                // each frame shift the sequence buffer
                shiftLeftSequence(framesShiftInSequence)
            }
        }

        // send pose data for drawing
        armLandmarkerHelperListener?.onPoseResults(
            PoseResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width
            )
        )

        Log.i(TAG, "The landmarks sequence: " + this.resultsSequence.contentToString())
    }

    // Return errors thrown during detection to this ArmsLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        armLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    private fun shiftLeftSequence(shiftAmountFrame : Int = 1){
        val shift = shiftAmountFrame * this.frameKeypoints
        for (i in shift until this.resultsSequence.size) {
            val newIndex = i - shift
            this.resultsSequence[newIndex] = this.resultsSequence[i]
        }
    }

    // Creates an instance of the class
    companion object {
        const val TAG = "ArmsLandmarkerHelper"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_ARM_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_ARM_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_ARM_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_HANDS = 2
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        const val MODEL_POSE_LANDMARKER_FULL = 0
        const val MODEL_POSE_LANDMARKER_LITE = 1
        const val MODEL_POSE_LANDMARKER_HEAVY = 2

        // Returns only the needed keypoints from the pose landmarker
        fun filterPoseLandmarks(poseLandmarks : PoseLandmarkerResult): List<NormalizedLandmark>? {
            poseLandmarks?.let { poseLandmarkerResult ->
                if (poseLandmarkerResult.landmarks().isNotEmpty()) {
                    Log.i(TAG,"Pose landmarks size: " + poseLandmarkerResult.landmarks().first().size.toString())
                    return poseLandmarkerResult.landmarks().first().subList(11, 23) // from 11 to 23 keypoints included both
                }
            }
            return null
        }
    }

    data class HandResultBundle(
        val handResults: List<HandLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )
    data class PoseResultBundle(
        val poseResults: List<PoseLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )
    // Used only for gallery detections
    data class ArmResultBundle(
        val handResults: List<HandLandmarkerResult>,
        val poseResults: List<PoseLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )


    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onHandResults(handResultBundle: HandResultBundle)
        fun onPoseResults(poseResultBundle: PoseResultBundle)
    }
}


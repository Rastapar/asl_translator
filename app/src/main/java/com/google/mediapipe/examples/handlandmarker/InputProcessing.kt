package com.google.mediapipe.examples.handlandmarker.fragment

import android.content.Context
import android.util.Log
import com.google.mediapipe.examples.handlandmarker.ml.ModelOpt1
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.io.File
import android.content.res.AssetManager
import com.google.mediapipe.examples.handlandmarker.ml.Actionv2
import com.google.mediapipe.examples.handlandmarker.ml.Actionv3
import com.google.mediapipe.examples.handlandmarker.ml.ModelOpt2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader
//import com.chaquo.python.Python

class InputProcessing(
    private val context: Context?,
    //private var interpreter: Interpreter?,
) {

    /*fun loadModel() {
        val model_name = "model-opt1.tflite"
        val modelMappedByteBuffer = FileUtil.loadMappedFile(
            context!!,
            model_name
        )   // !! operator will throw an exception if the context is null
        val modelByteBuffer: ByteBuffer = modelMappedByteBuffer.duplicate()
        interpreter = Interpreter(modelByteBuffer)
    }*/

    fun setInterpreter() {
        /*  // GPU acceleration
        val compatList = CompatibilityList()

        val options = if(compatList.isDelegateSupportedOnThisDevice) {
            // if the device has a supported GPU, add the GPU delegate
            Model.Options.Builder().setDevice(Model.Device.GPU).build()
        } else {
            // if the GPU is not supported, run on 4 threads
            Model.Options.Builder().setNumThreads(4).build()
        }

        // Initialize the model as usual feeding in the options object
        val myModel = MyModel.newInstance(context, options)
        */

        // Run inference per sample code
        val model = ModelOpt1.newInstance(this.context!!)
        Log.i("Model Output", model.toString())

        // Process the input
        val listKeypoints:List<Float> = readFileToList("please.txt")
        val byteBufferKeypoints:ByteBuffer = floatsToByteBuffer(listKeypoints)

        Log.i("Input Processing", listKeypoints.size.toString())
        Log.i("Input byteBuffer", byteBufferKeypoints.getFloat(79772).toString())

        // Creates inputs for reference.
        // 1 sequence, 12 frames per sequence, 1662 keypoints per frame
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1,12,1662), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBufferKeypoints)

        Log.i("Input TensorBuffer", inputFeature0.getFloatValue(19943).toString())

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Print in Log the output
        // actions = ['please', 'thank you', 'you are welcome']
        Log.i("Output of Lite Model", outputFeature0.floatArray.contentToString())

        // Releases model resources if no longer used.
        model.close()
    }

    // Converts a List of floats to a byteBuffer
    // TODO algo falla cuando el modelo lee el byBuffer
    fun floatsToByteBuffer(floats: List<Float>): ByteBuffer {
        val bufferSize = floats.size * 4 // Each float takes 4 bytes
        val byteBuffer = ByteBuffer.allocate(bufferSize)
        byteBuffer.clear()

        floats.forEach { float ->
            byteBuffer.putFloat(float)
        }

        //byteBuffer.flip() // Prepare the ByteBuffer for reading
        byteBuffer.position(0)

        return byteBuffer
    }


    // Reads values from a file where each row is a float
    // Returns a List with them
    fun readFileToList(filename: String): List<Float> {
        val values = mutableListOf<Float>()
        val assetManager = this.context?.assets
        assetManager?.open(filename)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line: String? = reader.readLine()
                while (line != null) {
                    val floatValue = line.toFloatOrNull()
                    if (floatValue != null) {
                        values.add(floatValue)
                    }
                    line = reader.readLine()
                }
            }
        }
        Log.i("Input List File first value", values.size.toString())
        return values
    }



}
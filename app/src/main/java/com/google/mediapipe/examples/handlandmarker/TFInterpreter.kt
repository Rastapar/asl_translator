package com.google.mediapipe.examples.handlandmarker
/*
import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

class TFInterpreter (
    private var context: Context?
){
    fun setInterpreter(){
        val initializeTask: Task<Void> by lazy { TfLite.initialize(context!!) }
        lateinit var interpreter: InterpreterApi

        val model_name = "actionv2.tflite"
        val modelMappedByteBuffer = FileUtil.loadMappedFile(
            context!!,
            model_name
        )   // !! operator will throw an exception if the context is null
        val modelByteBuffer: ByteBuffer = modelMappedByteBuffer.duplicate()

        initializeTask.addOnSuccessListener {
            val interpreterOption =
                InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
            interpreter = InterpreterApi.create(
                modelByteBuffer,
                interpreterOption
            )
        }
        .addOnFailureListener { e ->
            Log.e("Interpreter", "Cannot initialize interpreter", e)
        }
    }
}

 */
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
import android.content.pm.PackageItemInfo
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarksConnections
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarksConnections
import java.util.*
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var resultsHand: HandLandmarkerResult? = null
    private var resultsPose: List<NormalizedLandmark>? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    // Vacia todos los atributos de esta clase
    fun clear() {
        resultsHand = null
        resultsPose = null
        linePaint.reset()
        pointPaint.reset()
        invalidate()
        initPaints()
    }

    // Inicializa los atributos de los colores de la clase
    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    /* A partir del atributo 'result' dibuja en la pantalla los puntos y
    * las uniones de los landmarks.
    * Esta funcion se llama cuando se llama la funcion invalidate() */
    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        resultsHand?.let { handLandmarkerResult ->  // llamamos handLandmarkerResult a la variable results dentro de los corchetes
            for (landmark in handLandmarkerResult.landmarks()) {
                for (normalizedLandmark in landmark) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        pointPaint
                    )
                }
                HandLandmarksConnections.HAND_CONNECTIONS.forEach {
                    canvas.drawLine(
                        landmark.get(it!!.start())
                            .x() * imageWidth * scaleFactor,
                        landmark.get(it.start())
                            .y() * imageHeight * scaleFactor,
                        landmark.get(it.end())
                            .x() * imageWidth * scaleFactor,
                        landmark.get(it.end())
                            .y() * imageHeight * scaleFactor,
                        linePaint
                    )
                }
            }
        }
        resultsPose?.let { poseLandmarkerResult ->
            Log.i("Overlay","Drawing with size " + poseLandmarkerResult.size.toString())
            if (poseLandmarkerResult.isNotEmpty()) {
                for (normalizedLandmark in poseLandmarkerResult) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        pointPaint
                    )
                }

                // Draw pose landmark connections
                val pose_connections : Set<Pair<Int, Int>> = setOf(
                    Pair(11,12),
                    Pair(11,13),
                    Pair(12,14),
                    Pair(13,15),
                    Pair(14,16),
                    Pair(15,17),
                    Pair(15,19),
                    Pair(15,21),
                    Pair(16,18),
                    Pair(16,20),
                    Pair(16,22),
                    Pair(17,19),
                    Pair(18,20),
                )

                pose_connections.forEach {
                    val firstPoint = it.first - 11
                    val secondPoint = it.second - 11
                    canvas.drawLine(
                        poseLandmarkerResult.get(firstPoint).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.get(firstPoint).y() * imageHeight * scaleFactor,
                        poseLandmarkerResult.get(secondPoint).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.get(secondPoint).y() * imageHeight * scaleFactor,
                        linePaint
                    )
                }


                /*PoseLandmarksConnections.POSE_LANDMARKS.forEach {
                    canvas.drawLine(
                        poseLandmarkerResult.get(it!!.start()).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.get(it.start()).y() * imageHeight * scaleFactor,
                        poseLandmarkerResult.get(it.end()).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.get(it.end()).y() * imageHeight * scaleFactor,
                        linePaint
                    )
                }*/
            }
        }
    }

    /* Funcion usada por la clase de la cámara para
    dar valores a los landmarks de las manos */
    fun setHandResults(
        handLandmarkerResults: HandLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        resultsHand = handLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    /* Funcion usada por la clase de la cámara para
    dar valores a los landmarks de los brazos */
    fun setPoseResults(
        poseLandmarkerResults: List<NormalizedLandmark>?,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        resultsPose = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
    }
}

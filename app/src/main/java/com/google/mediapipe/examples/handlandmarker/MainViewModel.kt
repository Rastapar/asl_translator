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

import androidx.lifecycle.ViewModel

/**
 *  This ViewModel is used to store hand landmarker helper settings
 */
class MainViewModel : ViewModel() {
    // hand private var
    private var _delegate: Int = ArmsLandmarkerHelper.DELEGATE_CPU
    private var _minArmDetectionConfidence: Float =
        ArmsLandmarkerHelper.DEFAULT_ARM_DETECTION_CONFIDENCE
    private var _minArmTrackingConfidence: Float = ArmsLandmarkerHelper
        .DEFAULT_ARM_TRACKING_CONFIDENCE
    private var _minArmPresenceConfidence: Float = ArmsLandmarkerHelper
        .DEFAULT_ARM_PRESENCE_CONFIDENCE
    private var _maxHands: Int = 2 //ArmsLandmarkerHelper.DEFAULT_NUM_HANDS

    // pose private var
    private var _poseModel = ArmsLandmarkerHelper.MODEL_POSE_LANDMARKER_FULL


    // hand set up
    val currentDelegate: Int get() = _delegate
    val currentMinArmDetectionConfidence: Float
        get() = _minArmDetectionConfidence
    val currentMinArmTrackingConfidence: Float
        get() = _minArmTrackingConfidence
    val currentMinArmPresenceConfidence: Float
        get() = _minArmPresenceConfidence
    val currentMaxHands: Int get() = _maxHands

    val currentPoseModel: Int get() = _poseModel


    fun setDelegate(delegate: Int) {
        _delegate = delegate
    }
    fun setMinArmDetectionConfidence(confidence: Float) {
        _minArmDetectionConfidence = confidence
    }
    fun setMinArmTrackingConfidence(confidence: Float) {
        _minArmTrackingConfidence = confidence
    }
    fun setMinArmPresenceConfidence(confidence: Float) {
        _minArmPresenceConfidence = confidence
    }
    fun setMaxHands(maxResults: Int) {
        _maxHands = maxResults
    }
    fun setPoseModel(model: Int) {
        _poseModel = model
    }
}

import os
from os.path import join, dirname
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

model = Sequential()

def main():
    print("Hello Android from python")

def tfVersion():
    print("python tf version ", tf.__version__)

def setModel():

    ############################## LOAD TF LITE MODEL
    # Path to the TFLite model file
    model_path = join(dirname(__file__), 'actionv2.tflite')

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Allocate the tensors
    interpreter.allocate_tensors()

    ################################## GET INPUT AND OUTPUT DETAILS
    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print the details
    print('Input details:', input_details[0]['shape'])
    print('Output details:', output_details)

    ############################ PREPARE INPUT DATA
    data_path = 'please.txt'
    data_path2 = 'thank.txt'
    data_path3 = 'welcome.txt'

    filename = join(dirname(__file__), data_path)

    # Create a input shaped array
    data_sequence = np.zeros((1, 12, 1662))

    with open(filenam, 'r') as file:
        for frame in range(12):
            for keypoint in range(1662):
                value = readLine()
                if (value):
                    data_sequence[0][frame][keypoint] = np.float(value)
                else:
                    break

    # Print the converted list
    print(len(data_sequence))
    print(data_sequence.shape)
    print(data_sequence[0][11][1661])

    ######################## RUN INFERENCE
    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Set printing format
    float_formatter = "{:.8f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    # Process the output data as needed
    print('Output:', output_data)


def convertModel():

    # Directory where the model is saved (in .h5 format)
    saved_model_dir = join(dirname(__file__), 'actionv5.tf')

    # Load the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Specifications
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    # Convertion
    tflite_model = converter.convert()

    # Directory where the lite-model is going to be saved
    lite_model_path = join(dirname(__file__), 'actionv5.tflite')

    # Save the model.
    with open(lite_model_path, 'wb') as f:
      f.write(tflite_model)

def loadWeights():
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)


    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(12,174)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    weight_path = join(dirname(__file__), 'actions3.h5')
    model.load_weights(weight_path)

    '''
    data_path = 'please.txt'
    data_path2 = 'thank.txt'
    data_path3 = 'welcome.txt'

    filename = join(dirname(__file__), data_path3)

    # Create a input shaped array
    data_sequence = np.zeros((1, 12, 1662))

    with open(filename, 'r') as file:
        for frame in range(12):
            for keypoint in range(1662):
                value = file.readline()
                if (value):
                    data_sequence[0][frame][keypoint] = np.float(value)
                else:
                    break

    # Print the converted list
    print(len(data_sequence))
    print(data_sequence.shape)
    print(data_sequence[0][11][1661])
    '''


def runModel(list_sequence):
    print("Data type of argument from kotlin to python ", type(list_sequence))
    np_sequence = np.array(list_sequence)
    print("python list is ", type(np_sequence))
    print("With len ", len(np_sequence))
    print("Random value", np_sequence[3])
    data_sequence = np_sequence.reshape(1, 12, 174).astype(float)

    res = model.predict(data_sequence)
    print("The result is: ", res)

    maxIndex = np.argmax(res)

    return maxIndex

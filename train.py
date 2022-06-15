import glob
import os
import hashlib
import argparse
from mkdir_p import mkdir_p

from PIL import Image

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

# Number of output neurons
OUT_SHAPE = 4

# Height and Width of model input. For RGB images, use 3 channels
INPUT_WIDTH = 200#256
INPUT_HEIGHT = 78#192 - 66
INPUT_CHANNELS = 3

# Percentage of validation split. Set to a higher value when you have large training data.
VALIDATION_SPLIT = 0.15
# Data augmentation not tested
USE_REVERSE_IMAGES = False

def customized_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    val = K.mean(K.square((y_pred - y_true)), axis=-1)
    return val
    
# Create CNN, check kep_prob parameter since it controls dropout layers
def create_model(keep_prob=0.6):
    # Keras sequential model
    model = Sequential()

    # Input layer with defined size
    model.add(BatchNormalization(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))

    # Convolutional layers
    """ model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) """

    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    # Dense layers
    model.add(Flatten())
    """ model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out)) """

    model.add(Dense(4000, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(4000, activation='relu'))
    model.add(Dropout(drop_out))

    # Output layer
    model.add(Dense(OUT_SHAPE, activation='sigmoid', name="predictions"))

    return model

# Function to mix validation and training sets
def is_validation_set(string):
    string_hash = hashlib.md5(string.encode('utf-8')).digest()
    return int.from_bytes(string_hash[:2], byteorder='big') / 2 ** 16 > VALIDATION_SPLIT

# Load images and inputs files from recordings folder
def load_training_data():
    X_train, y_train = [], []
    X_val, y_val = [], []

    #images = []
    all_inputs = []

    # Recordings folder
    recordings = glob.iglob("recordings//*")
    # Iterates every folder on recordings
    i=1
    for recording in recordings:
        images = []
        # Check for all png files in folder
        filenames = list(glob.iglob('{}/*.png'.format(recording)))
        filenames.sort(key=lambda f: int(os.path.basename(f)[:-4]))

        for file in filenames:
            im = Image.open(file).resize((INPUT_WIDTH, INPUT_HEIGHT))
            im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
            images.append(im_arr)

        # Load inputs file and saves every line on vector
        inputs_list = [[int(l) for l in line] for line in open(("{}/inputs.txt").format(recording)).read().splitlines()]
        all_inputs.extend(inputs_list)

        # The number of lines on inputs file should be equal to the number of png files on folder
        assert len(filenames) == len(
            inputs_list), "For recording %s, the number of inputs values does not match the number of images." % recording

        """ # Now we're iterating for every pair of image and input line
        for file, inputs in zip(filenames, inputs_list):
            # Use as validation set if this function is true
            valid = is_validation_set(file)

            # Process image and convert to input array
            im = Image.open(file).resize((INPUT_WIDTH, INPUT_HEIGHT))
            im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

            if valid:
                X_train.append(im_arr)
                y_train.append(inputs)
            else:
                X_val.append(im_arr)
                y_val.append(inputs) """

        """ print(len(images))
        images = np.asarray(images)
        inputs_list = np.asarray(inputs_list) """

        if i==9 or i==10 or i==19 or i==20: #TEMPORAL
            X_val.extend(images)
            y_val.extend(inputs_list)
        else:
            X_train.extend(images)
            y_train.extend(inputs_list)
        i+=1

    #X_train, X_val, y_train, y_val = train_test_split(images, all_inputs, test_size=0.20, random_state=27)

    # Check for missing images or missing lines on inputs files
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # Process for input layer
    return np.asarray(X_train), \
           np.asarray(y_train), \
           np.asarray(X_val), \
           np.asarray(y_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load Training Data
    X_train, y_train, X_val, y_val = load_training_data()

    print(X_train.shape[0], 'training samples.')
    print(X_val.shape[0], 'validation samples.')

    # Training loop variables
    epochs = 100
    batch_size = 20

    model = create_model()

    mkdir_p("weights")
    weights_file = "weights/{}.hdf5".format(args.model)
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    #customized_loss
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    checkpointer = ModelCheckpoint(
        monitor='val_loss', filepath=weights_file, verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, validation_data=(X_val, y_val), callbacks=[checkpointer, earlystopping])
    model.save("weights/{}.hdf5".format(args.model))

# Import the necessary libraries

import matplotlib
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io
from numpy import fliplr

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

print("Versions of key libraries")
print("---")
print("tensorflow: ", tf.__version__)
print("numpy:      ", np.__version__)
print("matplotlib: ", matplotlib.__version__)
print("sklearn:    ", sklearn.__version__)


# Design the model layers
def createModel():
  model = Sequential()
  input_data = (512, 512, 1)
  model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_data, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  model.add(Conv2D(128, (2, 2), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(256, (2, 2), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(2))
  model.compile(loss='mse',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
  return model


# generator
def generate_for_kp(file_list, label_list, batch_size):
    while True:
        count = 0
        x, y = [], []
        for i, path in enumerate(file_list):
            if path.startswith('*'):
                path = path.strip('*#')
                # img=cv2.imread('E:/cat/original_images/'+path, cv2.IMREAD_GRAYSCALE)
                img = io.imread('E:/cat/original_images/'+path, as_gray=True)
                img = fliplr(img)
            else:
                path = path.strip('#')
                img = io.imread('E:/cat/original_images/'+path, as_gray=True)
            img = np.array(img)
            count += 1
            x.append(img)
            y.append((label_list[i, :]-256.0)/256.0)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 512, 512, 1).astype("float32")
                y = np.array(y)
                yield x, y
                x, y = [], []


# Process the original data
# ----
# - Step 1: Load the images and labels file.
# - Step 2: Increase the training set by flipping the picture horizontally and merge all the data and shuffle it randomly (I'm not sure if it's easier to overfit if we train with similar images back-to-back, so I shuffle it randomly)
# - Step 3: Divide training and test sets
# ----

left_names = [
    'Left_Eye_(Left)',
    'Left_Eye(Top)',
    'Left_Eye_(Right)',
    'Left_Eye(Bottom)',
    'Right_Eye_(Left)',
    'Right_Eye(Top)',
    'Right_Eye_(Right)',
    'Right_Eye(Bottom)',
    'Nose',
    'Lip_Left',
    'Upper_Lip',
    'Lip_Right',
    'Lower_Lip']
right_names = [
    'Right_Eye_(Right)',
    'Right_Eye(Top)',
    'Right_Eye_(Left)',
    'Right_Eye(Bottom)',
    'Left_Eye_(Right)',
    'Left_Eye(Top)',
    'Left_Eye_(Left)',
    'Left_Eye(Bottom)',
    'Nose',
    'Lip_Right',
    'Upper_Lip',
    'Lip_Left',
    'Lower_Lip']

# definde the size of this image
image_size_x = 512
image_size_y = 512
label_path = 'E:/cat/labeling_images/label.csv'
kp_df = pd.read_csv(label_path, header=0)

# Train a separate model for each checkpoint
for count_name in range(0, len(left_names)):
    left_name = left_names[count_name]
    right_name = right_names[count_name]
    # create the model
    modelname = 'CNN_' + left_name
    model = createModel()

    # Image enhancement
    left_kp = kp_df[['image_name', left_name+'_x', left_name+'_y']]
    left_kp.dropna(axis=0, how='any', inplace=True)
    right_kp = kp_df[['image_name', right_name+'_x', right_name+'_y']]
    right_kp.dropna(axis=0, how='any', inplace=True)
    right_kp['image_name'] = "*" + right_kp["image_name"]
    right_kp[right_name+'_x'] = image_size_x - right_kp[right_name+'_x']
    right_kp = right_kp.rename(columns={right_name+'_x': left_name + '_x', right_name+'_y': left_name + '_y'})
    all_data = shuffle(pd.concat([left_kp,right_kp], ignore_index=True), random_state=2).reset_index(drop=True)

    # Training set test set division
    X = all_data.image_name
    Y = all_data.drop(['image_name'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/5, random_state=1)
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # create the callbacks
    folderpath = 'model_sum/'
    filepath = folderpath + modelname + ".hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')

    # create the log
    csv_logger = CSVLogger(folderpath + modelname + '.csv')

    # call_backs setting
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    callbacks_list = [checkpoint, early_stop, csv_logger]
    print("Callbacks created:")
    print(callbacks_list[0])
    print(callbacks_list[1])
    print('')
    print("Path to model:", filepath)
    print("Path to log:  ", folderpath + modelname + '.csv')

    # model training
    model.fit_generator(generate_for_kp(X_train, Y_train, 64),
                        steps_per_epoch=int(len(X_train) / 64) + 1,
                        epochs=1000, verbose=1,
                        validation_data=generate_for_kp(X_test, Y_test, 64),
                        validation_steps=int(len(X_test) / 64) + 1,
                        callbacks=callbacks_list)

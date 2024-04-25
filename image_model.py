import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from constants import path_image, path_label

from data_preprocess import read_data, INPUT_IMG_SIZE

from additinal_functions import show_confusion_matrix

df = pd.read_csv(path_label)

df['number'] -= 1

x = df['text']
y = df['label']
n = df['number']

x_train, x_test, y_train, y_test, n_train, n_test = train_test_split(x, y, n, test_size=0.2, random_state=33)
x_train, x_val, y_train, y_val, n_train, n_val = train_test_split(x_train, y_train, n_train, test_size=0.2, random_state=33)

img_train, text_train, label_train = read_data(path_image, n_train)
img_val, text_val, label_val = read_data(path_image, n_val)
img_test, text_test, label_test = read_data(path_image, n_test)


def create_cnn_model(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_shape)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    dropout = Dropout(0.25)(pool2)
    flatten = Flatten()(dropout)

    dense = Dense(64, activation='relu')(flatten)

    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model


input_shape = Input((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3))

model_cnn = create_cnn_model(input_shape)

model_cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history_cnn = model_cnn.fit(
    img_train, label_train,
    batch_size=32,
    epochs=3,
    validation_data=(img_val, label_val)
)

model_cnn.evaluate( img_test, label_test)

pred = model_cnn.predict(img_test)
pred = pred > 0.5

show_confusion_matrix(y_test, pred)

metrics.classification_report(y_test, pred)
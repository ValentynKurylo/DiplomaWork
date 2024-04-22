import numpy as np
import pandas as pd

from transformers import BertTokenizer, TFBertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Embedding, LSTM, Concatenate
from keras.models import Model

import tensorflow as tf

from constants import path_image, path_label, TEXT_MAX_LENGTH, INPUT_IMG_SIZE, bert_model_name, fast_text

from data_preprocess import preprocess_text, read_data, preprocess_bert_data, preprocess_lstm_data, fit_tokenizer, create_embedding

from additinal_finctions import show_confusion_matrix

from threshold_algorithm import get_threshold_values


def create_multimodal_model(input_ids, attention_mask, image_input, text_input_lstm, word_index, fasttext_model, embedding_matrix, bert_model ):
    bert_output = bert_model([input_ids, attention_mask])[0]
    dense_layer = Dense(32, activation="relu")(bert_output)

    conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(image_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flatten_image = Flatten()(pool2)
    dense_layer_cnn = Dense(32, activation="relu")(flatten_image)

    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=fasttext_model.vector_size,
                                weights=[embedding_matrix],
                                input_length=TEXT_MAX_LENGTH,
                                trainable=False)(text_input_lstm)
    lstm_output = LSTM(64, activation='relu')(embedding_layer)
    dense_layer_lstm = Dense(32, activation='relu')(lstm_output)

    concatenated = Concatenate()([dense_layer, dense_layer_cnn, dense_layer_lstm])

    fc1 = Dense(64, activation='relu')(concatenated)
    dropout1 = Dropout(0.6)(fc1)
    output = Dense(1, activation='sigmoid')(fc1)

    model = Model(inputs=[[input_ids, attention_mask], image_input, text_input_lstm], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def make_prediction_and_get_results(model, label_test, test_seq_tf, test_mask_tf, img_test, sequences_test):
    prediction = model.predict([[test_seq_tf, test_mask_tf], img_test, sequences_test])
    prediction = prediction > 0.5

    show_confusion_matrix(label_test, prediction)

    metrics.classification_report(label_test, prediction)

    return prediction


def get_results_with_coefficients(label_test, pred):
    test_true, test_pred = [], []
    k1, k2 = get_threshold_values(pred, label_test, df)
    for i in range(len(pred)):
        if pred[i] >= k2:
            test_pred.append(1)
            test_true.append(label_test[i])
        elif pred[i] <= k1:
            test_pred.append(0)
            test_true.append(label_test[i])

    show_confusion_matrix(test_true, test_pred)

    metrics.classification_report(test_true, test_pred)


df = pd.read_csv(path_label)

df['number'] -= 1

x = df['text']
y = df['label']
n = df['number']

x = x.apply(preprocess_text)

x_train, x_test, y_train, y_test, n_train, n_test = train_test_split(x, y, n, test_size=0.2, random_state=33)
x_train, x_val, y_train, y_val, n_train, n_val = train_test_split(x_train, y_train, n_train, test_size=0.2, random_state=33)

img_train, text_train, label_train = read_data(path_image, n_train, df)

img_val, text_val, label_val = read_data(path_image, n_val, df)

img_test, text_test, label_test = read_data(path_image, n_test, df)

train_seq_tf, train_mask_tf = preprocess_bert_data(text_train)
val_seq_tf, val_mask_tf = preprocess_bert_data(text_val)
test_seq_tf, test_mask_tf = preprocess_bert_data(text_test)

tokenizer = fit_tokenizer(x_train)
sequences_train = preprocess_lstm_data(text_train, tokenizer)
sequences_val = preprocess_lstm_data(text_val, tokenizer)
sequences_test = preprocess_lstm_data(text_test, tokenizer)

embedding_matrix, fasttext_model, word_index = create_embedding(tokenizer)

input_shape = img_train[0].shape
image_input = Input(shape=input_shape, name='image_input')

input_ids = Input(shape=(TEXT_MAX_LENGTH,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(TEXT_MAX_LENGTH,), dtype=tf.int32, name="attention_mask")

text_input_lstm = Input(shape=(TEXT_MAX_LENGTH,))

bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_name)

model = create_multimodal_model(input_ids, attention_mask, image_input, text_input_lstm, word_index, fasttext_model, embedding_matrix, bert_model)

history = model.fit(
        [[train_seq_tf, train_mask_tf], img_train, sequences_train], label_train,
        batch_size=32,
        epochs=2,
        validation_data=([[val_seq_tf, val_mask_tf], img_val, sequences_val], label_val)
    )

model.evaluate([[test_seq_tf, test_mask_tf], img_test, sequences_test], label_test)

prediction = make_prediction_and_get_results(model, label_test, test_seq_tf, test_mask_tf, img_test, sequences_test)

get_results_with_coefficients(label_test, prediction)

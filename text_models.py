import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, Input

from transformers import BertTokenizer, TFBertForSequenceClassification

import tensorflow as tf

from constants import path_data, TEXT_MAX_LENGTH, bert_model_name

from data_preprocess import preprocess_text, fit_tokenizer, preprocess_lstm_data, create_embedding, preprocess_bert_data

from additinal_functions import show_confusion_matrix

df = pd.read_csv(path_data)

x, y = df['text'], df['label']

x = x.apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

vectorizer_count = CountVectorizer(max_features=5000, ngram_range=(1,1))

x_train = vectorizer_count.fit_transform(x_train)
x_test = vectorizer_count.transform(x_test)

####################################### LogisticRegression ##########################################

model_logistic_regression = LogisticRegression(solver='liblinear', C=0.7, penalty="l1")
model_logistic_regression.fit(x_train, y_train)

model_logistic_regression.score(x_test, y_test)

pred_logistic_regression = model_logistic_regression.predict(x_test)
pred_logistic_regression = pred_logistic_regression > 0.5

show_confusion_matrix(y_test, pred_logistic_regression)

metrics.classification_report(y_test, pred_logistic_regression)

####################################### SVM #########################################################

model_svc = SVC(kernel='linear', C=0.7)
model_svc.fit(x_train, y_train)

model_svc.score(x_test, y_test)

pred_svc = model_svc.predict(x_test)
pred_svc = pred_svc > 0.5

show_confusion_matrix(y_test, pred_svc)

metrics.classification_report(y_test, pred_svc)

####################################### LSTM #########################################################

tokenizer = fit_tokenizer(x_train)
sequences_train = preprocess_lstm_data(x_train, tokenizer)
sequences_test = preprocess_lstm_data(x_test, tokenizer)

embedding_matrix, fasttext_model, word_index = create_embedding(tokenizer)

def create_latm_model():
  return Sequential([
    Embedding(input_dim=len(word_index) + 1,
                    output_dim=fasttext_model.vector_size,
                    weights=[embedding_matrix],
                    input_length=TEXT_MAX_LENGTH,
                    trainable=False),
    LSTM(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])


model_lstm = create_latm_model()

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_lstm = model_lstm.fit(sequences_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model_lstm.evaluate(sequences_test, y_test)

pred_lstm = model_lstm.predict(x_test)
pred_lstm = pred_lstm > 0.5

show_confusion_matrix(y_test, pred_lstm)

metrics.classification_report(y_test, pred_lstm)

####################################### BERT #########################################################

train_seq_tf, train_mask_tf = preprocess_bert_data(x_train)
test_seq_tf, test_mask_tf = preprocess_bert_data(x_test)

bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_name)

def create_bert_model(bert_model):
    input_ids = Input(shape=(TEXT_MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(TEXT_MAX_LENGTH,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model([input_ids, attention_mask])[0]
    dense_layer = Dense(32, activation="relu")(bert_output)
    dropout_layer = Dropout(0.3)(dense_layer)
    output_layer = Dense(1, activation="sigmoid")(dropout_layer)

    model = Model(inputs=[input_ids, attention_mask], outputs=output_layer)

    return model


model_bert = create_bert_model(bert_model)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model_bert.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

history_bert = model_bert.fit(
    [train_seq_tf, train_mask_tf],
    y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
)

model_bert.evaluate([test_seq_tf, test_mask_tf], y_test)

pred_bert = model_bert.predict([test_seq_tf, test_mask_tf])
pred_bert = pred_bert > 0.5

show_confusion_matrix(y_test, pred_bert)

metrics.classification_report(y_test, pred_bert)
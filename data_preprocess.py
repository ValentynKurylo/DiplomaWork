import numpy as np
from PIL import Image

import re
import nltk
from nltk.corpus import stopwords

from gensim.models import FastText

from transformers import BertTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from constants import INPUT_IMG_SIZE, bert_model_name, TEXT_MAX_LENGTH, fast_text

def preprocess_text(text):
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)


def read_data(path, data, df):
  dataset = np.empty([len(data), INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3])
  label = np.empty(len(data))
  text = []
  j = 0

  for i in data:
      image = Image.open(path + '/' + str(i) + '.png')
      image = image.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE))
      if image.mode != 'RGB':
          image = image.convert('RGB')
      image = np.array(image)
      image = image / 255.0
      label[j] = df['label'][i]
      dataset[j] = image
      text.append(df['text'][i])
      j = int(j + 1)

  return dataset, text, label


def preprocess_bert_data(text_data):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    tokens = tokenizer.batch_encode_plus(
        text_data,
        add_special_tokens=True,
        max_length=TEXT_MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True,
        return_tensors="pt"
    )

    seq_tf = tf.convert_to_tensor(tokens['input_ids'].numpy(), dtype=tf.int32)
    mask_tf = tf.convert_to_tensor(tokens['attention_mask'].numpy(), dtype=tf.int32)

    return seq_tf, mask_tf


def fit_tokenizer(x_train):
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(x_train)
    return tokenizer


def preprocess_lstm_data(text_data, tokenizer):
    sequences = tokenizer.texts_to_sequences(text_data)
    return pad_sequences(sequences, maxlen=TEXT_MAX_LENGTH)


def create_embedding(tokenizer):
    fasttext_model = FastText.load_fasttext_format(fast_text)

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, fasttext_model.vector_size))

    for word, i in word_index.items():
        if word in fasttext_model.wv:
            embedding_vector = fasttext_model.wv[word]
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, fasttext_model, word_index
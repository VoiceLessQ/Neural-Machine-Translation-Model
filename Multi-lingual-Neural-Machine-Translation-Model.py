import csv
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from nltk.translate.bleu_score import corpus_bleu
import os
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers
import random
import string
import re

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

def load_glob_embedding(num_words, embed_size=100, word_index=None):
    from numpy import asarray
    from numpy import zeros

    embeddings_dictionary = dict()
    glove_file = open('Multi-lingual-Neural-Machine-Translation-Model.py/glove/glove.6B.'+str(embed_size)+'d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((num_words, embed_size))
    for index, word in enumerate(word_index):
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix
    
english_sentences = load_data('data\danish.txt')
danish_sentences = load_data('data\danish.txt')

text_pairs = []
for english,danish, in zip(english_sentences, danish_sentences):
    english = "[starten] " + english + " [enden]"
    danish = "[startdk] " + danish + " [enddk]"


    text_pairs.append((english, danish))

## Here, we define the constants 

embed_dim = 200
latent_dim = 1024
vocab_size = 30000
sequence_length = 20
batch_size = 64

print(random.choice(text_pairs))

random.shuffle(text_pairs)

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

## Preprocess the dataset to remove unneccessary tokens.

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_source_texts = [pair[0] for pair in train_pairs]
train_target_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_source_texts)
target_vectorization.adapt(train_target_texts)

def format_dataset(eng, dk):
    eng = source_vectorization(eng)
    dk = target_vectorization(dk)
    return ({
        "source": eng,
        "target": dk[:, :-1],
    }, dk[:, 1:])

def make_dataset(pairs):
    eng_texts, dk_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    dk_texts = list(dk_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, dk_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['source'].shape: {inputs['source'].shape}")
    print(f"inputs['target'].shape: {inputs['target'].shape}")
    print(f"targets.shape: {targets.shape}")

import tensorflow.keras as keras
from tensorflow.keras import layers

embedding_matrix = load_glob_embedding(vocab_size, 200, target_vectorization.get_vocabulary())

source = keras.Input(shape=(None,), dtype="int64", name="source")
x = layers.Embedding(vocab_size, embed_dim, weights=[embedding_matrix], mask_zero=True,
                     name='embed_encoder', trainable=False)(source)
encoded_source = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_encoder1')(x)
encoded_source = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_encoder2')(encoded_source)
encoded_source = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_encoder3')(encoded_source)
_, *encoder_states = layers.LSTM(latent_dim, activation='relu', return_state=True, name='rnn_encoder4')(encoded_source)

past_target = keras.Input(shape=(None,), dtype="int64", name="target")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name='embed_decoder')(past_target)

decoder_rnn = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_decoder1')
x = decoder_rnn(x, initial_state=encoder_states)
x = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_decoder2')(x)
x = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_decoder3')(x)
x = layers.LSTM(latent_dim, return_sequences=True, activation='relu', name='lstm_decoder4')(x)

x = layers.Dropout(0.5)(x)

target_next_step = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax", name='output'))(x)

seq2seq_rnn = keras.Model([source, past_target], target_next_step)

seq2seq_rnn.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

seq2seq_rnn.summary()

seq2seq_rnn.fit(train_ds, epochs=10, validation_data=val_ds)

seq2seq_rnn.save('Multi-lingual-Neural-Machine-Translation-Model.py/english_danish.h5')
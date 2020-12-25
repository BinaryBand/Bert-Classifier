import os
import json

import pandas as pd
import numpy as np

import tensorflow as tf

# Load the required submodules
from official.nlp import bert
from official.nlp import optimization
import official.nlp.bert.tokenization
import official.nlp.bert.configs
import official.nlp.bert.bert_models


checkpoint_dir = "uncased_L-12_H-768_A-12"


# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(checkpoint_dir, "vocab.txt"),
    do_lower_case=True)


def encode_sentence(sentence):
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def bert_encode(sentences, max_len=512):
    sentences = [s[:max_len] for s in sentences]
    input_word_ids = tf.ragged.constant([encode_sentence(s) for s in np.array(sentences)]).to_tensor()
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    input_type_ids = tf.zeros_like(input_word_ids).to_tensor()
    return [input_word_ids, input_mask, input_type_ids]


bert_config_file = open(os.path.join(checkpoint_dir, "bert_config.json"))
config_dict = json.loads(bert_config_file.read())
bert_config = bert.configs.BertConfig.from_dict(config_dict)


bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)


checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(os.path.join(checkpoint_dir, "bert_model.ckpt"))


# Load data from the dataset
df = pd.read_csv("data.csv")
train_data, train_labels = bert_encode(df["text"]), np.array([df["label"]])

# Set up epochs and steps
num_epochs = 3
batch_size = 16
train_data_size = len(train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * num_epochs
warmup_steps = int(num_epochs * train_data_size * 0.1 / batch_size)


# creates an optimizer with learning rate schedule
optimizer = optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)


bert_classifier.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy", dtype=tf.float32)])


bert_classifier.fit(
    train_data, train_labels,
    batch_size=batch_size,
    validation_split=0.2,
    epochs=num_epochs)


my_examples = bert_encode(["I hate this movie", "Do not watch this. It's bad.", "Fantastic film!", "Very nice."])
results = bert_classifier(my_examples, training=False)
for res in results: print(np.argmax(res), res.numpy())

dataset_name = "imdb"
tf.saved_model.save(bert_classifier, export_dir=f"trained_models/{dataset_name}")
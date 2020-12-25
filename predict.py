import numpy as np
import tensorflow as tf

from official.nlp import bert
import official.nlp.bert.tokenization


tokenizer = bert.tokenization.FullTokenizer(
    vocab_file="uncased_L-12_H-768_A-12/vocab.txt",
    do_lower_case=True)


def encode_sentence(sentence):
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def bert_encode(sentences, max_len=512):
    sentences = [s[:max_len] for s in sentences]
    input_word_ids = tf.ragged.constant([encode_sentence(s) for s in np.array(sentences)])
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    input_type_ids = tf.zeros_like(input_word_ids).to_tensor()
    return [input_word_ids.to_tensor(), input_mask, input_type_ids]


bert_classifier = tf.saved_model.load("trained_model")


my_examples = bert_encode(["A mediocre waste of time.", "This movie is bad.", "I loved this film.", "Fantastic film! I have never seen such art!!"])
results = bert_classifier(my_examples, training=False)
for res in results: print(np.argmax(res), res.numpy())
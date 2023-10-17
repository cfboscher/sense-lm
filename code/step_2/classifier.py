import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import RobertaConfig, TFRobertaModel

import numpy as np
import random as rn


##fixing numpy RS
np.random.seed(42)

##fixing tensorflow RS
tf.random.set_seed(42)

##python RS
rn.seed(12)


def build_model(config):

    MAX_LEN = config.tokenizer_max_len_step2
    # below three variable will going to feed into the TFRoberta model
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained('step_2/roberta-base/config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained('step_2/roberta-base/roberta-base.h5', config=config)
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    x1 = tf.keras.layers.Dropout(0.05)(x[0])
    #     x1 = x[0]
    x1 = tf.keras.layers.Conv1D(filters=1, kernel_size=1,
                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45))(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    x2 = tf.keras.layers.Dropout(0.05)(x[0])
    #     x2 = x[0]
    x2 = tf.keras.layers.Conv1D(filters=1, kernel_size=1,
                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45))(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model
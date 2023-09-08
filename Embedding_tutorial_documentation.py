# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:53:34 2022

@author: cresis
"""

import tf
import keras_nlp

seq_length = 50
vocab_size = 5000
embed_dim = 128

inputs = tf.keras.Input(shape=(seq_length,))
# inputs.shape -> TensorShape([None, 50])

# Notes
# Embedding inputs -> vocab_size ( the max value expected or the dimension of input?? Please confirm)
# From documentation input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
#                    output_dim: Integer. Dimension of the dense embedding.

token_embeddings = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_dim
)(inputs)

# Output
# token_embeddings.shape
# TensorShape([None, 50, 128])

position_embeddings = keras_nlp.layers.PositionEmbedding(
    sequence_length=seq_length
)(token_embeddings)
# I'm not sure I can do this because it only allows size (Batch x )

outputs = token_embeddings + position_embeddings
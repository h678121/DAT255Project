import json
import os
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
import keras.ops as ops
from keras.layers import TextVectorization
import pandas as pd

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.token_embeddings = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports_masking = True

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs,0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.bool)
            mask = tf.tile(mask, [1, tf.shape(inputs)[1], 1])
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        out1 = self.layernorm1(inputs + attn_output)
        proj_output = self.dense_proj(out1)
        return self.layernorm2(out1 + proj_output)

@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "model.keras",
        custom_objects={
            "TransformerEncoder":TransformerEncoder,
            "PositionalEmbedding": PositionalEmbedding
        }
        )
    return model

@st.cache_resource
def load_vectorizer():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vectorizer = TextVectorization(
        max_tokens=20000,
        output_mode="int",
        output_sequence_length=300,
        vocabulary=vocab
    )

    return vectorizer

model=load_model()
vectorizer = load_vectorizer()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


label_names = [
    'toxicity',
    'severe_toxicity',
    'obscene',
    'threat',
    'insult',
    'identity_attack',
    'sexual_explicit'
]

def predict(text):
    x = vectorizer([text])
    preds = model.predict(x)[0]
    return { label_names[i]: float(preds[i]) 
            for i in range(len(label_names))
    }

st.title("Toxicity Classification Model")
st.write("Write down a comment and the model will try to predict a score of how toxic the comment is.")
single_comment_input=st.text_input("Comment","comment here")
if st.button("Run"):
    result = predict(single_comment_input)
    st.write(result)
st.write("Find out the average toxicity of a list of comments. This uses the JSON file in the project folder")
if st.button("Run file"):
    file=load_json("comments_file.JSON")
    results=[]
    for item in file:
        text = item["text"]
        pred = predict(text)
        results.append(pred)
    st.write(results)
    df = pd.DataFrame(results)
    avg_scores = df.mean(numeric_only=True)
    st.write("The average toxicity score for this comments is:")
    st.write(avg_scores)
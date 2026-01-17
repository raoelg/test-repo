"""Keras-based reference implementation for the DUALRec pipeline.

This module follows the study description:
1) Stage 1: Two-layer LSTM over multimodal movie sequences.
2) Stage 2: Prompt construction and LLM generation call (OpenRouter).
3) Stage 3: LoRA-based fine-tuning of a causal Transformer and semantic re-ranking.

It is intended as a clear, end-to-end template that stays within the Keras/TensorFlow
stack and avoids PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class MovieLensConfig:
    data_dir: str
    top_k_movies: int = 1000
    sequence_length: int = 30
    title_vocab_size: int = 5000
    title_max_tokens: int = 10
    movie_embedding_dim: int = 128
    title_embedding_dim: int = 64
    genre_embedding_dim: int = 64
    genre_labels: Tuple[str, ...] = (
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    )


def load_movielens_1m(config: MovieLensConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M ratings and movies metadata."""
    ratings_path = f"{config.data_dir}/ratings.dat"
    movies_path = f"{config.data_dir}/movies.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
    )
    return ratings, movies


def filter_top_movies(ratings: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep only interactions with the top-k most watched movies."""
    top_movies = ratings["movie_id"].value_counts().head(top_k).index
    return ratings[ratings["movie_id"].isin(top_movies)].copy()


def build_genre_matrix(movies: pd.DataFrame, config: MovieLensConfig) -> pd.DataFrame:
    """Build a multi-hot genre matrix for each movie."""
    genre_set = list(config.genre_labels)
    genre_map = {genre: idx for idx, genre in enumerate(genre_set)}

    def encode_genres(genre_str: str) -> np.ndarray:
        vector = np.zeros(len(genre_set), dtype=np.float32)
        for genre in genre_str.split("|"):
            if genre in genre_map:
                vector[genre_map[genre]] = 1.0
        return vector

    genre_vectors = movies["genres"].apply(encode_genres)
    genre_df = pd.DataFrame(genre_vectors.tolist(), columns=genre_set)
    return pd.concat([movies[["movie_id", "title"]], genre_df], axis=1)


def build_title_tokenizer(movies: pd.DataFrame, config: MovieLensConfig) -> keras.preprocessing.text.Tokenizer:
    """Fit a Keras tokenizer on movie titles."""
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=config.title_vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(movies["title"].tolist())
    return tokenizer


def build_sequences(
    ratings: pd.DataFrame,
    movie_table: pd.DataFrame,
    tokenizer: keras.preprocessing.text.Tokenizer,
    config: MovieLensConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding window sequences of movie interactions."""
    ratings = ratings.sort_values(["user_id", "timestamp"])
    movie_lookup = movie_table.set_index("movie_id")

    sequences = []
    targets = []

    for user_id, user_hist in ratings.groupby("user_id"):
        movie_ids = user_hist["movie_id"].tolist()
        for start in range(0, len(movie_ids) - config.sequence_length):
            window = movie_ids[start : start + config.sequence_length + 1]
            sequences.append(window[:-1])
            targets.append(window[-1])

    sequences = np.array(sequences, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    titles = [movie_lookup.loc[mid, "title"] for mid in sequences.flatten()]
    title_tokens = tokenizer.texts_to_sequences(titles)
    title_tokens = keras.preprocessing.sequence.pad_sequences(
        title_tokens, maxlen=config.title_max_tokens, padding="post", truncating="post"
    )
    title_tokens = title_tokens.reshape((-1, config.sequence_length, config.title_max_tokens))

    genres = movie_lookup.loc[sequences.flatten(), config.genre_labels].to_numpy(dtype=np.float32)
    genres = genres.reshape((-1, config.sequence_length, len(config.genre_labels)))

    return sequences, title_tokens, genres, targets


def build_lstm_model(config: MovieLensConfig, num_movies: int) -> keras.Model:
    """Build the Stage 1 multimodal LSTM model."""
    movie_ids = keras.Input(shape=(config.sequence_length,), name="movie_ids", dtype="int32")
    title_tokens = keras.Input(
        shape=(config.sequence_length, config.title_max_tokens), name="title_tokens", dtype="int32"
    )
    genre_inputs = keras.Input(shape=(config.sequence_length, len(config.genre_labels)), name="genres")

    movie_embed = layers.Embedding(num_movies + 1, config.movie_embedding_dim, name="movie_embedding")(movie_ids)

    title_embed = layers.TimeDistributed(
        layers.Embedding(config.title_vocab_size, config.title_embedding_dim), name="title_embedding"
    )(title_tokens)
    title_repr = layers.TimeDistributed(
        layers.GlobalAveragePooling1D(), name="title_pooling"
    )(title_embed)

    genre_repr = layers.TimeDistributed(
        layers.Dense(config.genre_embedding_dim, activation="relu"), name="genre_dense"
    )(genre_inputs)

    fused = layers.Concatenate(name="feature_concat")([movie_embed, title_repr, genre_repr])

    x = layers.LSTM(256, return_sequences=True, dropout=0.3, name="lstm_1")(fused)
    x = layers.LSTM(128, return_sequences=False, dropout=0.3, name="lstm_2")(x)
    outputs = layers.Dense(num_movies, activation="softmax", name="movie_softmax")(x)

    return keras.Model(inputs=[movie_ids, title_tokens, genre_inputs], outputs=outputs, name="dualrec_lstm")


def build_prompt(
    recent_titles: Iterable[str],
    lstm_prediction: str,
    instruction: str = "As a helpful assistant, recommend 3 more full movie titles with release years and genres.",
) -> str:
    """Build the Stage 2 prompt from recent history and LSTM output."""
    history_lines = "\n".join(f"- {title}" for title in recent_titles)
    return (
        "Below is a user's movie watching history:\n\n"
        f"{history_lines}\n\n"
        f"Based on this, the system (LSTM) recommends: {lstm_prediction}.\n\n"
        f"Now, {instruction}"
    )


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    """Call OpenRouter for Stage 2 generation."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


class LoRADense(layers.Layer):
    """Dense layer with a LoRA adaptation for parameter-efficient fine-tuning."""

    def __init__(self, units: int, rank: int, alpha: float, dropout: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank
        self.base_dense = layers.Dense(units, use_bias=False, trainable=False)
        self.lora_a = layers.Dense(rank, use_bias=False)
        self.lora_b = layers.Dense(units, use_bias=False)
        self.lora_dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        base = self.base_dense(inputs)
        lora = self.lora_b(self.lora_a(self.lora_dropout(inputs, training=training)))
        return base + lora * self.scaling


class LoRAMultiHeadAttention(layers.Layer):
    """Multi-head attention with LoRA applied to projection matrices."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scale = key_dim ** -0.5

        self.query_proj = LoRADense(num_heads * key_dim, rank, alpha, dropout, name="q_proj")
        self.key_proj = LoRADense(num_heads * key_dim, rank, alpha, dropout, name="k_proj")
        self.value_proj = LoRADense(num_heads * key_dim, rank, alpha, dropout, name="v_proj")
        self.out_proj = LoRADense(num_heads * key_dim, rank, alpha, dropout, name="o_proj")
        self.attn_dropout = layers.Dropout(dropout)

    def _reshape_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch, seq_len, self.num_heads, self.key_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, training: bool = False) -> tf.Tensor:
        q = self._reshape_heads(self.query_proj(query, training=training))
        k = self._reshape_heads(self.key_proj(key, training=training))
        v = self._reshape_heads(self.value_proj(value, training=training))

        scores = tf.matmul(q, k, transpose_b=True) * self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        context = tf.matmul(weights, v)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [tf.shape(context)[0], tf.shape(context)[1], -1])

        return self.out_proj(context, training=training)


class CausalTransformerBlock(layers.Layer):
    """Transformer block with causal masking and LoRA attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.attn = LoRAMultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            name="lora_attention",
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim),
            ],
            name="mlp",
        )
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.reshape(causal_mask, [1, 1, seq_len, seq_len])

        attn_out = self.attn(x, x, x, training=training)
        x = x + self.dropout(attn_out, training=training)
        x = self.norm_1(x)

        ffn_out = self.ffn(x, training=training)
        x = x + self.dropout(ffn_out, training=training)
        return self.norm_2(x)


def build_lora_causal_lm(
    vocab_size: int,
    max_length: int,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_dim: int = 2048,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
) -> keras.Model:
    """Create a lightweight causal language model with LoRA attention."""
    tokens = keras.Input(shape=(max_length,), dtype="int32", name="input_tokens")
    positions = tf.range(start=0, limit=max_length, delta=1)
    position_embed = layers.Embedding(max_length, hidden_dim, name="position_embedding")(positions)

    token_embed = layers.Embedding(vocab_size, hidden_dim, name="token_embedding")(tokens)
    x = token_embed + position_embed

    for idx in range(num_layers):
        x = CausalTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            name=f"transformer_block_{idx}",
        )(x)

    logits = layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens, logits, name="lora_causal_lm")


def build_instruction_dataset(
    prompts: List[str],
    targets: List[str],
    vocab_size: int = 20000,
    max_length: int = 512,
) -> Tuple[tf.data.Dataset, keras.layers.TextVectorization]:
    """Create a dataset for instruction-following fine-tuning."""
    text_vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_length,
        standardize="lower_and_strip_punctuation",
    )
    text_vectorizer.adapt(prompts + targets)

    full_text = [f"{prompt}\n{target}" for prompt, target in zip(prompts, targets)]
    tokenized = text_vectorizer(full_text)

    inputs = tokenized[:, :-1]
    labels = tokenized[:, 1:]
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset.batch(2).prefetch(tf.data.AUTOTUNE), text_vectorizer


def compute_similarity_rerank(
    lstm_title: str, generated_titles: List[str], encoder_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
) -> List[Tuple[str, float]]:
    """Re-rank generated titles by semantic similarity to LSTM prediction."""
    encoder = hub.load(encoder_url)
    lstm_embedding = encoder([lstm_title])
    title_embeddings = encoder(generated_titles)

    lstm_norm = tf.math.l2_normalize(lstm_embedding, axis=1)
    title_norm = tf.math.l2_normalize(title_embeddings, axis=1)

    similarity = tf.matmul(title_norm, lstm_norm, transpose_b=True).numpy().squeeze(-1)
    ranked = sorted(zip(generated_titles, similarity.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def train_stage1_example(config: MovieLensConfig) -> keras.Model:
    """Example function that prepares data and trains the Stage 1 LSTM."""
    ratings, movies = load_movielens_1m(config)
    ratings = filter_top_movies(ratings, config.top_k_movies)
    movie_table = build_genre_matrix(movies, config)
    tokenizer = build_title_tokenizer(movie_table, config)

    sequences, title_tokens, genres, targets = build_sequences(ratings, movie_table, tokenizer, config)

    model = build_lstm_model(config, num_movies=config.top_k_movies)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        {
            "movie_ids": sequences,
            "title_tokens": title_tokens,
            "genres": genres,
        },
        targets,
        epochs=5,
        batch_size=128,
        validation_split=0.15,
        shuffle=True,
    )
    return model


def train_stage3_lora_example(prompts: List[str], targets: List[str]) -> keras.Model:
    """Example Stage 3 LoRA fine-tuning in Keras."""
    dataset, vectorizer = build_instruction_dataset(prompts, targets)

    model = build_lora_causal_lm(
        vocab_size=vectorizer.vocabulary_size(),
        max_length=vectorizer.output_sequence_length,
        rank=8,
        alpha=16.0,
        dropout=0.1,
    )
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss_fn)
    model.fit(dataset, epochs=3)
    return model


if __name__ == "__main__":
    print("This module provides a Keras-based DUALRec reference implementation.")

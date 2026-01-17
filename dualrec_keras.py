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
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
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
    """Load MovieLens 1M ratings and movies metadata.

    Downloads and extracts the dataset if it is not already available.
    """
    data_root = keras.utils.get_file(
        fname="ml-1m.zip",
        origin="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        extract=True,
        cache_dir=config.data_dir,
        cache_subdir=".",
    )
    data_root = Path(data_root).with_suffix("")
    ratings_path = data_root / "ratings.dat"
    movies_path = data_root / "movies.dat"

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
        attn_layer = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name=f"mha_{idx}",
        )
        attn_out = attn_layer(x, x, use_causal_mask=True)

        lora_down = layers.Dense(rank, use_bias=False, name=f"lora_down_{idx}")(x)
        lora_down = layers.Dropout(dropout, name=f"lora_dropout_{idx}")(lora_down)
        lora_up = layers.Dense(hidden_dim, use_bias=False, name=f"lora_up_{idx}")(lora_down)
        lora_scaled = layers.Lambda(lambda t: t * (alpha / rank), name=f"lora_scale_{idx}")(lora_up)

        x = layers.Add(name=f"attn_residual_{idx}")([x, attn_out, lora_scaled])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"attn_norm_{idx}")(x)

        ffn = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim),
            ],
            name=f"ffn_{idx}",
        )
        ffn_out = ffn(x)
        x = layers.Add(name=f"ffn_residual_{idx}")([x, ffn_out])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_norm_{idx}")(x)

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
    lstm_title: str, generated_titles: List[str]
) -> List[Tuple[str, float]]:
    """Re-rank generated titles by semantic similarity to LSTM prediction."""
    vectorizer = layers.TextVectorization(output_sequence_length=50)
    vectorizer.adapt([lstm_title] + generated_titles)

    def encode(texts: List[str]) -> tf.Tensor:
        tokens = vectorizer(texts)
        return tf.cast(tokens, tf.float32)

    lstm_embedding = encode([lstm_title])
    title_embeddings = encode(generated_titles)

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

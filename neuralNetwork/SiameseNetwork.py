import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


graphlet_counts_df = pd.read_csv("neuralNetwork/train_data/graphlet_counts.csv").drop(
    columns=["Unnamed: 0"]
)
similarity_measures_df = pd.read_csv("neuralNetwork/train_data/similarity_measures.csv")
similarity_measures_df.rename(columns={"Unnamed: 0": "pair"}, inplace=True)


def generate_pairs(df):
    pairs, labels = [], df["Hellinger"]

    for row in df["pair"]:
        nets = row.split("---")
        pairs.append([graphlet_counts_df[nets[0]], graphlet_counts_df[nets[1]]])

    return np.array(pairs), np.array(labels)


pairs, labels = generate_pairs(similarity_measures_df)


def siamese_network(input_shape):
    model = Sequential(
        [
            Dense(128, activation="sigmoid", input_shape=input_shape),
            # Dense(64, activation="relu", input_shape=input_shape),
            # Dense(32, activation="relu"),
            # Dense(16, activation="relu"),
            Dense(64, activation="sigmoid"),
            # Dense(32, activation="sigmoid"),
        ]
    )
    return model


def create_siamese_model(input_dim):
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    siamese_model = siamese_network((input_dim,))
    encoded_a = siamese_model(input_a)
    encoded_b = siamese_model(input_b)

    similarity_score = Dense(1, activation="linear")(
        Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_a, encoded_b])
    )

    model = Model(inputs=[input_a, input_b], outputs=similarity_score)
    return model


input_dim = 30
siamese_model = create_siamese_model(input_dim)
siamese_model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mae"])

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
siamese_model.fit(
    [pairs[:, 0], pairs[:, 1]],
    labels,
    epochs=1000,
    batch_size=30,
    validation_split=0.2,
    callbacks=[early_stopping],
)

test_pairs, test_labels = generate_pairs(similarity_measures_df)
similarity_scores = siamese_model.predict([test_pairs[:, 0], test_pairs[:, 1]])

print("Sample similarity scores:")
for i in range(10):
    print(
        f"Pair {i}: True Similarity = {labels[i]:.4f}, Predicted Similarity = {similarity_scores[i][0]:.4f}"
    )

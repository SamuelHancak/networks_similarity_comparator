from itertools import combinations
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping


class NetworkClass:
    def __init__(self, graphlet_counts_df):
        self.input_dim = 30
        self.graphlet_counts_df = graphlet_counts_df
        self.train_graphlet_counts_df = pd.read_csv(
            "neuralNetwork/train_data/graphlet_counts.csv"
        )
        self.train_similarity_measures_df = pd.read_csv(
            "neuralNetwork/train_data/similarity_measures.csv"
        )

    def __generate_pairs_labels(self, df):
        pairs, labels = [], self.train_similarity_measures_df["Hellinger"]

        for row in self.train_similarity_measures_df["Pair"]:
            nets = row.split("---")
            pairs.append([df[nets[0]], df[nets[1]]])

        return np.array(pairs), np.array(labels)

    def __siamese_network(self):
        model = Sequential(
            [
                Dense(256, activation="sigmoid", input_shape=(self.input_dim,)),
                Dense(128, activation="sigmoid"),
                Dense(64, activation="sigmoid"),
            ]
        )

        return model

    def __create_siamese_model(self):
        input_a = Input(shape=(self.input_dim,))
        input_b = Input(shape=(self.input_dim,))

        siamese_model = self.__siamese_network()
        encoded_a = siamese_model(input_a)
        encoded_b = siamese_model(input_b)

        similarity_score = Dense(1, activation="linear")(
            Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_a, encoded_b])
        )

        model = Model(inputs=[input_a, input_b], outputs=similarity_score)

        return model

    def __compile_model(self):
        self.model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mae"])

    def __train_model(self, epochs=1000, batch_size=30, validation_split=0.2):
        pairs, labels = self.__generate_pairs_labels(self.train_graphlet_counts_df)
        early_stopping = EarlyStopping(
            monitor="val_mae", patience=10, restore_best_weights=True
        )
        self.model.fit(
            [pairs[:, 0], pairs[:, 1]],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
        )

    def __generate_pairs(self):
        column_combinations = list(combinations(self.graphlet_counts_df.columns, 2))
        pairs = []
        for col1, col2 in column_combinations:
            pairs.append([self.graphlet_counts_df[col1], self.graphlet_counts_df[col2]])
        return np.array(pairs)

    def predict_similarity(self):
        self.model = self.__create_siamese_model()
        self.__compile_model()
        self.__train_model()
        pairs = self.__generate_pairs()

        similarity_scores = self.model.predict([pairs[:, 0], pairs[:, 1]]).flatten()

        return similarity_scores

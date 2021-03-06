from abc import ABC, abstractmethod
from argparse import Namespace

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Model(ABC):
    """ An abstract class which outlines the structure of model classes. All classes which inherit from this class
        should implement the abstract methods """

    @abstractmethod
    def build(self, model_params: Namespace):
        """ This method does not require implementation inside abstract class"""
        ...


class CNNModel(Model):

    def __init__(self, num_features: int, max_sequence_length: int):
        """ Init method
        Args:
            num_features (int): Total number of words
            max_sequence_length (int): Maximum allowed length for an inout sequence
        """
        self.num_features = num_features
        self.max_sequence_length = max_sequence_length

    def build(self, model_params: Namespace):
        """ Creates a cnn model, compiles it and returns it
        Args:
        Returns:
            Built model
        """
        print("[CNNModel::build] Building CNN model")
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_sequence_length,), name="input"))
        model.add(layers.Embedding(input_dim=self.num_features,
                                   output_dim=model_params.embedding_dim,
                                   input_length=self.max_sequence_length))
        model.add(layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=model_params.optimizer,
                      loss=model_params.loss,
                      metrics=model_params.metrics)
        return model


class LSTMModel(Model):

    def __init__(self, num_features: int, max_sequence_length: int):
        """ Init method
        Args:
            num_features (int): Total number of words
            max_sequence_length (int): Maximum allowed length for an inout sequence
        """
        self.num_features = num_features
        self.max_sequence_length = max_sequence_length

    def build(self, model_params: Namespace):
        """ Creates an lstm model, compiles it and returns it
        Args:
        Returns:
            Built model
        """
        print("[LSTMModel::build] Building LSTM model")
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_sequence_length,), name="input"))
        model.add(layers.Embedding(input_dim=self.num_features,
                                   output_dim=model_params.embedding_dim,
                                   input_length=self.max_sequence_length))
        model.add(layers.LSTM(128, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=model_params.optimizer,
                      loss=model_params.loss,
                      metrics=model_params.metrics)
        return model


class HybridModel(Model):

    def __init__(self, num_features: int, max_sequence_length: int):
        """ Init method
        Args:
            num_features (int): Total number of words
            max_sequence_length (int): Maximum allowed length for an inout sequence
        """
        self.num_features = num_features
        self.max_sequence_length = max_sequence_length

    def build(self, model_params: Namespace):
        """ Creates an hybrid (lstm + cnn) model, compiles it and returns it
        Args:
        Returns:
            Built model
        """
        print("[HybridModel::build] Building Hybrid model")
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_sequence_length,), name="input"))
        model.add(layers.Embedding(input_dim=self.num_features,
                                   output_dim=model_params.embedding_dim,
                                   input_length=self.max_sequence_length))
        model.add(layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(128, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=model_params.optimizer,
                      loss=model_params.loss,
                      metrics=model_params.metrics)
        return model

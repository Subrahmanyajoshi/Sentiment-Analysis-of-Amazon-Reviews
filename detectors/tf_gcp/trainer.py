import os
import pickle
import zipfile
from argparse import Namespace
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from detectors.tf_gcp.common import BucketOps, SystemOps
from detectors.tf_gcp.callbacks import CallBacksCreator
from detectors.tf_gcp.data_ops.data_generator import DataGenerator
from detectors.tf_gcp.data_ops.io_ops import CloudIO, LocalIO
from detectors.tf_gcp.models.models import CNNModel, LSTMModel, HybridModel


class TokenizerDetails(object):

    def __init__(self, **kwargs):
        """ Init method
        """
        self.tokenizer = kwargs.get('tokenizer', None)
        self.top_k = kwargs.get('top_k', 20000)
        self.max_sequence_length = kwargs.get('max_sequence_length', 500)


class Trainer(object):
    MODEL_NAME = 'Amazon_Reviews_Analysis.hdf5'
    TOP_K = 20000
    MAX_SEQUENCE_LENGTH = 500

    def __init__(self, config: dict):
        """ Init method
        Args:
            config (dict): Dictionary containing configurations
        """
        self.run_type = config.get('train_type', 'unk').strip()
        self.train_params = Namespace(**config.get('train_params'))
        self.model_params = Namespace(**config.get('model_params'))
        self.cp_path = None
        self.csv_path = None
        self.bucket = None
        self.tokenizer = Tokenizer(num_words=Trainer.TOP_K)

        # Create a unique directory inside mentioned output directory for each training run. This makes sure that,
        # models or checkpoints or anything else that gets dumped during training, doesn't get overwritten.
        self.output_dir = os.path.join(self.train_params.output_dir,
                                       f"{self.model_params.model}_{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}")

        bucket_name = 'unk'

        # Get bucket name
        if self.train_params.data_dir.startswith('gs://'):
            bucket_name = self.train_params.data_dir.split('gs://')[1].split('/')[0]
        elif self.train_params.output_dir.startswith('gs://'):
            bucket_name = self.train_params.data_dir.split('gs://')[1].split('/')[0]
        if bucket_name != 'unk':
            self.bucket = BucketOps.get_bucket(bucket_name)

    @staticmethod
    def clean_up():
        """ Deletes temporary directories created while training
        """
        print(f"[Trainer::cleanup] Cleaning up...")
        SystemOps.run_command('rm *.csv.gz')
        SystemOps.check_and_delete('checkpoints')
        SystemOps.check_and_delete('trained_model')
        SystemOps.check_and_delete('parser_output')
        SystemOps.check_and_delete('train_logs.csv')
        SystemOps.check_and_delete('config.yaml')

    def load_data(self):
        """ Loads data from train_val.zip file
        """
        print(f"[Trainer::load_data] Copying data from {self.train_params.data_dir} to here...")
        SystemOps.run_command(f"gsutil -m cp -r "
                              f"{os.path.join(self.train_params.data_dir, 'train_val.zip')} ./")
        with zipfile.ZipFile('train_val.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        SystemOps.check_and_delete('train_val.zip')

    def preprocess(self) -> Tuple:
        """ Converts strings to a sequence of integers using keras tokenizer.
        """
        train_df = pd.read_csv('train_text.csv.gz')
        val_df = pd.read_csv('val_text.csv.gz')
        lines = list(train_df['input']) + list(val_df['input'])

        print("[Trainer::preprocess] Fitting tokenizer on texts...")
        self.tokenizer.fit_on_texts(lines)
        print(f"[Trainer::preprocess] Size of word index: {len(self.tokenizer.word_index)}")

        print("[Trainer::preprocess] Converting texts to sequences...")
        X_train = self.tokenizer.texts_to_sequences(list(train_df['input']))
        X_val = self.tokenizer.texts_to_sequences(list(val_df['input']))

        print("[Trainer::preprocess] Padding sequences so that they have the same length...")
        X_train = sequence.pad_sequences(X_train, maxlen=Trainer.MAX_SEQUENCE_LENGTH)
        X_val = sequence.pad_sequences(X_val, maxlen=Trainer.MAX_SEQUENCE_LENGTH)

        y_train = np.array(train_df['labels'])
        y_val = np.array(val_df['labels'])

        with open(os.path.join('parser_output', 'word_index.txt'), 'w') as fstream:
            for word, index in self.tokenizer.word_index.items():
                if index < Trainer.TOP_K:  # only save mappings for TOP_K words
                    fstream.write("{}:{}\n".format(word, index))
        print("[Trainer::preprocess] Dumped word index to word_index.txt")

        return X_train, y_train, X_val, y_val

    def save_tokenizer(self):
        """ Saves tokenizer object as a pickle file.
        """
        tokenizer_pickle = TokenizerDetails(tokenizer=self.tokenizer, top_k=Trainer.TOP_K,
                                            max_sequence_length=Trainer.MAX_SEQUENCE_LENGTH)
        with open('parser_output/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self):
        """ Creates dataset, preprocesses it, builds model, trains is and saves it to the specified destination directory
        """
        if self.bucket is not None:
            io_operator = CloudIO(bucket=self.bucket)
            self.load_data()
        else:
            io_operator = LocalIO()
        callbacks = CallBacksCreator.get_callbacks(callbacks_config=self.train_params.callbacks,
                                                   model_type=self.model_params.model,
                                                   io_operator=io_operator,
                                                   out_dir=self.output_dir)

        print("[Trainer::train] Loaded data")
        SystemOps.create_dir('parser_output')
        X_train, y_train, X_val, y_val = self.preprocess()

        self.save_tokenizer()
        print(f"Dumping tokenizer pickle file to {self.output_dir}")
        io_operator.write('parser_output', self.output_dir, use_system_cmd=False)

        num_features = min(len(self.tokenizer.word_index) + 1, Trainer.TOP_K)
        if self.model_params.model == 'CNN':
            Model = CNNModel(num_features=num_features,
                             max_sequence_length=Trainer.MAX_SEQUENCE_LENGTH).build(self.model_params)
        elif self.model_params.model == 'LSTM':
            Model = LSTMModel(num_features=num_features,
                              max_sequence_length=Trainer.MAX_SEQUENCE_LENGTH).build(self.model_params)
        elif self.model_params.model == 'Hybrid':
            Model = HybridModel(num_features=num_features,
                                max_sequence_length=Trainer.MAX_SEQUENCE_LENGTH).build(self.model_params)
        else:
            raise NotImplementedError(f"{self.model_params.model} model is currently not supported. "
                                      f"Please choose between CNN, LSTM and Hybrid")
        Model.summary()
        print(f"[Trainer::train] Built {self.model_params.model} model")

        SystemOps.check_and_delete('checkpoints')
        SystemOps.create_dir('checkpoints')

        print("[Trainer::train] Creating train and validation generators...")
        train_generator = DataGenerator(input_text=X_train,
                                        labels=y_train,
                                        batch_size=self.train_params.batch_size)
        validation_generator = DataGenerator(input_text=X_val,
                                             labels=y_val,
                                             batch_size=self.train_params.batch_size)

        print("[Trainer::train] Started training")
        _ = Model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.train_params.num_epochs,
            callbacks=callbacks,
            steps_per_epoch=self.train_params.steps_per_epoch,
            workers=self.train_params.workers,
            use_multiprocessing=self.train_params.use_multiprocessing
        )

        # save model as hdf5 file
        SystemOps.create_dir('trained_model')
        SystemOps.create_dir(os.path.join('trained_model', datetime.now().strftime("%Y_%m_%d-%H:%M:%S")))
        model_path = os.path.join('trained_model', f"{self.model_params.model}_{Trainer.MODEL_NAME}")
        Model.save_weights(model_path)

        print(f"[Trainer::train] Copying trained model to {self.output_dir}")
        io_operator.write('trained_model', self.output_dir)

        print(f"[Trainer::train] Copying train logs to {self.output_dir}")
        io_operator.write('train_logs.csv', self.output_dir, use_system_cmd=False)

import argparse
import os
import pickle
from typing import Dict

import pandas as pd
import numpy as np

from tqdm import tqdm
from argparse import Namespace
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import sequence

from detectors.tf_gcp.common import YamlConfig, SystemOps
from detectors.tf_gcp.models.models import CNNModel, LSTMModel, HybridModel
from detectors.vertex_ai_job import Trainer


class Predictor(object):

    def __init__(self, config: Dict):
        """ Init method
        Args:
            config (Dict): A dictionary containing user configurations.
        """
        self.config = config.get('predict_params', {})
        self.data_path = self.config.get('data_path')
        self.result_path = self.config.get('result_path')
        self.model_params = Namespace(**config.get('model_params'))
        self.model_path = self.config.get('model_path')
        self.tokenizer_path = self.config.get('tokenizer_path')
        self.test_data = self.load_data()
        self.tokenizer_details = self.load_tokenizer()
        self.model = self.load_model()

    def load_data(self):
        """ loads test data from the specified directory
        """
        if self.data_path.startswith('gs://'):
            print(f'[Predictor::load_data] Copying test data {self.data_path} to here...')
            SystemOps.run_command(f"gsutil -m cp -r {self.data_path} ./")
            self.data_path = os.path.basename(self.data_path)

        print(f'[Predictor::load_data] Reading texts from {self.data_path}')
        test_data = pd.read_csv(self.data_path)
        return test_data
        
    def load_tokenizer(self):
        """ Loads tokenizer from the pickle file. This file is created during training
        """
        if self.tokenizer_path.startswith('gs://'):
            print(f'[Predictor::load_tokenizer] Copying tokenizer {self.tokenizer_path} to here...')
            SystemOps.run_command(f"gsutil -m cp -r {self.tokenizer_path} ./")
            self.tokenizer_path = os.path.basename(self.tokenizer_path)

        with open(self.tokenizer_path, 'rb') as handle:
            tokenizer_details = pickle.load(handle)
        return tokenizer_details

    def load_model(self):
        """ Loads the model saved during training
        """
        if self.model_path.startswith('gs://'):
            print(f'[Predictor::load_model] Copying model {self.model_path} to here...')
            SystemOps.run_command(f"gsutil -m cp -r {self.model_path} ./")
            self.model_path = os.path.basename(self.model_path)

        num_features = min(len(self.tokenizer_details.tokenizer.word_index) + 1, self.tokenizer_details.top_k)

        # Load the correct model based on user configurations
        if self.model_params.model == 'CNN':
            model = CNNModel(num_features=num_features,
                             max_sequence_length=self.tokenizer_details.max_sequence_length).build(self.model_params)
        elif self.model_params.model == 'LSTM':
            model = LSTMModel(num_features=num_features,
                              max_sequence_length=self.tokenizer_details.max_sequence_length).build(self.model_params)
        elif self.model_params.model == 'Hybrid':
            model = HybridModel(num_features=num_features,
                                max_sequence_length=self.tokenizer_details.max_sequence_length).build(self.model_params)
        else:
            raise NotImplementedError(f"{self.model_params.model} model is currently not supported. "
                                      f"Please choose between CNN, LSTM and Hybrid")

        print(f"[Predictor::load_model] Loading weights for {self.model_params.model} model from {self.model_path}")
        model.load_weights(self.model_path)
        return model

    def predict(self, review: np.ndarray):
        """ Takes one value and calculates it's prediction
        Args:
            review (np.ndarray): A review test in the form of sequence of integers
        """
        result = self.model.predict(np.array([review]))
        result = result[0][0]
        if result > 0.5:
            return 1
        else:
            return 0

    def run(self):
        """ Loads test data and model, and creates predictions
        """
        lines = list(self.test_data['input'])
        true_labels = []
        predicted_labels = []
        
        if 'labels' in self.test_data.columns:
            true_labels = list(self.test_data['labels'])
        else:
            print(f"[Predictor::run] Labels are not found in {self.data_path} file. "
                  f"Performance metrics and Confusion matrix will not be calculated")
        
        lines = self.tokenizer_details.tokenizer.texts_to_sequences(lines)
        lines = sequence.pad_sequences(lines, maxlen=self.tokenizer_details.max_sequence_length)

        for line in tqdm(lines, desc="Predicting"):
            predicted_labels.append(self.predict(line))

        self.test_data['predictions'] = predicted_labels
        if not self.result_path.endswith('.csv'):
            raise ValueError(f"Cannot save result csv file! Specified path {self.result_path} is not a csv file...")

        if self.result_path.startswith("gs://"):
            directory, file_name = os.path.split(self.result_path)
            self.test_data.to_csv(file_name, index=False)
            print(f'[Predictor::run] Copying result csv file to Google Storage bucket...')
            SystemOps.run_command(f"gsutil mv -r {file_name} {self.result_path}")
        else:
            self.test_data.to_csv(self.result_path, index=False)

        if len(true_labels) != 0:
            cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
            tn, fp, fn, tp = cm.ravel()
            metrics = {'val_accuracy': (tp + tn) / len(true_labels), 'val_precision': tp / (tp + fp),
                       'val_recall': tp / (tp + fn), 'val_f1': tp / (tp + 0.5 * (fp + fn))}

            for key, value in metrics.items():
                print(f"{key}: {value}")

    def clean_up(self):
        """ Cleans up all temporarily created directories while running
        """
        SystemOps.check_and_delete(self.data_path)
        SystemOps.check_and_delete(self.model_path)
        SystemOps.check_and_delete(self.tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', required=False,
                        help='A boolean switch to tell the script to run predictor')
    parser.add_argument('--train', action='store_true', required=False,
                        help='A boolean switch to tell the script to run trainer')
    parser.add_argument('--config', type=str, required=True,
                        help='Yaml configuration file path')

    args = parser.parse_args()

    if not args.train and not args.predict:
        raise ValueError('Please specify either --train or --predict command line argument while running')

    config = YamlConfig.load(filepath=args.config)

    if args.train:
        print('[main] Initialising training')
        trainer = Trainer(config=config)
        trainer.train()
        Trainer.clean_up()

    if args.predict:
        print('[main] Initialising testing')
        predictor = Predictor(config=config)
        predictor.run()
        predictor.clean_up()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, Exception):
        print("An exception has caused the system to terminate")
        Trainer.clean_up()
        raise

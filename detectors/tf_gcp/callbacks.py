import importlib
import os

from typing import Dict, Union
from tensorflow.keras.callbacks import Callback

from detectors.tf_gcp.data_ops.io_ops import LocalIO, CloudIO


class GCSCallback(Callback):
    """ A custom callback created to copy checkpoints created by ModelCheckpoint callback to GCS bucket.
        ModelCheckpoint writes to a local directory called 'checkpoints' and this custom callback will check
        this directory at the end of every epoch and if any checkpoints are found, they are copied to GCS bucket """

    def __init__(self, cp_path: str, io_operator: Union[LocalIO, CloudIO]):
        """ init method
        Args:
            cp_path (str): GCS/Local path to checkpoints directory
            io_operator (Union[LocalIO, CloudIO]): an operator which contains functions to copy or move from
                                                   one path to other
        """
        super(GCSCallback, self).__init__()
        self.checkpoint_path = cp_path
        self.io_operator = io_operator

    def on_epoch_end(self, epoch, logs=None):
        for cp_file in os.listdir('./checkpoints'):
            src_path = os.path.join('./checkpoints', cp_file)
            self.io_operator.write(src_path=src_path, dest_path=self.checkpoint_path, use_system_cmd=False)


class CallBacksCreator(object):

    @staticmethod
    def get_callbacks(callbacks_config: Dict, model_type: str, io_operator: Union[LocalIO, CloudIO], out_dir: str):
        """ creates callbacks
        Args:
            callbacks_config (Dict): a dictionary containing callback configurations.
            model_type (str): specifies which type of model is being used.
            io_operator (Union[LocalIO, CloudIO]): an operator which contains functions to copy or move from
                                                   one path to other.
            out_dir (str): Directory where output artifacts of the trainer are to be dumped.
        Returns:
            A list containing created callback objects
        """
        callbacks = []
        module = importlib.import_module('tensorflow.keras.callbacks')
        cp_path = os.path.join(out_dir, 'checkpoints')
        for cb in callbacks_config:
            if cb == 'ModelCheckpoint':
                _, filename = os.path.split(callbacks_config[cb]['filepath'])
                callbacks_config[cb]['filepath'] = os.path.join('./checkpoints',
                                                                f"{model_type}_{filename}")

            if cb == 'CSVLogger':
                _, filename = os.path.split(callbacks_config[cb]['filename'])
                callbacks_config[cb]['filename'] = filename

            if cb == 'TensorBoard':
                callbacks_config[cb]['log_dir'] = f"{callbacks_config[cb]['log_dir']}_{model_type}"

            obj = getattr(module, cb)
            callbacks.append(obj(**callbacks_config[cb]))
        gcs_callback = GCSCallback(cp_path=cp_path, io_operator=io_operator)
        callbacks.append(gcs_callback)
        return callbacks

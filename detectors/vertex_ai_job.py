import argparse
import os

from detectors.tf_gcp.common import YamlConfig
from detectors.tf_gcp.data_ops.io_ops import CloudIO
from detectors.tf_gcp.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, help='config file containing train configurations',
                        required=False)
    args = parser.parse_args()

    CloudIO.copy_from_gcs(args.train_config, './')
    config = YamlConfig.load(filepath=os.path.abspath('config.yaml'))
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()

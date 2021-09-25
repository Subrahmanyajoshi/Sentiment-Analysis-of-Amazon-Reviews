
## Overview
- The goal of this project is to provide a platform to build and train machine learning models for text
classification using Google Cloud's AI platform.
- In this project I have used Amazon reviews dataset to detect sentiments from review texts.

## Dataset
- The dataset was obtained from Kaggle [here](https://www.kaggle.com/bittlingmayer/amazonreviews?select=train.ft.txt.bz2).
- Dataset is already split into train and test datasets. All details about the data also can be found in kaggle itself.

## Data preprocessing
- Original dataset file was in bz2 compressed format.
- Decompressing it get's text files which, in every line contain label and review text.
- These text files were converted into csv format to access and process them faster.
- These text files were too big to be processed in local machine. I have used google cloud's 
AI platform notebooks to process them.
- All preprocessing steps are available in [this](tools/preprocessor.ipynb) notebook.


## Training model locally

- Install packages from requirements.txt
```shell
pip install -r requirements.txt
```
- Open config file at config/config.yaml and update it accordingly.
- Go to project root and run following. It sets environment variable 
   PYTHONPATH to project root so that modules can be imported easily.
   
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
- Run trainer
```shell
python3 -m detectors.detector --train --config='./config/config.yaml'
```


## Submitting Training job to AI Platform

- Go to google cloud console and create and open an instance of AI Notebooks. 
   If not known how to do that, follow the procedure given [here](https://cloud.google.com/notebooks/docs/create-new).
   (Create the notebook with low specifications, as we will not be running actual training here. 
   This just acts as a base machine to submit the job to AI platform. 
   The best choice is n1-standard-2 machines which have 7.5 gb memory and 2 vCpus).
- Open a terminal and clone this repository.
```shell
git clone https://github.com/Subrahmanyajoshi/Breast-Cancer-Detection.git
```
- Create a google storage bucket. If not known how to do that, 
   follow the procedure given [here](https://cloud.google.com/storage/docs/creating-buckets).
- Create a folder inside newly created bucket named 'train_data'.
- Zip 'train_text.csv.gz' and 'val_text.csv.gz' obtained after preprocessing, into a file named 'train_val.zip'.
- Upload train_val.zip into train_data folder inside bucket.
- Open config file at config/config.yaml and update it accordingly. Make sure to mention full paths
   starting from 'gs://' while specifying paths inside the bucket.
- Open the notebook detectors/tf_gcp/ai_platform_trainer.ipynb and run the notebook 
   following the steps given there.
- The notebook will submit the training job to AI Platform. 

### Activating and using tensorboard to monitor training

- Tensorboard will be running on port 6006 by default.
- A firewall rule must be set up to open this port, follow the procedure given
   [here](https://docs.bitnami.com/google/faq/administration/use-firewall/).
- Once done, open a terminal and run following. Provide the path to tensorboard directory, 
   specified in config file.
```shell
tensorboard --logdir <path/to/log/directory> --bind_all
```
- Get the external Ip of the VM on which notebook is running, from 'VM Instances' page on google cloud console.
- Open a browser and open following link .
```text
http://<external_ip_address>:6006/
```

## Predicting
- Open config file at config/config.yaml and update model path, and data path at the very bottom.
- Go to project root and run following. It sets environment variable PYTHONPATH to project root so that 
   modules can be imported easily.
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
- Run predictor
```shell
python3 -m detectors.detector --predict --config='./config/config.yaml'
```
- Classification results will be printed on the screen. 

## Results
coming soon :P
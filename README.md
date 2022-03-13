
## Overview
- The goal of this project is to provide a platform to build and train machine learning models for text
classification using Google Cloud's Vertex AI.
- In this project I have used Amazon reviews dataset to extract sentiments from review texts.

## Dataset
- The dataset was obtained from Kaggle [here](https://www.kaggle.com/bittlingmayer/amazonreviews?select=train.ft.txt.bz2).
- Dataset was already split into train and test datasets. All details about the data also can be found in kaggle itself.

## Data preprocessing
- Original dataset file was in bz2 compressed format.
- Decompressing it get's text files which, in every line contain a label and a review text.
- These text files were converted into csv format to access and process them faster.
- These text files were too big to be processed in local machine. I have used google cloud's 
AI platform notebooks to process them.
- All preprocessing steps are available in [this](tools/preprocessor.ipynb) notebook.


## Training models locally

- Install packages from requirements.txt
```shell
pip install -r requirements_baremetal.txt
```
- Open config file at config/config.yaml and update it accordingly.
- Make sure 'model_type' parameter under 'model_params' section is set to model under consideration.
- Go to project root and run following. It sets environment variable 
   PYTHONPATH to project root so that modules can be imported easily.
   
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
- Run trainer
```shell
python3 -m detectors.detector --train --config='./config/config.yaml'
```


## Submitting Training job to Vertex AI

- Go to google cloud console and create and open an instance of AI Notebooks. 
   If not known how to do that, follow the procedure given [here](https://cloud.google.com/notebooks/docs/create-new).
   (Create the notebook with low specifications, as we will not be running actual training here. 
   This just acts as a base machine to submit the job to Vertex AI. 
   The best choice is n1-standard-2 machines which have 7.5 gb memory and 2 vCpus).
- Open a terminal and clone this repository.
```shell
git clone https://github.com/Subrahmanyajoshi/Sentiment-Analysis-of-Amazon-Reviews.git
```
- Navigate to project root.
- Create a google storage bucket. If not known how to do that, 
   follow the procedure given [here](https://cloud.google.com/storage/docs/creating-buckets).
- Create a folder inside newly created bucket named 'train_data'.
- Zip 'train_text.csv.gz' and 'val_text.csv.gz' obtained after preprocessing, into a file named 'train_val.zip'.
- Upload train_val.zip into train_data folder inside bucket.
- Open config file at config/config.yaml and update it accordingly. Make sure to mention full paths
   starting from 'gs://' while specifying paths inside the bucket.
- Open the [detectors/vertex_ai_job_submitter](detectors/vertex_ai_job_submitter.sh) shell script.
- Change the environment variables if required.
- Run the shell script
```shell
./detectors/vertex_ai_job_submitter.sh
```
- The shell script will create a custom training job and submit it to Vertex AI. 

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
- Open config file at config/config.yaml and update parameters under 'predict_params' section.
- Make sure 'model_type' parameter under 'model_params' section is set to model under consideration.
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
- Three types of model were used
    1. A single dimensional CNN model.
    2. An LSTM model.
    3. A Hybrid model which consists of both CNN layers and LSTM cells.
- CNN and Hybrid models were run for a batch size of 2048, for 5 epochs.
- LSTM kept throwing out of memory error, so batch size had to be reduced to 1024.
- CNN had the best loss/accuracy at 3rd epoch. Hybrid had the same at 2nd epoch. LSTM had it at 5th epoch.
- Model performance report: (Testing is done on 5000 reviews)

    | Model  | Train Accuracy  | Validation Accuracy  | Test Accuracy  |
    |---|---|---|---|
    |CNN| 94.08% | 94.13% | 93.20%  | 
    |LSTM| 95.23% | 94.92% | 94.32% | 
    |Hybrid| 94.99% | 95.01% | 94.28%  | 

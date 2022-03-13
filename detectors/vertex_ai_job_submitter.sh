# Training tensorflow model using Google Cloud Vertex AI

# Environment Variables
export PROJECT='text-analysis-323506'
export REGION='us-central1'
export BUCKET='text-analysis-323506'
export MACHINE_TYPE='n1-highmem-4'
export REPLICA_COUNT=1
export EXECUTE_IMAGE_URI='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest'
export LOCAL_PACKAGE_PATH='/home/jupyter/Sentiment-Analysis-of-Amazon-Reviews'
export PYTHON_MODULE='detectors.vertex_ai_job'
export JOBNAME=sentiment_analysis_$(date -u +%y%m%d_%H%M%S)

# gcloud configurations
gcloud config set project $PROJECT
gcloud config set compute/region $REGION                                                                

# Move latest config file to google storage bucket
gsutil -m cp -r ./config gs://$BUCKET

# Submit training job  
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=$JOBNAME \
    --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTE_IMAGE_URI,local-package-path=$LOCAL_PACKAGE_PATH,python-module=$PYTHON_MODULE \
    --args=--train-config=gs://$BUCKET/config/config.yaml

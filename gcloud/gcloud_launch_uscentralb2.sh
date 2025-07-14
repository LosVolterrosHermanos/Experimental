#!/bin/bash

export PROJECT_ID=dana-star
export TPU_NAME=dana-star-1
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-64
export RUNTIME_VERSION=tpu-ubuntu2204-base

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

gcloud compute tpus tpu-vm create $TPU_NAME \
    --network=dana-star-network \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --preemptible

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='git clone -b timescale-experiment https://github.com/LosVolterrosHermanos/Experimental'

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='bash Experimental/gcloud/setup.sh'

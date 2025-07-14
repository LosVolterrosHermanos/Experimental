#!/bin/bash

export PROJECT_ID=dana-star
export TPU_NAME=dana-star-europe
export ZONE=europe-west4-a
export ACCELERATOR_TYPE=v3-8
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
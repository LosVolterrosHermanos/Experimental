#!/bin/bash

export PROJECT_ID=dana-star
export TPU_NAME=dana-star-central
export ZONE=us-central1-f
export ACCELERATOR_TYPE=v2-8
export RUNTIME_VERSION=tpu-ubuntu2204-base

gcloud compute tpus tpu-vm create $TPU_NAME \
    --network=dana-star-network \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --preemptible
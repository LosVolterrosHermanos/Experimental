#!/bin/bash

export PROJECT_ID=dana-star
export TPU_NAME=dana-star-1
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=tpu-ubuntu2204-base

gcloud compute tpus tpu-vm create $TPU_NAME \
    --network=dana-star-network \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --preemptible
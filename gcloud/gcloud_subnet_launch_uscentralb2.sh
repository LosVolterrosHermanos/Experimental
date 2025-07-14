#!/bin/bash
# https://cloud.google.com/vpc/docs/create-modify-vpc-networks
# this may not be ZONE-gated???
export PROJECT_ID=dana-star
export ZONE=us-central2-b

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

gcloud compute networks create dana-star-network \
    --subnet-mode=auto \
    --bgp-routing-mode=regional \
    --mtu=1460

gcloud compute firewall-rules create dana-star-firewall \
    --network dana-star-network \
    --allow tcp,udp,icmp \
    --source-ranges 0.0.0.0/0

gcloud compute firewall-rules create dana-star-firewall-ssh \
    --network dana-star-network \
    --allow tcp:22,tcp:3389,icmp
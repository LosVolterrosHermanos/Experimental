#!/bin/bash
#setup script 

cd ~/Experimental/gcloud

pip install -r requirements.txt

cd ~/Experimental 

pip install -e . 

cd ~/Experimental/dana-nonquadratic-tests/gpt2

python grab_fineweb.py



#!/bin/bash

cd classifer/defense/ONION
DATA_TO_ONION="classifer/train/final_train_with_camouflage.json"
DATA_DIR="classifer/data/yelp/badnets"
THRESHOLD=18       
BATCH_SIZE=32         
 
python onion.py \
    --data_to_clean "$DATA_TO_ONION" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE"



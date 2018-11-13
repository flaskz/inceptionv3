#!/bin/bash

export TFHUB_CACHE_DIR="./tfhub"

rm ./retrain_checkpoint/*
python retrain.py --image_dir "$1" \
		--output_graph "$2/graphs/output_graph_$3.pb" \
                --output_labels "$2/labels/output_labels_$3.txt" \
                --bottleneck_dir "./bottleneck" \
                --summaries_dir "$2/summaries/"

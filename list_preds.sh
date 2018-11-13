#!/bin/bash
python lista_resultados.py \ 
	--graph $1/graphs/output_graph_$2.pb \
       	--labels $1/labels/output_labels_$2.txt \
	--input_layer Placeholder \
	--output_layer final_result \ 
	--validation_dir $3 > logs/logs_$4.txt


#!/bin/bash
outputfolder=$1/$5/saved_models_$2_$3_$4
mkdir -p ${outputfolder}
mkdir data
#raw_data_file=../data/data_codebert.jsonl
raw_data_file=$6
python run_mask_prediction.py \
    --output_dir=${outputfolder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --do_test \
    --do_abstraction \
    --data_file=${raw_data_file} \
    --epoch 20 \
    --code_length 512 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --dataname $1 \
    --test_projects $2 $3 $4 \
    --seed 123456 2>&1| tee ${outputfolder}/${5}.log
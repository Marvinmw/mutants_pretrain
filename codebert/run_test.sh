#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J bert
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#conda activate codebert

outputfolder=$1/$5/saved_models_$2_$3_$4
#raw_data_file=../data/data_codebert.jsonl
raw_data_file=$6
python run.py \
    --output_dir=${outputfolder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --data_file=${raw_data_file} \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --dataname $1 \
    --test_projects $2 $3 $4 \
    --seed 123456 2>&1| tee ${outputfolder}/${5}_test.log

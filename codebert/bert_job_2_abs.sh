#!/bin/bash -l
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J b2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert


#bash run_mask_prediction.sh bert_context Cli Codec Collections train_0 ../data/data_codebert.jsonl_context
#bash run_mask_prediction.sh bert_context Compress Csv Gson train_1 ../data/data_codebert.jsonl_context
bash run_mask_prediction.sh bert_context_abs JacksonCore JacksonDatabind JacksonXml train_2 ../data/data_codebert_abs_token_diff.jsonl_context
bash run_mask_prediction.sh bert_context_abs Jsoup JxPath Lang train_3 ../data/data_codebert_abs_token_diff.jsonl_context
bash run_mask_prediction.sh bert_context_abs Math Mockito Time train_4 ../data/data_codebert_abs_token_diff.jsonl_context
#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J ibir_context
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert
name=ibir
bash run.sh ${name}_context Codec JacksonDatabind Csv train_0 ../data/data_${name}.jsonl_context
bash run.sh ${name}_context JxPath Compress Jsoup train_1 ../data/data_${name}.jsonl_context
bash run.sh ${name}_context Lang Math Mockito train_2 ../data/data_${name}.jsonl_context
bash run.sh ${name}_context JacksonCore Time  Cli train_3 ../data/data_${name}.jsonl_context
bash run.sh ${name}_context Gson JacksonXml Collections  train_4 ../data/data_${name}.jsonl_context

bash run_freeze.sh ${name}_context_freeze Codec JacksonDatabind Csv train_0 ../data/data_${name}.jsonl_context
bash run_freeze.sh ${name}_context_freeze JxPath Compress Jsoup train_1 ../data/data_${name}.jsonl_context
bash run_freeze.sh ${name}_context_freeze Lang Math Mockito train_2 ../data/data_${name}.jsonl_context
bash run_freeze.sh ${name}_context_freeze JacksonCore Time  Cli train_3 ../data/data_${name}.jsonl_context
bash run_freeze.sh ${name}_context_freeze Gson JacksonXml Collections  train_4 ../data/data_${name}.jsonl_context
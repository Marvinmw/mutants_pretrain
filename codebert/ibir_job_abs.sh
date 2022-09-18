#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J ibir
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert
name=ibir_abs
bash run.sh ${name} Codec JacksonDatabind Csv train_0 ../data/data_${name}.jsonl
bash run.sh ${name} JxPath Compress Jsoup train_1 ../data/data_${name}.jsonl
bash run.sh ${name} Lang Math Mockito train_2 ../data/data_${name}.jsonl
bash run.sh ${name} JacksonCore Time  Cli train_3 ../data/data_${name}.jsonl
bash run.sh ${name} Gson JacksonXml Collections  train_4 ../data/data_${name}.jsonl


bash run_freeze.sh ${name}_freeze Codec JacksonDatabind Csv train_0 ../data/data_${name}.jsonl
bash run_freeze.sh ${name}_freeze JxPath Compress Jsoup train_1 ../data/data_${name}.jsonl
bash run_freeze.sh ${name}_freeze Lang Math Mockito train_2 ../data/data_${name}.jsonl
bash run_freeze.sh ${name}_freeze JacksonCore Time  Cli train_3 ../data/data_${name}.jsonl
bash run_freeze.sh ${name}_freeze Gson JacksonXml Collections  train_4 ../data/data_${name}.jsonl

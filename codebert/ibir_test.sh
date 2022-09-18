#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=0-05:00:00
#SBATCH --qos=normal
#SBATCH -J ibir
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert

bash run_test.sh ibir Cli Codec Collections train_0_test_mcc ../data/data_ibir.jsonl
bash run_test.sh ibir Compress Csv Gson train_1_test_mcc ../data/data_ibir.jsonl
bash run_test.sh ibir JacksonCore JacksonDatabind JacksonXml train_2_test_mcc ../data/data_ibir.jsonl
bash run_test.sh ibir Jsoup JxPath Lang train_3_test_mcc ../data/data_ibir.jsonl
bash run_test.sh ibir Math Mockito Time train_4_test_mcc ../data/data_ibir.jsonl
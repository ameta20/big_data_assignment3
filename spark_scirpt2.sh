#!/bin/bash -l
### Request one sequential task requiring half the memory of a regular iris node for 1 day
#SBATCH -J task2       # Job name
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --qos=normal
#SBATCH --mem=64GB         # if above 112GB: consider bigmem p



export SPARK_DRIVER_MEMORY=16G
export SPARK_EXECUTOR_MEMORY=16G

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
module load devel/Spark/3.5.4-foss-2023b-Java-17
source .venv/bin/activate
pip install --user nltk


MY_QUERIES=(
    "algebra"
    "war"
    "century"
    "philosophy"
    "pollution"
    "university"
)

# Construct the --queries argument string
QUERY_ARG_STRING=""
for q_item in "${MY_QUERIES[@]}"; do
  QUERY_ARG_STRING+="\"${q_item}\" " 
done


# Run your Spark job
time spark-submit run_lsa.py  \
 --tokenizer_type nlp \
    --num_freq_terms 20000 \
    --k_svd 250 \
    --sample_fraction 1.0 \
    --queries ${QUERY_ARG_STRING} \
    > logs/partC.txt 2>&1



# MODEL_TYPE="llama2"
# MODEL_TYPE="llama3"
# MODEL_TYPE="llama3.1"
# MODEL_TYPE="openchat3.6"
# MODEL_TYPE="qwen2.5"
# MODEL_TYPE="starlinglm"
# MODEL_TYPE="deepseek-distill-llama"
MODEL_TYPE="deepseek-distill-qwen"

# DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/I-FSJ"
# DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/HarmBench-V5"
# DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V5"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3.1-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-OpenChat3.6-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Qwen2.5-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-StarlingLM-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-DeepSeek-Distill-Llama-t50"
DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-DeepSeek-Distill-Qwen-t50"

# OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v0"

# DATASET_NAME="harmbench_behaviors"
# DATASET_NAME="harmful_behaviors"
DATASET_NAME="unfiltered"

python Self-Instruct-FSJ/src/put_analysis_to_data.py \
    --pt_data_path ${OUTPUT_DIR}/${DATASET_NAME}_ppl.jsonl \
    --json_data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/${DATASET_NAME}.json

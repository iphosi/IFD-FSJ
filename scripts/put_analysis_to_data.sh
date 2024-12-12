# MODEL_TYPE="llama2"
# MODEL_TYPE="llama3"
# MODEL_TYPE="mistral"
MODEL_TYPE="openchat3.6"
# MODEL_TYPE="qwen2.5"
# MODEL_TYPE="starlinglm"

DATA_DIR="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50"
# DATA_DIR="IFD-FSJ/datasets/benchmarks/AdvBench-V5"

# OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v0"

# DATASET_NAME="harmful_behaviors"
DATASET_NAME="unfiltered"

python Superfiltering/code_ifd/put_analysis_to_data.py \
    --pt_data_path ${OUTPUT_DIR}/${DATASET_NAME}_ppl.jsonl \
    --json_data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/${DATASET_NAME}.json

# MODEL_TYPE="llama3"
MODEL_TYPE="llama2"
# DATA_DIR="IFD-FSJ/datasets/benchmarks/AdvBench"
# DATASET_NAME="harmful_behaviors"
DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B"
DATASET_NAME="unfiltered"
# OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v1"

python Superfiltering/code_ifd/put_analysis_to_data.py \
    --pt_data_path ${OUTPUT_DIR}/${DATASET_NAME}_ppl.jsonl \
    --json_data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/${DATASET_NAME}.json

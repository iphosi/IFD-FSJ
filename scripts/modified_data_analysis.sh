export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_TYPE="llama3"
MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_TYPE="llama2"
DATA_DIR="IFD-FSJ/datasets/benchmarks/AdvBench"
DATASET_NAME="harmful_behaviors"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B"
# DATASET_NAME="unfiltered"
OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v1"

SYSTEM_MESSAGE_VERSION="v1"

mkdir -p ${OUTPUT_DIR}

python Superfiltering/code_ifd/modified_data_analysis.py \
    --data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --instruction_col_name "instruction" \
    --response_col_name "adv_prefix" \
    --save_path ${OUTPUT_DIR}/${DATASET_NAME}_ppl.jsonl \
    --model_name_or_path IFD-FSJ/models/${MODEL_NAME} \
    --max_length 1280 \
    --system_message_version ${SYSTEM_MESSAGE_VERSION} \
    > IFD-FSJ/log/log_5.out 2>&1 &

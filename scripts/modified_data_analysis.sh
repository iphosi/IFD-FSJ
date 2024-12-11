export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

MODEL_NAME="Meta-Llama-3-8B-Instruct"
MODEL_TYPE="llama3"
# MODEL_NAME="Llama-2-7b-chat-hf"
# MODEL_TYPE="llama2"
# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_TYPE="qwen2.5"

# DATA_DIR="IFD-FSJ/datasets/benchmarks/AdvBench-V0"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/AdvBench-Qwen2.5-t300"
# DATA_DIR="IFD-FSJ/datasets/benchmarks/I-FSJ"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/I-FSJ/llama2_sep"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/I-FSJ-Llama2-t300"
DATA_DIR="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama3-t50"

# DATASET_NAME="harmful_behaviors"
DATASET_NAME="unfiltered"
# DATASET_NAME="instruction"
# DATASET_NAME="mistral_demonstration"

OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v0"

SYSTEM_MESSAGE_VERSION="v0"

mkdir -p ${OUTPUT_DIR}

python Superfiltering/code_ifd/modified_data_analysis.py \
    --data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --instruction_col_name "instruction" \
    --response_col_name "output" \
    --save_path ${OUTPUT_DIR}/${DATASET_NAME}_ppl.jsonl \
    --model_name_or_path IFD-FSJ/models/${MODEL_NAME} \
    --max_length 1280 \
    --system_message_version ${SYSTEM_MESSAGE_VERSION} \
    > IFD-FSJ/log/log_0.out 2>&1 &

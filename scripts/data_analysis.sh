export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

MODEL_NAME="gpt2"
# MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_TYPE="llama2"
# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_TYPE="llama3"
# MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
# MODEL_TYPE="llama3.1"
# MODEL_NAME="OpenChat-3.6-8B"
# MODEL_TYPE="openchat3.6"
# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_TYPE="qwen2.5"
# MODEL_NAME="Starling-LM-7B-beta"
# MODEL_TYPE="starlinglm"

# DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/I-FSJ"
# DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/HarmBench-V5"
DATA_DIR="Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V2"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3.1-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-OpenChat3.6-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Qwen2.5-t50"
# DATA_DIR="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-StarlingLM-t50"

# DATASET_NAME="harmbench_behaviors"
DATASET_NAME="harmful_behaviors"
# DATASET_NAME="unfiltered"

OUTPUT_DIR="${DATA_DIR}/${MODEL_TYPE}/w_chat_template/sys_msg_v0"

SYSTEM_MESSAGE_VERSION="v0"

mkdir -p ${OUTPUT_DIR}

python Self-Instruct-FSJ/src/data_analysis.py \
    --data_path ${OUTPUT_DIR}/${DATASET_NAME}.json \
    --instruction_col_name "instruction" \
    --response_col_name "adv_prefix" \
    --save_path ${OUTPUT_DIR}/${DATASET_NAME}_instruction_gpt2_ppl_w20_s20.jsonl \
    --model_name_or_path Self-Instruct-FSJ/models/${MODEL_NAME} \
    --max_length 1280 \
    --system_message_version ${SYSTEM_MESSAGE_VERSION} \
    --analyse_instruction \
    --window_size 20 \
    --stride 20 \
    > Self-Instruct-FSJ/log/log_1.out 2>&1 &

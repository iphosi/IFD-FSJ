export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

MODEL_NAME="Meta-Llama-3-8B-Instruct"
MODEL_TYPE="llama3"
# MODEL_NAME="Llama-2-7b-chat-hf"
# MODEL_TYPE="llama2"
DATA_DIR="IFD-FSJ/datasets/demonstrations/I-FSJ/no_sep"
DATASET_NAME="mistral_demonstration_list_official"
OUTPUT_DIR="${DATA_DIR}/wo_chat_template"

mkdir -p ${OUTPUT_DIR}

python Superfiltering/code_ifd/data_analysis.py \
    --data_path ${DATA_DIR}/${DATASET_NAME}.json \
    --save_path ${OUTPUT_DIR}/${DATASET_NAME}_${MODEL_TYPE}_ppl.jsonl \
    --model_name_or_path IFD-FSJ/models/${MODEL_NAME} \
    --max_length 1280 \
    > IFD-FSJ/log/log_0.out 2>&1 &

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


# DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca3-8B"
DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B"
# OUTPUT_DIR="${DATA_DIR}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0"
INPUT_DATASET_NAME="unfiltered"
OUTPUT_DATASET_NAME="filtered_ifd_1.0_1.2"

python Superfiltering/code_diversity_fla/modified_do_fla.py \
    --json_data_path ${OUTPUT_DIR}/${INPUT_DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/demo_v4/${OUTPUT_DATASET_NAME}.json \
    --ifd_num 512 \
    --ifd_lower 1.0 \
    --ifd_lupper 1.2 \
    --fla_num 256 \
    --emb_model IFD-FSJ/models/all-MiniLM-L6-v2 \
    > IFD-FSJ/log/log_5.out 2>&1 &

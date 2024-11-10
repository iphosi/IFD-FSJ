export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca3-8B"
# DATA_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B"
# OUTPUT_DIR="${DATA_DIR}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/w_chat_template/sys_msg_v0"
INPUT_DATASET_NAME="unfiltered_llama2_ifd_fillna"
OUTPUT_DATASET_NAME="filtered_llama2_ifd_0.0_0.4_fla_333_256"

python Superfiltering/code_diversity_fla/do_fla.py \
    --json_data_path ${OUTPUT_DIR}/${INPUT_DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/${OUTPUT_DATASET_NAME}.json \
    --ifd_num 333 \
    --ifd_lower 0.0 \
    --ifd_lupper 0.4 \
    --fla_num 256 \
    --emb_model IFD-FSJ/models/all-MiniLM-L6-v2 \
    > IFD-FSJ/log/log_3.out 2>&1 &

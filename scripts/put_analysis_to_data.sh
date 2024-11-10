MODEL_TYPE="llama3"
# MODEL_TYPE="llama2"
DATA_DIR="IFD-FSJ/datasets/demonstrations/I-FSJ/no_sep"
DATASET_NAME="mistral_demonstration_list_official"
# OUTPUT_DIR="${DATA_DIR}/wo_chat_template"
OUTPUT_DIR="${DATA_DIR}/w_chat_template/sys_msg_v0"

python Superfiltering/code_ifd/put_analysis_to_data.py \
    --pt_data_path ${OUTPUT_DIR}/${DATASET_NAME}_${MODEL_TYPE}_ppl.jsonl \
    --json_data_path ${DATA_DIR}/${DATASET_NAME}.json \
    --json_save_path ${OUTPUT_DIR}/${DATASET_NAME}_${MODEL_TYPE}_ifd.json

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v0"
SUBSET_NAME="harmful_behaviors_ifd_0.0_0.2"
BENCHMARK_NAME="AdvBench/${SUBSET_NAME}"

# demo_version_choices=(demo_v1 demo_v2 demo_v3 demo_v4)
# demo_version_choices=(demo_v4.0 demo_v5.0)
# demo_version_choices=(demo_v1.0 demo_v1.1 demo_v1.2 demo_v1.3 demo_v2.0 demo_v2.1 demo_v2.2 demo_v2.3)
demo_version_choices=(demo_v3.1 demo_v3.2 demo_v3.3 demo_v4.0 demo_v5.0)
# demo_version_choices=(demo_v3.0)
# demo_version_choices=(demo_v1.0 demo_v1.1 demo_v1.2 demo_v1.3)
# demo_version_choices=(demo_v2.0 demo_v2.1 demo_v2.2 demo_v2.3)
# demo_version_choices=(demo_v3.0 demo_v3.1 demo_v3.2 demo_v3.3)
# demo_version_choices=(demo_v1.2 demo_v1.3 demo_v2.2 demo_v2.3 demo_v3.2 demo_v3.3)

declare -A demo_path_dict
declare -A demo_embed_path_dict

demo_path_dict=(
    [demo_v0]="${DEMO_DIR}/ppl_2.0_12.0/demo_v0/filtered_ifd_0.4_1.2.json"
    [demo_v1]="${DEMO_DIR}/ppl_2.0_12.0/demo_v1/filtered_ifd_0.4_0.6.json"
    [demo_v2]="${DEMO_DIR}/ppl_2.0_12.0/demo_v2/filtered_ifd_0.6_0.8.json"
    [demo_v3]="${DEMO_DIR}/ppl_2.0_12.0/demo_v3/filtered_ifd_0.8_1.0.json"
    [demo_v4]="${DEMO_DIR}/ppl_2.0_12.0/demo_v4/filtered_ifd_1.0_1.2.json"
    [demo_v1.0]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.0/filtered_ifd_0.4_1.0.json"
    [demo_v1.1]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.1/filtered_ifd_0.4_0.6.json"
    [demo_v1.2]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.2/filtered_ifd_0.6_0.8.json"
    [demo_v1.3]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.3/filtered_ifd_0.8_1.0.json"
    [demo_v2.0]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.0/filtered_ifd_0.4_1.0.json"
    [demo_v2.1]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.1/filtered_ifd_0.4_0.6.json"
    [demo_v2.2]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.2/filtered_ifd_0.6_0.8.json"
    [demo_v2.3]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.3/filtered_ifd_0.8_1.0.json"
    [demo_v3.0]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.0/filtered_ifd_0.4_1.0.json"
    [demo_v3.1]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.1/filtered_ifd_0.4_0.6.json"
    [demo_v3.2]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.2/filtered_ifd_0.6_0.8.json"
    [demo_v3.3]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.3/filtered_ifd_0.8_1.0.json"
    [demo_v4.0]="${DEMO_DIR}/ppl_c_10.0_12.0/demo_v4.0/filtered_ifd_0.4_1.0.json"
    [demo_v5.0]="${DEMO_DIR}/ppl_c_12.0_14.0/demo_v5.0/filtered_ifd_0.4_1.0.json"
)
demo_embed_path_dict=(
    [demo_v0]="${DEMO_DIR}/ppl_2.0_12.0/demo_v0/instruction_embed_arr.npy"
    [demo_v1]="${DEMO_DIR}/ppl_2.0_12.0/demo_v1/instruction_embed_arr.npy"
    [demo_v2]="${DEMO_DIR}/ppl_2.0_12.0/demo_v2/instruction_embed_arr.npy"
    [demo_v3]="${DEMO_DIR}/ppl_2.0_12.0/demo_v3/instruction_embed_arr.npy"
    [demo_v4]="${DEMO_DIR}/ppl_2.0_12.0/demo_v4/instruction_embed_arr.npy"
    [demo_v1.0]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.0/instruction_embed_arr.npy"
    [demo_v1.1]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.1/instruction_embed_arr.npy"
    [demo_v1.2]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.2/instruction_embed_arr.npy"
    [demo_v1.3]="${DEMO_DIR}/ppl_c_4.0_6.0/demo_v1.3/instruction_embed_arr.npy"
    [demo_v2.0]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.0/instruction_embed_arr.npy"
    [demo_v2.1]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.1/instruction_embed_arr.npy"
    [demo_v2.2]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.2/instruction_embed_arr.npy"
    [demo_v2.3]="${DEMO_DIR}/ppl_c_6.0_8.0/demo_v2.3/instruction_embed_arr.npy"
    [demo_v3.0]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.0/instruction_embed_arr.npy"
    [demo_v3.1]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.1/instruction_embed_arr.npy"
    [demo_v3.2]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.2/instruction_embed_arr.npy"
    [demo_v3.3]="${DEMO_DIR}/ppl_c_8.0_10.0/demo_v3.3/instruction_embed_arr.npy"
    [demo_v4.0]="${DEMO_DIR}/ppl_c_10.0_12.0/demo_v4.0/instruction_embed_arr.npy"
    [demo_v5.0]="${DEMO_DIR}/ppl_c_12.0_14.0/demo_v5.0/instruction_embed_arr.npy"
)

# num_shots_choices=(2 4 6 8 10 12 14 16)
# num_shots_choices=(10 12 14 16)
num_shots_choices=(8)

declare -A time_cost_dict
time_cost_dict=(
    [2]=5
    [4]=15
    [6]=30
    [8]=40
    [10]=50
    [12]=70
    [14]=90
    [16]=110
    [20]=205
    [24]=205
)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python IFD-FSJ/src/select_demos.py \
            --selector_mode "greedy" \
            --wrt_adv_prefix \
            --benchmark_name ${BENCHMARK_NAME} \
            --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}.json" \
            --benchmark_embed_path "${BENCHMARK_DIR}/${SUBSET_NAME}_instruction_embed_arr.npy" \
            --demo_version ${demo_version} \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --demo_embed_path ${demo_embed_path_dict[${demo_version}]} \
            --output_dir "IFD-FSJ/evaluation" \
            --model_name ${MODEL_NAME} \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --num_shots ${num_shots} \
            --sim_threshold 0.5 \
            --lower_value_threshold -1 \
            --upper_value_threshold 0.04 \
            --relax_ratio 0.2 \
            --num_cands_per_attempt 16 \
            --max_num_attempts 1 \
            > IFD-FSJ/log/log_1.out 2>&1 &

        sleep ${time_cost_dict[${num_shots}]}m
        # sleep 1m
    done
done
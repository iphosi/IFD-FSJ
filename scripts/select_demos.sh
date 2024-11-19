export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

# DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0"
SUBSET_NAME="harmful_behaviors_subset_50"
BENCHMARK_NAME="AdvBench/${SUBSET_NAME}"

# demo_version_choices=(demo_v0 demo_v1 demo_v2 demo_v3 demo_v4)
demo_version_choices=(demo_v3)

declare -A demo_path_dict
declare -A demo_embed_path_dict

demo_path_dict=(
    [demo_v0]="${DEMO_DIR}/filtered_llama2_ifd_0.0_0.4_fla_577_256.json"
    [demo_v1]="${DEMO_DIR}/filtered_llama2_ifd_0.4_0.6_fla_2969_256.json"
    [demo_v2]="${DEMO_DIR}/filtered_llama2_ifd_0.6_0.8_fla_12689_256.json"
    [demo_v3]="${DEMO_DIR}/filtered_llama2_ifd_0.8_1.0_fla_8642_256.json"
    [demo_v4]="${DEMO_DIR}/filtered_llama2_ifd_1.0_inf_fla_1239_256.json"
    [demo_v5]="${DEMO_DIR}/filtered_llama2_ifd_0.0_inf_fla_26116_256.json"
)
demo_embed_path_dict=(
    [demo_v0]="${DEMO_DIR}/filtered_llama2_ifd_0.0_0.4_fla_577_256_instruction_embed_arr.npy"
    [demo_v1]="${DEMO_DIR}/filtered_llama2_ifd_0.4_0.6_fla_2969_256_instruction_embed_arr.npy"
    [demo_v2]="${DEMO_DIR}/filtered_llama2_ifd_0.6_0.8_fla_12689_256_instruction_embed_arr.npy"
    [demo_v3]="${DEMO_DIR}/filtered_llama2_ifd_0.8_1.0_fla_8642_256_instruction_embed_arr.npy"
    [demo_v4]="${DEMO_DIR}/filtered_llama2_ifd_1.0_inf_fla_1239_256_instruction_embed_arr.npy"
    [demo_v5]="${DEMO_DIR}/filtered_llama2_ifd_0.0_inf_fla_26116_256_instruction_embed_arr.npy"
)

# num_shots_choices=(2 4 6 8 10 12 14 16)
num_shots_choices=(6)

declare -A time_cost_dict
time_cost_dict=(
    [2]=5
    [4]=15
    [6]=80
    [8]=120
    [10]=75
    [12]=115
    [14]=165
    [16]=205
)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python IFD-FSJ/src/select_demos.py \
            --selector_mode "ifd_rejection" \
            --benchmark_name ${BENCHMARK_NAME} \
            --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}_llama2_ifd.json" \
            --benchmark_embed_path "${BENCHMARK_DIR}/${SUBSET_NAME}_llama2_ifd_instruction_embed_arr.npy" \
            --demo_version ${demo_version} \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --demo_embed_path ${demo_embed_path_dict[${demo_version}]} \
            --output_dir "IFD-FSJ/evaluation" \
            --model_name ${MODEL_NAME} \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --num_shots ${num_shots} \
            --sim_threshold 0.5 \
            --ifd_drop_threshold 0.1 \
            --relax_ratio 0.2 \
            --max_num_attempts 8 \
            > IFD-FSJ/log/log_0.out 2>&1 &

        sleep ${time_cost_dict[${num_shots}]}m
    done
done
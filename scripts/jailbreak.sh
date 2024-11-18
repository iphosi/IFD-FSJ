export CUDA_VISIBLE_DEVICES=6,7,8,9
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
# demo_version_choices=(demo_v6 demo_v7 demo_v8 demo_v9)
demo_version_choices=(demo_v3)
# demo_version_choices=(demo_v9)

declare -A demo_path_dict

# demo_path_dict=(
#     [demo_v0]="${DEMO_DIR}/filtered_llama3_ifd_0.0_0.4_fla_573_256.json"
#     [demo_v1]="${DEMO_DIR}/filtered_llama3_ifd_0.4_0.6_fla_2316_256.json"
#     [demo_v2]="${DEMO_DIR}/filtered_llama3_ifd_0.6_0.8_fla_8992_256.json"
#     [demo_v3]="${DEMO_DIR}/filtered_llama3_ifd_0.8_1.0_fla_8077_256.json"
# )

demo_path_dict=(
    [demo_v0]="${DEMO_DIR}/filtered_llama2_ifd_0.0_0.4_fla_577_256.json"
    [demo_v1]="${DEMO_DIR}/filtered_llama2_ifd_0.4_0.6_fla_2969_256.json"
    [demo_v2]="${DEMO_DIR}/filtered_llama2_ifd_0.6_0.8_fla_12689_256.json"
    [demo_v3]="${DEMO_DIR}/filtered_llama2_ifd_0.8_1.0_fla_8642_256.json"
    [demo_v4]="${DEMO_DIR}/filtered_llama2_ifd_1.0_inf_fla_1239_256.json"
    [demo_v5]="${DEMO_DIR}/filtered_llama2_ifd_0.0_0.4_fla_333_256.json"
    [demo_v6]="${DEMO_DIR}/filtered_llama2_ifd_0.4_0.6_fla_1832_256.json"
    [demo_v7]="${DEMO_DIR}/filtered_llama2_ifd_0.6_0.8_fla_9169_256.json"
    [demo_v8]="${DEMO_DIR}/filtered_llama2_ifd_0.8_1.0_fla_8063_256.json"
    [demo_v9]="${DEMO_DIR}/filtered_llama2_ifd_1.0_inf_fla_1471_256.json"
)

# num_shots_choices=(4)
num_shots_choices=(2 4 6 8)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python IFD-FSJ/src/jailbreak.py \
            --selector_mode "ifd_rejection" \
            --compute_adv_prompt_ifd \
            --use_adv_prompt \
            --benchmark_name ${BENCHMARK_NAME} \
            --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}_llama2_ifd.json" \
            --demo_version ${demo_version} \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --output_dir "IFD-FSJ/evaluation" \
            --model_name ${MODEL_NAME} \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --num_shots ${num_shots} \
            --num_return_sequences 8 \
            --temperature 1.0 \
            --top_p 1.0 \
            --max_tokens 100 \
            > IFD-FSJ/log/log_1.out 2>&1 &

        sleep 2m
    done
done
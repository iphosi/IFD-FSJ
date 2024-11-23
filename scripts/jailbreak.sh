export CUDA_VISIBLE_DEVICES=6,7,8,9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

# DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v0"
SUBSET_NAME="harmful_behaviors_subset_50"
BENCHMARK_NAME="AdvBench/${SUBSET_NAME}"

# demo_version_choices=(demo_v0 demo_v1 demo_v2 demo_v3 demo_v4)
demo_version_choices=(demo_v4)

declare -A demo_path_dict

demo_path_dict=(
    [demo_v0]="${DEMO_DIR}/demo_v0/filtered_ifd_0.4_1.2.json"
    [demo_v1]="${DEMO_DIR}/demo_v1/filtered_ifd_0.4_0.6.json"
    [demo_v2]="${DEMO_DIR}/demo_v2/filtered_ifd_0.6_0.8.json"
    [demo_v3]="${DEMO_DIR}/demo_v3/filtered_ifd_0.8_1.0.json"
    [demo_v4]="${DEMO_DIR}/demo_v4/filtered_ifd_1.0_1.2.json"
)

# num_shots_choices=(2 4 6 8 10 12 14 16)
num_shots_choices=(8)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python IFD-FSJ/src/jailbreak.py \
            --selector_mode "greedy" \
            --wrt_adv_prefix \
            --benchmark_name ${BENCHMARK_NAME} \
            --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}.json" \
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
            > IFD-FSJ/log/log_3.out 2>&1 &

        sleep 2m
    done
done
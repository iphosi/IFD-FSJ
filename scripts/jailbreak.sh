export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

# DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v1"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v1"
SUBSET_NAME="harmful_behaviors_ifd_0.0_0.2"
BENCHMARK_NAME="AdvBench/llama2/w_chat_template/sys_msg_v1/${SUBSET_NAME}"

# demo_version_choices=(demo_v0 demo_v1 demo_v2 demo_v3 demo_v4)
demo_version_choices=(demo_v3.1)

declare -A demo_path_dict

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

# num_shots_choices=(2 4 6 8 10 12 14 16)
num_shots_choices=(6)

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
            --system_message_version v1 \
            > IFD-FSJ/log/log_0.out 2>&1 &

        sleep 2m
    done
done
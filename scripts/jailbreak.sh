export CUDA_VISIBLE_DEVICES=2,3,6,9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Meta-Llama-3-8B-Instruct"
# MODEL_TYPE="llama3"
MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"
MODEL_TYPE="llama2"
# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Qwen2.5-7B-Instruct"
# MODEL_TYPE="qwen2.5"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_v0"
SUBSET_NAME="harmful_behaviors_subset"
# SUBSET_NAME="harmful_behaviors_ppl_c_0.0_9.0"
# SUBSET_NAME="harmful_behaviors_ppl_c_0.0_3.0"
BENCHMARK_NAME="AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_v0/${SUBSET_NAME}"

# BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/I-FSJ/${MODEL_TYPE}/w_chat_template/sys_msg_v0"
# # SUBSET_NAME="mistral_demonstration"
# SUBSET_NAME="instruction"
# BENCHMARK_NAME="I-FSJ/${MODEL_TYPE}/w_chat_template/sys_msg_v0/${SUBSET_NAME}"

demo_version_choices=(demo_v5.1)
# demo_version_choices=(demo_v5)

declare -A demo_path_dict

demo_path_dict=(
    [demo_v0]="IFD-FSJ/datasets/demonstrations"
    [demo_v5.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.1/filtered.json"
)

# num_shots_choices=(2 4 6 8 10 12 14 16)
num_shots_choices=(2)

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
            --num_return_sequences 16 \
            --temperature 1.0 \
            --top_p 1.0 \
            --max_tokens 100 \
            --system_message_version v0 \
            > IFD-FSJ/log/log_0.out 2>&1 &

        sleep 3m
    done
done
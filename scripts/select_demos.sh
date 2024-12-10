export CUDA_VISIBLE_DEVICES=6,9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"
MODEL_TYPE="llama2"

# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Qwen2.5-7B-Instruct"
# MODEL_TYPE="qwen2.5"

SYSTEM_MESSAGE_VERSION="v0"

BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}"
SUBSET_NAME="harmful_behaviors_subset"
BENCHMARK_NAME="AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}/${SUBSET_NAME}"

# BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/AdvBench/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}"
# SUBSET_NAME="harmful_behaviors_ppl_c_0.0_3.0"
# BENCHMARK_NAME="AdvBench/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}/${SUBSET_NAME}"

# BENCHMARK_DIR="IFD-FSJ/datasets/benchmarks/I-FSJ/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}"
# SUBSET_NAME="instruction"
# BENCHMARK_NAME="I-FSJ/${MODEL_TYPE}/w_chat_template/sys_msg_${SYSTEM_MESSAGE_VERSION}/${SUBSET_NAME}"

# demo_version_choices=(demo_v1 demo_v2 demo_v3 demo_v4)
demo_version_choices=(demo_v5.1)

declare -A demo_path_dict
declare -A demo_embed_path_dict

demo_path_dict=(
    [demo_v2.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V2-Llama2-t50/llama2_sep_3/llama2/w_chat_template/sys_msg_v0/demo_v2.1/filtered.json"
    [demo_v2]="IFD-FSJ/datasets/demonstrations/AdvBench-V2-Llama2-t200/llama2/w_chat_template/sys_msg_v0/demo_v2/filtered_ppl_c_0.0_4.0.json"
    [demo_v4.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V4-Llama2-t400/llama2_sep_1/llama2/w_chat_template/sys_msg_v0/demo_v4.1/filtered.json"
    [demo_v4.2]="IFD-FSJ/datasets/demonstrations/AdvBench-V4-Llama2-t400/llama2_sep_4/llama2/w_chat_template/sys_msg_v0/demo_v4.2/filtered.json"
    [demo_v5.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.1/filtered.json"
    [demo_v6.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V6-Llama2-t50/no_sep/llama2/w_chat_template/sys_msg_v0/demo_v6.1/filtered.json"
    [demo_v10]="IFD-FSJ/datasets/demonstrations/I-FSJ/llama2_sep/llama2/w_chat_template/sys_msg_v0/mistral_demonstration_list_official.json"
)
demo_embed_path_dict=(
    [demo_v2.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V2-Llama2-t50/llama2_sep_3/llama2/w_chat_template/sys_msg_v0/demo_v2.1/instruction_embed_arr.npy"
    [demo_v2]="IFD-FSJ/datasets/demonstrations/AdvBench-V2-Llama2-t200/llama2/w_chat_template/sys_msg_v0/demo_v2/instruction_embed_arr.npy"
    [demo_v4.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V4-Llama2-t400/llama2_sep_1/llama2/w_chat_template/sys_msg_v0/demo_v4.1/instruction_embed_arr.npy"
    [demo_v4.2]="IFD-FSJ/datasets/demonstrations/AdvBench-V4-Llama2-t400/llama2_sep_4/llama2/w_chat_template/sys_msg_v0/demo_v4.2/instruction_embed_arr.npy"
    [demo_v5.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.1/instruction_embed_arr.npy"
    [demo_v6.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V6-Llama2-t50/no_sep/llama2/w_chat_template/sys_msg_v0/demo_v6.1/instruction_embed_arr.npy"
    [demo_v10]="IFD-FSJ/datasets/demonstrations/I-FSJ/llama2_sep/llama2/w_chat_template/sys_msg_v0/instruction_embed_arr.npy"
)

# num_shots_choices=(2 4 6 8 10 12 14 16)
num_shots_choices=(8)

declare -A time_cost_dict
time_cost_dict=(
    [2]=15
    [4]=40
    [6]=120
    [8]=180
    [10]=170
    [12]=200
    [14]=220
    [16]=250
    [20]=205
    [24]=205
)
# time_cost_dict=(
#     [2]=10
#     [4]=25
#     [6]=35
#     [8]=45
#     [10]=60
#     [12]=80
#     [14]=100
#     [16]=120
#     [20]=205
#     [24]=205
# )

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
            --sim_threshold 1 \
            --lower_value_threshold -1 \
            --upper_value_threshold 0.04 \
            --relax_ratio 0.2 \
            --num_cands_per_attempt 64 \
            --max_num_attempts 1 \
            --system_message_version ${SYSTEM_MESSAGE_VERSION} \
            > IFD-FSJ/log/log_0.out 2>&1 &

        # sleep ${time_cost_dict[${num_shots}]}m
        sleep 1m
    done
done

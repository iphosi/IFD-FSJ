export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_TYPE="llama2"
# MODEL_NAME="Meta-Llama-3-8B-Instruct"
# MODEL_TYPE="llama3"
# MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
# MODEL_TYPE="llama3.1"
# MODEL_NAME="OpenChat-3.6-8B"
# MODEL_TYPE="openchat3.6"
# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_TYPE="qwen2.5"
# MODEL_NAME="Starling-LM-7B-beta"
# MODEL_TYPE="starlinglm"

MODEL_PATH="Self-Instruct-FSJ/models/${MODEL_NAME}"

ADV_PREFIX_PATH="Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_v0/harmful_behaviors_subset.json"

DEMO_DATA_DIR="Self-Instruct-FSJ/evaluation/${MODEL_NAME}/AdvBench-V5/${MODEL_TYPE}/w_chat_template/sys_msg_v0/harmful_behaviors_subset/random/demonstrations"

demo_version_choices=(demo_v5.1.1)

declare -A demo_path_dict

demo_path_dict=(
    [demo_v5.1.1]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.1.1/filtered.json"
    [demo_v5.1.2]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama3/w_chat_template/sys_msg_v0/demo_v5.1.2/filtered.json"
    [demo_v5.1.3]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama3.1/w_chat_template/sys_msg_v0/demo_v5.1.3/filtered.json"
    [demo_v5.1.4]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/openchat3.6/w_chat_template/sys_msg_v0/demo_v5.1.4/filtered.json"
    [demo_v5.1.5]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/qwen2.5/w_chat_template/sys_msg_v0/demo_v5.1.5/filtered.json"
    [demo_v5.1.6]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/starlinglm/w_chat_template/sys_msg_v0/demo_v5.1.6/filtered.json"
    [demo_v5.2.2]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3-t50/llama3/w_chat_template/sys_msg_v0/demo_v5.2.2/filtered.json"
    [demo_v5.3.3]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Llama3.1-t50/llama3.1/w_chat_template/sys_msg_v0/demo_v5.3.3/filtered.json"
    [demo_v5.4.4]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-OpenChat3.6-t50/openchat3.6/w_chat_template/sys_msg_v0/demo_v5.4.4/filtered.json"
    [demo_v5.5.5]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Qwen2.5-t50/qwen2.5/w_chat_template/sys_msg_v0/demo_v5.5.5/filtered.json"
    [demo_v5.6.6]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-StarlingLM-t50/starlinglm/w_chat_template/sys_msg_v0/demo_v5.6.6/filtered.json"
    [demo_v5.5.1]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Qwen2.5-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.5.1/filtered.json"
    [demo_v5.6.1]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-StarlingLM-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.6.1/filtered.json"
    [demo_v6.1.1]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V6-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v6.1.1/filtered.json"
    [demo_v6.2.2]="Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V6-Llama3-t50/llama3/w_chat_template/sys_msg_v0/demo_v6.2.2/filtered.json"
)

num_shots_choices=(2)

declare -A time_cost_dict
time_cost_dict=(
    [2]=2
    [4]=4
    [8]=8
)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python Self-Instruct-FSJ/src/get_in_context_arr.py \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --demo_version "${demo_version}" \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --demo_data_dir ${DEMO_DATA_DIR} \
            --num_shots ${num_shots} \
            --adv_prefix_path ${ADV_PREFIX_PATH} \
            --wrt_adv_prefix \
            > Self-Instruct-FSJ/log/log_0.out 2>&1 &

        sleep ${time_cost_dict[${num_shots}]}m
        # sleep 1m
    done
done

export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

# MODEL_NAME="Qwen2.5-7B-Instruct"
# MODEL_PATH="IFD-FSJ/models/Qwen2.5-7B-Instruct"

ADV_PREFIX_PATH="IFD-FSJ/datasets/benchmarks/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset.json"

DEMO_DATA_DIR="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/random/demonstrations"

# demo_version_choices=(demo_v3 demo_v4)
demo_version_choices=(demo_v5.1)
# demo_version_choices=(demo_v4 demo_v3)

declare -A demo_path_dict

demo_path_dict=(
    [demo_v5.1]="IFD-FSJ/datasets/demonstrations/AdvBench-V5-Llama2-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.1/filtered.json"
)

# num_shots_choices=(2 3 4 5 6 7 8)
num_shots_choices=(8)

declare -A time_cost_dict
time_cost_dict=(
    [2]=1
    [4]=3
    [6]=7
    [8]=10
    [10]=20
    [12]=30
    [14]=40
    [16]=80
)

for num_shots in "${num_shots_choices[@]}";
do
    for demo_version in "${demo_version_choices[@]}";
    do
        echo "Demo version: ${demo_version}"
        echo "Demo path: ${demo_path_dict[${demo_version}]}"
        echo "Num shots: ${num_shots}"
        python IFD-FSJ/src/get_in_context_arr.py \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --demo_version ${demo_version} \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --demo_data_dir ${DEMO_DATA_DIR} \
            --num_shots ${num_shots} \
            --adv_prefix_path ${ADV_PREFIX_PATH} \
            --wrt_adv_prefix \
            > IFD-FSJ/log/log_1.out 2>&1 &

        # sleep ${time_cost_dict[${num_shots}]}m
        sleep 1m
    done
done

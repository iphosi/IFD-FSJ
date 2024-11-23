export CUDA_VISIBLE_DEVICES=8,9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

DEMO_DATA_DIR="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors_subset_50/random/demonstrations"
DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0"

ADV_PROMPT_PATH="IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset_50.json"

# demo_version_choices=(demo_v3 demo_v4)
demo_version_choices=(demo_v3)
# demo_version_choices=(demo_v4 demo_v3)

declare -A demo_path_dict

demo_path_dict=(
    [demo_v0]="${DEMO_DIR}/ppl_0.0_10.0/demo_v0/filtered_ifd_0.4_1.2.json"
    [demo_v1]="${DEMO_DIR}/ppl_0.0_10.0/demo_v1/filtered_ifd_0.4_0.6.json"
    [demo_v2]="${DEMO_DIR}/ppl_0.0_10.0/demo_v2/filtered_ifd_0.6_0.8.json"
    [demo_v3]="${DEMO_DIR}/ppl_0.0_10.0/demo_v3/filtered_ifd_0.8_1.0.json"
    [demo_v4]="${DEMO_DIR}/ppl_0.0_10.0/demo_v4/filtered_ifd_1.0_1.2.json"
)

# num_shots_choices=(2 3 4 5 6 7 8)
num_shots_choices=(16)

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
            --adv_prefix_path ${ADV_PROMPT_PATH} \
            --wrt_adv_prefix \
            > IFD-FSJ/log/log_6.out 2>&1 &

        sleep ${time_cost_dict[${num_shots}]}m
    done
done

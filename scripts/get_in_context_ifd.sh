export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_PATH="IFD-FSJ/models/Llama-2-7b-chat-hf"

DATA_DIR="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/random/w_adv_prompt"
DEMO_DIR="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0"

# demo_version_choices=(demo_v0 demo_v1 demo_v2 demo_v3 demo_v4)
demo_version_choices=(demo_v2)

declare -A demo_path_dict
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

# num_shots_choices=(2 3 4 5 6 7 8)
num_shots_choices=(8)

declare -A time_cost_dict
time_cost_dict=(
    [2]=5
    [3]=15
    [4]=25
    [5]=35
    [6]=45
    [7]=55
    [8]=65
)

for demo_version in "${demo_version_choices[@]}";
do
    for num_shots in "${num_shots_choices[@]}";
    do
        echo ${demo_version}
        echo ${demo_path_dict[${demo_version}]}
        echo ${num_shots}
        python IFD-FSJ/src/get_in_context_ifd.py \
            --model_path ${MODEL_PATH} \
            --max_length 4096 \
            --demo_version ${demo_version} \
            --demo_path ${demo_path_dict[${demo_version}]} \
            --data_dir ${DATA_DIR} \
            --num_shots ${num_shots} \
            --num_responses_per_instruction 4 \
            > IFD-FSJ/log/log_4.out 2>&1 &

        sleep ${time_cost_dict[${num_shots}]}m
    done
done